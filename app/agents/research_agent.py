import asyncio
import json
import logging
import time

from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from typing import Any

from app.agents.tool_definitions import RESEARCH_TOOL_DEFINITIONS
from app.agents.tool_mapping import TOOL_NAME_MAPPING
from app.core.config import settings
from app.llm.base import LLMProvider
from app.llm.pricing import estimate_cost_usd
from app.tools.base import ToolResult
from app.tools.registry import ToolRegistry
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


AGENT_INSTRUCTIONS = """
You are a research assistant.

Use the available search tools to gather evidence for the user's research
question, then answer it.

Rules:

- Search before answering; do not rely only on prior knowledge.
- Prefer specific search queries instead of copying the question blindly.
- Avoid duplicate searches.
- When you have enough evidence, stop calling tools and write the answer.
- Base the answer on the search results. Do not invent facts that are not
  supported by the research, and say when the evidence is limited.
"""


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    # USD, from the model's prices in app/llm/pricing.py (cached input
    # tokens are priced per call, in _extract_usage). None when a call's
    # model has no known pricing, since any total would then be too low.
    estimated_cost: float | None = 0.0


def format_cost(cost_usd: float | None) -> str:
    return "unknown" if cost_usd is None else f"${cost_usd:.6f}"


@dataclass
class AgentResult:
    answer: str
    tool_results: list[ToolResult]
    # Number of LLM calls made (iterations).
    iteration_count: int
    # Number of tool calls the model requested, including invalid ones
    # (unknown tool, bad arguments) that never ran.
    tool_call_count: int
    # Tokens and estimated cost of every LLM call in the run.
    usage: TokenUsage
    success: bool = True
    error: str | None = None


@dataclass
class ToolExecution:
    result: ToolResult
    # Wall-clock time for the whole search, including retries and the waits
    # between them.
    duration_ms: float
    # Searches actually started (0 if the time limit had already passed).
    attempts: int


def _elapsed_ms(started_at: float) -> float:
    return (time.perf_counter() - started_at) * 1000


# Bump when the checkpoint layout changes, so old checkpoints can be
# recognised (and refused) instead of being misread. 2: the flat layout.
CHECKPOINT_VERSION = 2


@dataclass
class AgentCheckpoint:
    """Everything needed to resume a run safely after its last iteration.

    Stored (as JSON) in agent_runs.state after every iteration, so a
    resumed run continues with the same conversation, limits and totals.
    """

    # Iterations (LLM calls) finished so far; a resume starts at the next.
    iteration: int
    # The conversation so far, in Responses API input format: the question,
    # each iteration's model output items, and a function_call_output per
    # tool call.
    input_items: list[dict[str, Any]]
    # Tool calls the model requested, including invalid ones.
    tool_call_count: int
    # Token usage and estimated cost so far (the token budget carries over).
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost: float | None
    # Searches actually run, counted against max_tool_calls. Can be lower
    # than tool_call_count (invalid or over-limit calls don't run); without
    # it, a resume would get a fresh search limit.
    executed_tool_calls: int
    version: int = CHECKPOINT_VERSION

    @property
    def usage(self) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=self.total_tokens,
            estimated_cost=self.estimated_cost,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> "AgentCheckpoint":
        """Rebuilds and checks a checkpoint from stored data.

        Stored state can be corrupted or edited by hand, and resuming from a
        bad one would send the model a broken conversation or wrong limits,
        so every field is checked and any problem raises ValueError.
        """

        if not isinstance(data, dict):
            raise ValueError(
                "Invalid agent checkpoint: expected an object, "
                f"got {type(data).__name__}."
            )

        # Missing version / executed_tool_calls: written without them, so
        # assume the current layout, and that every requested call ran (the
        # safe direction: it can only leave fewer searches, never more).
        version = data.get("version", CHECKPOINT_VERSION)

        if version != CHECKPOINT_VERSION:
            raise ValueError(
                f"Unsupported agent checkpoint version {version!r} "
                f"(expected {CHECKPOINT_VERSION})."
            )

        required_fields = {
            "iteration",
            "input_items",
            "tool_call_count",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "estimated_cost",
        }

        missing = required_fields - data.keys()

        if missing:
            raise ValueError(
                f"Checkpoint missing fields: {sorted(missing)}"
            )

        executed_tool_calls = data.get(
            "executed_tool_calls",
            data["tool_call_count"],
        )

        for name, value in [
            ("iteration", data["iteration"]),
            ("tool_call_count", data["tool_call_count"]),
            ("executed_tool_calls", executed_tool_calls),
            ("input_tokens", data["input_tokens"]),
            ("output_tokens", data["output_tokens"]),
            ("total_tokens", data["total_tokens"]),
        ]:
            # bool is a subclass of int, so True would otherwise pass as 1.
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(
                    f"Checkpoint {name} must be an integer."
                )

            if value < 0:
                raise ValueError(
                    f"Checkpoint {name} must not be negative."
                )

        # Checkpoints are only taken after a finished iteration.
        if data["iteration"] < 1:
            raise ValueError(
                "Checkpoint iteration must be at least 1."
            )

        # Every search that ran was a requested tool call.
        if executed_tool_calls > data["tool_call_count"]:
            raise ValueError(
                "Checkpoint executed_tool_calls must not be greater than "
                "tool_call_count."
            )

        estimated_cost = data["estimated_cost"]

        # null means the model's pricing is unknown.
        if estimated_cost is not None and (
            not isinstance(estimated_cost, (int, float))
            or isinstance(estimated_cost, bool)
        ):
            raise ValueError(
                "Checkpoint estimated_cost must be numeric."
            )

        if estimated_cost is not None and estimated_cost < 0:
            raise ValueError(
                "Checkpoint estimated_cost must not be negative."
            )

        input_items = data["input_items"]

        if not isinstance(input_items, list):
            raise ValueError(
                "Checkpoint input_items must be a list."
            )

        # At least the question, and every item a conversation object.
        if not input_items:
            raise ValueError(
                "Checkpoint input_items must not be empty."
            )

        for index, item in enumerate(input_items):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Checkpoint input_items[{index}] must be an object."
                )

        return cls(
            iteration=data["iteration"],
            input_items=input_items,
            tool_call_count=data["tool_call_count"],
            input_tokens=data["input_tokens"],
            output_tokens=data["output_tokens"],
            total_tokens=data["total_tokens"],
            estimated_cost=(
                float(estimated_cost)
                if estimated_cost is not None
                else None
            ),
            executed_tool_calls=executed_tool_calls,
            version=version,
        )


def serialize_input_item(item: Any) -> dict[str, Any]:
    """One conversation item as a plain dict.

    The question and tool outputs are already dicts. The model's output items
    are OpenAI SDK (pydantic) objects; model_dump gives the same fields the
    SDK sends when they are passed back as input. Test fakes are dataclasses.
    """

    if isinstance(item, dict):
        return item

    if hasattr(item, "model_dump"):
        return item.model_dump(mode="json", exclude_none=True)

    if is_dataclass(item):
        return asdict(item)

    raise TypeError(f"Can't checkpoint input item of type {type(item).__name__}")


# on_tool_call(tool, query, iteration) runs before a search and may return a
# step object (e.g. a saved ResearchStep); on_tool_result(step, execution)
# gets that same object back with the search result and its timing.
OnToolCall = Callable[[str, str, int], Awaitable[Any]]
OnToolResult = Callable[[Any, ToolExecution], Awaitable[None]]
# on_checkpoint(checkpoint) runs after each finished iteration, e.g. to
# save it. checkpoint is _serialize_checkpoint() output: a JSON string;
# _deserialize_checkpoint() turns it back.
OnCheckpoint = Callable[[str], Awaitable[None]]


class AgentPhase(str, Enum):
    """What the agent is doing, reported through on_status_change. The
    values match AgentRunStatus, so the workflow can store them as-is."""

    # Calling the LLM.
    RUNNING = "running"
    # The model asked for searches; they are running.
    WAITING_FOR_TOOL = "waiting_for_tool"
    # Searches finished; results are being saved and added to the
    # conversation.
    PROCESSING_RESULT = "processing_result"


# on_status_change(phase) runs whenever the agent moves to another phase.
OnStatusChange = Callable[[AgentPhase], Awaitable[None]]


class TokenBudgetExceeded(RuntimeError):
    """Raised by _check_token_budget; run() turns it into a failed result."""


class ResearchAgent:

    # Limits not passed to __init__ come from settings (AGENT_* environment
    # variables); the defaults are shown in brackets.

    # Time limit for a whole run. Once it passes, no more LLM calls or
    # searches start, and running ones are cut short; the run fails. [300]
    MAX_EXECUTION_SECONDS = settings.agent_max_execution_seconds

    # Wait before each retry: RETRY_BACKOFF_SECONDS * 2 ** (attempt - 1),
    # i.e. 1s, then 2s.
    RETRY_BACKOFF_SECONDS = 1.0

    def __init__(
        self,
        llm: LLMProvider,
        registry: ToolRegistry,
        on_tool_call: OnToolCall | None = None,
        on_tool_result: OnToolResult | None = None,
        on_checkpoint: OnCheckpoint | None = None,
        on_status_change: OnStatusChange | None = None,
        max_iterations: int = 6,
        max_tool_calls: int = 8,
        tool_timeout_seconds: float = 30.0,
        max_tool_retries: int = 2,
        max_query_length: int = 500,
        max_result_length: int = 12_000,
        max_total_tokens: int = 20_000,
    ):
        self.llm = llm
        self.registry = registry

        self.on_tool_call = on_tool_call
        self.on_tool_result = on_tool_result
        self.on_checkpoint = on_checkpoint
        self.on_status_change = on_status_change

        # Each iteration is one LLM call; this stops a model that keeps
        # calling tools from looping forever.
        self.max_iterations = max_iterations

        # Total searches per run. Calls past the limit are not run; the model
        # is told to answer, and is offered no tools from then on.
        self.max_tool_calls = max_tool_calls

        # A search that takes longer is cancelled and counts as a failure.
        self.tool_timeout_seconds = tool_timeout_seconds

        # Extra attempts after a failed or timed-out search.
        self.max_tool_retries = max_tool_retries

        # Longer search queries from the model are cut to this length.
        self.max_query_length = max_query_length

        # Tool output sent back to the model is cut to this length, to keep
        # the prompt small. The full result is still saved.
        self.max_result_length = max_result_length

        # Token budget for a whole run (input + output over all LLM calls).
        # Once it is reached, the agent makes no more LLM calls and runs no
        # more searches; a final answer that goes over is still returned.
        self.max_total_tokens = max_total_tokens

        # perf_counter() value when the current run must stop; None outside
        # run() (e.g. when _execute_tool is called directly).
        self._deadline: float | None = None

    async def run(
        self,
        question: str,
        checkpoint: AgentCheckpoint | dict[str, Any] | str | None = None,
    ) -> AgentResult:

        # With a checkpoint, carry on after its last iteration. The
        # conversation (which already starts with the question), counters and
        # token usage all carry over, so the iteration, tool-call and token
        # limits apply to the whole job, not to each attempt. tool_results
        # only holds this attempt's searches; earlier ones are in the
        # database. The time limit starts fresh.
        if checkpoint is None:

            input_items: list[Any] = [
                {
                    "role": "user",
                    "content": question,
                }
            ]

            iteration_count = 0
            tool_call_count = 0

            # Searches actually run, counted against max_tool_calls.
            executed_tool_calls = 0

            usage = TokenUsage()

        else:

            restored = self._deserialize_checkpoint(
                checkpoint
            )

            # A copy, so the run never changes the checkpoint it was given.
            input_items = list(restored.input_items)

            iteration_count = restored.iteration

            tool_call_count = restored.tool_call_count
            executed_tool_calls = restored.executed_tool_calls

            usage = TokenUsage(
                input_tokens=restored.input_tokens,
                output_tokens=restored.output_tokens,
                total_tokens=restored.total_tokens,
                estimated_cost=restored.estimated_cost,
            )

            logger.info(
                "Research agent resuming after iteration %d (%d searches, "
                "%d tokens used so far)",
                iteration_count,
                executed_tool_calls,
                usage.total_tokens,
            )

        tool_results: list[ToolResult] = []

        self._deadline = time.perf_counter() + self.MAX_EXECUTION_SECONDS

        while iteration_count < self.max_iterations:

            iteration_count += 1

            logger.info(
                "Agent iteration %s",
                iteration_count,
            )

            await self._emit_status(AgentPhase.RUNNING)

            # Once the search budget is used up, offer no tools so the model
            # has to answer with what it has.
            tools = (
                RESEARCH_TOOL_DEFINITIONS
                if executed_tool_calls < self.max_tool_calls
                else []
            )

            # Refuse to make another LLM call once the budget is used up.
            try:
                self._check_token_budget(usage)
            except TokenBudgetExceeded as exc:
                return self._over_budget_result(
                    exc,
                    tool_results,
                    iteration_count - 1,
                    tool_call_count,
                    usage,
                )

            # The time limit only cuts short the LLM call and the searches,
            # never the callbacks, so a database write isn't interrupted
            # halfway.
            try:
                response = await asyncio.wait_for(
                    self.llm.generate_with_tools(
                        input_items=input_items,
                        tools=tools,
                        instructions=AGENT_INSTRUCTIONS,
                    ),
                    timeout=self._time_left(),
                )
            except asyncio.TimeoutError:
                return self._timed_out_result(
                    tool_results,
                    iteration_count,
                    tool_call_count,
                    usage,
                )

            self._add_usage(
                usage,
                self._extract_usage(response),
            )

            function_calls = [
                item
                for item in response.output
                if item.type == "function_call"
            ]

            tool_call_count += len(function_calls)

            # No tool calls: the model has written its final answer.
            if not function_calls:

                # A final checkpoint, so the stored state covers the whole
                # run: every iteration, the answer, and the final usage.
                input_items.extend(response.output)

                await self._emit_checkpoint(
                    input_items=input_items,
                    iteration=iteration_count,
                    tool_call_count=tool_call_count,
                    executed_tool_calls=executed_tool_calls,
                    usage=usage,
                )

                logger.info(
                    "Research agent finished after %d iterations and %d tool "
                    "calls; tokens=%d cost=%s",
                    iteration_count,
                    len(tool_results),
                    usage.total_tokens,
                    format_cost(usage.estimated_cost),
                )

                return AgentResult(
                    answer=response.output_text,
                    tool_results=tool_results,
                    success=bool(response.output_text),
                    error=None if response.output_text else "Model returned an empty answer.",
                    iteration_count=iteration_count,
                    tool_call_count=tool_call_count,
                    usage=usage,
                )

            # Also check right after the call: if this response used up the
            # budget, its searches are not run either (each one would only
            # make the next LLM call bigger). Earlier results are kept.
            try:
                self._check_token_budget(usage)
            except TokenBudgetExceeded as exc:
                return self._over_budget_result(
                    exc,
                    tool_results,
                    iteration_count,
                    tool_call_count,
                    usage,
                )

            # The Responses API needs the model's function_call items in the
            # input, followed by one function_call_output per call_id.
            input_items.extend(response.output)

            # (tool, query) for each valid call, None for invalid ones.
            parsed_calls = [
                self._parse_call(call)
                for call in function_calls
            ]

            # Valid calls past max_tool_calls are skipped (not run).
            over_limit = []

            for index, parsed in enumerate(parsed_calls):

                if parsed is None:
                    over_limit.append(False)
                    continue

                if executed_tool_calls >= self.max_tool_calls:
                    parsed_calls[index] = None
                    over_limit.append(True)
                    continue

                executed_tool_calls += 1
                over_limit.append(False)

            if any(over_limit):
                logger.warning(
                    "Tool call limit reached (%d); skipped %d calls",
                    self.max_tool_calls,
                    sum(over_limit),
                )

            await self._emit_status(AgentPhase.WAITING_FOR_TOOL)

            # Callbacks usually share one database session, which can't be
            # used by several tasks at once, so they run one at a time. Only
            # the searches themselves run in parallel.
            steps = []

            for parsed in parsed_calls:

                step = None

                if parsed is not None and self.on_tool_call is not None:
                    step = await self.on_tool_call(
                        parsed[0],
                        parsed[1],
                        iteration_count,
                    )

                steps.append(step)

            executions = await asyncio.gather(
                *(
                    self._search(parsed)
                    for parsed in parsed_calls
                )
            )

            await self._emit_status(AgentPhase.PROCESSING_RESULT)

            for call, step, execution, skipped in zip(
                function_calls,
                steps,
                executions,
                over_limit,
            ):

                result = None

                if execution is not None:

                    result = execution.result

                    tool_results.append(result)

                    if self.on_tool_result is not None:
                        await self.on_tool_result(step, execution)

                if skipped:
                    output = (
                        f"Error: tool call limit of {self.max_tool_calls} "
                        "reached; this search was not run. Answer with the "
                        "evidence you already have."
                    )
                else:
                    output = self._format_output(call, result)

                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": output,
                    }
                )

            # The iteration is complete: every call has its output. Save
            # where the run is, so a crash from here on loses at most the
            # next iteration.
            await self._emit_checkpoint(
                input_items=input_items,
                iteration=iteration_count,
                tool_call_count=tool_call_count,
                executed_tool_calls=executed_tool_calls,
                usage=usage,
            )

        logger.warning(
            "Research agent stopped after %d iterations without a final answer",
            self.max_iterations,
        )

        return AgentResult(
            answer="",
            tool_results=tool_results,
            success=False,
            error=f"No final answer after {self.max_iterations} iterations.",
            iteration_count=self.max_iterations,
            tool_call_count=tool_call_count,
            usage=usage,
        )

    async def _emit_status(self, phase: AgentPhase) -> None:

        if self.on_status_change is not None:
            await self.on_status_change(phase)

    async def _emit_checkpoint(
        self,
        input_items: list[Any],
        iteration: int,
        tool_call_count: int,
        executed_tool_calls: int,
        usage: TokenUsage,
    ) -> None:

        if self.on_checkpoint is None:
            return

        checkpoint = self._serialize_checkpoint(
            input_items=input_items,
            iteration=iteration,
            tool_call_count=tool_call_count,
            usage=usage,
            # Can be lower than tool_call_count (invalid or over-limit calls
            # don't run), so it's passed explicitly.
            executed_tool_calls=executed_tool_calls,
        )

        await self.on_checkpoint(checkpoint)

    def _serialize_checkpoint(
        self,
        input_items: list[Any],
        iteration: int,
        tool_call_count: int = 0,
        usage: TokenUsage | None = None,
        executed_tool_calls: int | None = None,
    ) -> str:
        """The run's position as a JSON string (AgentCheckpoint fields).

        Taken as a snapshot: later iterations change usage and input_items,
        not the string.
        """

        usage = usage if usage is not None else TokenUsage()

        checkpoint = AgentCheckpoint(
            iteration=iteration,
            input_items=[
                serialize_input_item(item)
                for item in input_items
            ],
            tool_call_count=tool_call_count,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            estimated_cost=usage.estimated_cost,
            executed_tool_calls=(
                tool_call_count
                if executed_tool_calls is None
                else executed_tool_calls
            ),
        )

        return json.dumps(asdict(checkpoint))

    def _deserialize_checkpoint(
        self,
        checkpoint: str | dict[str, Any] | AgentCheckpoint,
    ) -> AgentCheckpoint:
        """Accepts a checkpoint as a JSON string, as already-parsed data
        (e.g. the agent_runs.state JSONB value), or as an AgentCheckpoint.
        Anything invalid raises ValueError."""

        if isinstance(checkpoint, AgentCheckpoint):
            return checkpoint

        if isinstance(checkpoint, str):
            try:
                checkpoint = json.loads(checkpoint)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Invalid agent checkpoint JSON."
                ) from exc

        return AgentCheckpoint.from_dict(checkpoint)

    def _extract_usage(self, response) -> TokenUsage:

        # Token counts and estimated cost for one LLM call, from
        # response.usage (Responses API).
        usage = getattr(response, "usage", None)

        if usage is None:
            logger.warning("LLM response has no token usage; counting 0 tokens")
            return TokenUsage()

        input_tokens = getattr(
            usage,
            "input_tokens",
            0,
        ) or 0

        output_tokens = getattr(
            usage,
            "output_tokens",
            0,
        ) or 0

        total_tokens = getattr(
            usage,
            "total_tokens",
            input_tokens + output_tokens,
        ) or (input_tokens + output_tokens)

        # usage.input_tokens_details.cached_tokens: input tokens read from
        # the prompt cache, already included in input_tokens.
        details = getattr(usage, "input_tokens_details", None)

        cached_input_tokens = getattr(
            details,
            "cached_tokens",
            0,
        ) or 0

        # The response names the exact model snapshot that ran; fall back to
        # the model the provider was asked for.
        model = (
            getattr(response, "model", None)
            or getattr(self.llm, "model", None)
        )

        # Cached input tokens only change the price, so they're used here
        # and not kept.
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost=estimate_cost_usd(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_input_tokens,
            ),
        )

    def _add_usage(
        self,
        usage: TokenUsage,
        call_usage: TokenUsage,
    ) -> None:

        # Adds one LLM call's tokens, and their estimated cost, to the run's
        # running total. One unpriced call makes the whole run's cost
        # unknown.
        usage.input_tokens += call_usage.input_tokens
        usage.output_tokens += call_usage.output_tokens
        usage.total_tokens += call_usage.total_tokens

        if (
            usage.estimated_cost is None
            or call_usage.estimated_cost is None
        ):
            usage.estimated_cost = None
        else:
            usage.estimated_cost += call_usage.estimated_cost

    def _check_token_budget(
        self,
        usage: TokenUsage,
    ) -> None:

        if usage.total_tokens >= self.max_total_tokens:
            raise TokenBudgetExceeded(
                "Agent token budget exceeded: "
                f"{usage.total_tokens} >= "
                f"{self.max_total_tokens}"
            )

    def _over_budget_result(
        self,
        exc: TokenBudgetExceeded,
        tool_results: list[ToolResult],
        iteration_count: int,
        tool_call_count: int,
        usage: TokenUsage,
    ) -> AgentResult:

        # Returned instead of raising, so the caller still gets the token
        # usage (and the results so far) and can save them.
        logger.warning(
            "%s after %d iterations; stopping",
            exc,
            iteration_count,
        )

        return AgentResult(
            answer="",
            tool_results=tool_results,
            success=False,
            error=str(exc),
            iteration_count=iteration_count,
            tool_call_count=tool_call_count,
            usage=usage,
        )

    def _time_left(self) -> float | None:

        # Seconds until the run's deadline (never negative); None when there
        # is no deadline.
        if self._deadline is None:
            return None

        return max(self._deadline - time.perf_counter(), 0.0)

    def _timed_out_result(
        self,
        tool_results: list[ToolResult],
        iteration_count: int,
        tool_call_count: int,
        usage: TokenUsage,
    ) -> AgentResult:

        logger.warning(
            "Research agent hit the %ss time limit after %d iterations",
            self.MAX_EXECUTION_SECONDS,
            iteration_count,
        )

        return AgentResult(
            answer="",
            tool_results=tool_results,
            success=False,
            error=(
                "Agent run timed out after "
                f"{self.MAX_EXECUTION_SECONDS:g} seconds."
            ),
            iteration_count=iteration_count,
            tool_call_count=tool_call_count,
            usage=usage,
        )

    def _parse_call(
        self,
        call,
    ) -> tuple[str, str] | None:

        # Returns None when the call itself is invalid (unknown tool, bad
        # arguments); the error is sent back to the model instead.
        tool_name = TOOL_NAME_MAPPING.get(call.name)

        if tool_name is None:
            logger.warning("Model called unknown tool %s", call.name)
            return None

        try:
            arguments = json.loads(call.arguments)
            query = arguments["query"].strip()
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
            logger.warning(
                "Invalid arguments for %s: %s",
                call.name,
                call.arguments,
            )
            return None

        if not query:
            logger.warning("Empty query for %s", call.name)
            return None

        if len(query) > self.max_query_length:
            logger.warning(
                "Query for %s is %d characters; cutting to %d",
                call.name,
                len(query),
                self.max_query_length,
            )
            query = query[: self.max_query_length]

        return tool_name, query

    async def _search(
        self,
        parsed: tuple[str, str] | None,
    ) -> ToolExecution | None:

        if parsed is None:
            return None

        tool_name, query = parsed

        logger.info("Research agent calling %s: %s", tool_name, query)

        return await self._execute_tool(
            tool_name=tool_name,
            query=query,
        )

    async def _execute_tool(
        self,
        tool_name: str,
        query: str,
    ) -> ToolExecution:

        # Runs one search with a timeout, retrying failures. A tool that raises
        # or times out becomes a failed ToolResult, so one broken tool doesn't
        # cancel the other searches in this round or end the agent run.

        tool = self.registry.get(tool_name)

        # perf_counter: a monotonic clock meant for measuring durations.
        started_at = time.perf_counter()

        last_result: ToolResult | None = None

        attempts = 0

        for attempt in range(
            self.max_tool_retries + 1
        ):

            # The search gets the tool timeout, or less if the run's time
            # limit is closer.
            timeout = self.tool_timeout_seconds
            time_left = self._time_left()

            if time_left is not None:

                if time_left <= 0:
                    logger.warning(
                        "Agent time limit reached; not running tool=%s",
                        tool_name,
                    )
                    break

                timeout = min(timeout, time_left)

            attempts += 1

            attempt_started_at = time.perf_counter()

            try:

                result = await asyncio.wait_for(
                    tool.search(query),
                    timeout=timeout,
                )

            except asyncio.TimeoutError:

                result = ToolResult(
                    tool=tool_name,
                    query=query,
                    content="",
                    success=False,
                    error=(
                        "Tool execution timed out "
                        f"after {timeout:.3g} seconds."
                    ),
                )

                logger.warning(
                    "Tool timeout tool=%s attempt=%s duration_ms=%.2f",
                    tool_name,
                    attempts,
                    _elapsed_ms(attempt_started_at),
                )

            except Exception as exc:

                result = ToolResult(
                    tool=tool_name,
                    query=query,
                    content="",
                    success=False,
                    error=str(exc),
                )

                logger.exception(
                    "Tool exception tool=%s attempt=%s duration_ms=%.2f",
                    tool_name,
                    attempts,
                    _elapsed_ms(attempt_started_at),
                )

            else:

                logger.info(
                    "Tool completed tool=%s success=%s "
                    "attempt=%s duration_ms=%.2f",
                    tool_name,
                    result.success,
                    attempts,
                    _elapsed_ms(attempt_started_at),
                )

            last_result = result

            if result.success:
                break

            if attempt < self.max_tool_retries:

                # 1s, then 2s with the default RETRY_BACKOFF_SECONDS.
                delay = self.RETRY_BACKOFF_SECONDS * 2 ** attempt

                # Don't wait for a retry that the time limit won't allow.
                time_left = self._time_left()

                if time_left is not None and time_left <= delay:
                    break

                logger.info(
                    "Retrying tool=%s in %ss",
                    tool_name,
                    delay,
                )

                await asyncio.sleep(delay)

        if last_result is None:
            last_result = ToolResult(
                tool=tool_name,
                query=query,
                content="",
                success=False,
                error="Agent time limit reached; search was not run.",
            )

        return ToolExecution(
            result=last_result,
            duration_ms=_elapsed_ms(started_at),
            attempts=attempts,
        )

    def _format_output(
        self,
        call,
        result: ToolResult | None,
    ) -> str:

        if result is None:
            return f"Error: invalid call to {call.name} with arguments {call.arguments}."

        if not result.success:
            return f"Error: {result.tool} search failed: {result.error}"

        if not result.content:
            return "No results found."

        if len(result.content) > self.max_result_length:
            return (
                result.content[: self.max_result_length]
                + "\n\n[Result truncated.]"
            )

        return result.content
