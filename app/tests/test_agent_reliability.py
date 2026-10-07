import asyncio

import pytest

from app.agents.research_agent import ResearchAgent
from app.tests.fake_agent_llm import FakeAgentLLM, FakeFunctionCall, FakeResponse
from app.tests.fake_agent_tools import FakeRegistry
from app.tools.base import ToolResult


class SlowTool:

    @property
    def name(self):
        return "tavily"

    async def search(self, query: str):

        await asyncio.sleep(2)

        return ToolResult(
            tool="tavily",
            query=query,
            content="too slow",
            success=True,
        )


class SlowRegistry:

    def get(self, name: str):
        return SlowTool()


@pytest.mark.asyncio
async def test_tool_timeout():

    agent = ResearchAgent(
        llm=FakeAgentLLM(),
        registry=SlowRegistry(),
    )

    agent.tool_timeout_seconds = 0.01

    execution = await agent._execute_tool(
        tool_name="tavily",
        query="test",
    )

    assert execution.result.success is False

    assert "timed out" in execution.result.error

    # 1 attempt + 2 retries, each cut off after 0.01s.
    assert execution.attempts == 3
    assert execution.duration_ms >= 30


def test_agent_default_limits():

    agent = ResearchAgent(llm=FakeAgentLLM(), registry=SlowRegistry())

    assert agent.max_iterations == 6
    assert agent.max_tool_calls == 8
    assert agent.tool_timeout_seconds == 30.0
    assert agent.max_tool_retries == 2
    assert agent.max_query_length == 500
    assert agent.max_result_length == 12_000
    assert agent.max_total_tokens == 20_000


@pytest.mark.asyncio
async def test_workflow_passes_settings_to_agent(db_session, monkeypatch):

    from app.core.config import settings
    from app.services import workflow_service
    from app.tests.fake_agent_tools import FakeRegistry
    from app.tests.test_workflow_service import create_task

    created = []

    class RecordingAgent(ResearchAgent):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            created.append(self)

    monkeypatch.setattr(workflow_service, "ResearchAgent", RecordingAgent)
    monkeypatch.setattr(workflow_service, "OpenAIProvider", FakeAgentLLM)
    monkeypatch.setattr(workflow_service, "ToolRegistry", FakeRegistry)

    task = await create_task(db_session)

    await workflow_service.execute_research_workflow(db=db_session, task=task)

    [agent] = created

    assert agent.max_iterations == settings.agent_max_iterations
    assert agent.max_tool_calls == settings.agent_max_tool_calls
    assert agent.tool_timeout_seconds == settings.agent_tool_timeout_seconds
    assert agent.max_tool_retries == settings.agent_max_tool_retries
    assert agent.max_query_length == settings.agent_max_query_length
    assert agent.max_result_length == settings.agent_max_result_length
    assert agent.max_total_tokens == settings.agent_max_total_tokens


def make_call(arguments: str) -> FakeFunctionCall:
    return FakeFunctionCall(
        type="function_call",
        name="tavily_search",
        arguments=arguments,
        call_id="call_1",
    )


def test_long_query_is_cut_to_max_length():

    agent = ResearchAgent(llm=FakeAgentLLM(), registry=SlowRegistry())
    agent.max_query_length = 10

    parsed = agent._parse_call(make_call('{"query": "  ' + "a" * 50 + '  "}'))

    assert parsed == ("tavily", "a" * 10)


def test_empty_query_is_invalid():

    agent = ResearchAgent(llm=FakeAgentLLM(), registry=SlowRegistry())

    assert agent._parse_call(make_call('{"query": "   "}')) is None
    assert agent._parse_call(make_call('{"query": 42}')) is None


def test_long_result_is_truncated_for_model():

    agent = ResearchAgent(llm=FakeAgentLLM(), registry=SlowRegistry())
    agent.max_result_length = 20

    result = ToolResult(
        tool="tavily",
        query="rag",
        content="x" * 100,
        success=True,
    )

    output = agent._format_output(make_call('{"query": "rag"}'), result)

    assert output == "x" * 20 + "\n\n[Result truncated.]"

    # The ToolResult itself (what gets saved) is not changed.
    assert result.content == "x" * 100


# -------------------------
# Maximum total execution time
# -------------------------


class SlowLLM:
    """Takes longer to answer than the agent's time limit."""

    async def generate_with_tools(self, input_items, tools, instructions=None):
        await asyncio.sleep(2)
        return FakeResponse(output=[], output_text="too late")


@pytest.mark.asyncio
async def test_run_stops_when_llm_exceeds_time_limit():

    agent = ResearchAgent(llm=SlowLLM(), registry=SlowRegistry())
    agent.MAX_EXECUTION_SECONDS = 0.05

    result = await agent.run(question="What is RAG?")

    assert result.success is False
    assert result.answer == ""
    assert result.error == "Agent run timed out after 0.05 seconds."
    assert result.iteration_count == 1


@pytest.mark.asyncio
async def test_time_limit_cuts_short_slow_search_and_ends_run():

    # The tool timeout (30s) is longer than the run's time limit, so the
    # search is cut short by the time limit instead, without retries.
    llm = FakeAgentLLM()

    agent = ResearchAgent(llm=llm, registry=SlowRegistry())
    agent.MAX_EXECUTION_SECONDS = 0.05
    agent.tool_timeout_seconds = 30

    started_at = asyncio.get_running_loop().time()

    result = await agent.run(question="What is RAG?")

    elapsed = asyncio.get_running_loop().time() - started_at

    assert elapsed < 1

    assert result.success is False
    assert result.error == "Agent run timed out after 0.05 seconds."

    # The search was tried once and saved as a failure.
    assert len(result.tool_results) == 1
    assert result.tool_results[0].success is False
    assert "timed out" in result.tool_results[0].error

    # No second LLM call after the time ran out.
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_no_search_starts_after_time_limit():

    agent = ResearchAgent(llm=FakeAgentLLM(), registry=SlowRegistry())

    # A deadline that has already passed.
    agent._deadline = 0.0

    execution = await agent._execute_tool(tool_name="tavily", query="rag")

    assert execution.result.success is False
    assert execution.result.error == "Agent time limit reached; search was not run."
    assert execution.attempts == 0


# -------------------------
# Tool duration
# -------------------------


class TimedTool:

    def __init__(self, delay: float):
        self.delay = delay

    @property
    def name(self):
        return "tavily"

    async def search(self, query: str):

        await asyncio.sleep(self.delay)

        return ToolResult(
            tool="tavily",
            query=query,
            content="ok",
            success=True,
        )


class TimedRegistry:

    def __init__(self, delay: float):
        self.tool = TimedTool(delay)

    def get(self, name: str):
        return self.tool


@pytest.mark.asyncio
async def test_execute_tool_returns_duration():

    agent = ResearchAgent(llm=FakeAgentLLM(), registry=TimedRegistry(0.05))

    execution = await agent._execute_tool(tool_name="tavily", query="rag")

    assert execution.result.success is True
    assert execution.attempts == 1
    assert 50 <= execution.duration_ms < 1000


@pytest.mark.asyncio
async def test_on_tool_result_receives_duration():

    executions = []

    async def on_tool_result(step, execution):
        executions.append(execution)

    await ResearchAgent(
        llm=FakeAgentLLM(),
        registry=TimedRegistry(0.05),
        on_tool_result=on_tool_result,
    ).run(question="What is RAG?")

    assert len(executions) == 1
    assert executions[0].result.query == "retrieval augmented generation"
    assert executions[0].duration_ms >= 50


# -------------------------
# Token budget with a high-usage model
# -------------------------


class HighUsageLLM:
    """Answers straight away, but each call uses 10,000 tokens."""

    def __init__(self):
        self.calls = 0

    async def generate_with_tools(
        self,
        input_items,
        tools,
        instructions=None,
    ):
        self.calls += 1

        return FakeResponse(
            output=[],
            output_text="answer",
            input_tokens=6_000,
            output_tokens=4_000,
        )


@pytest.mark.asyncio
async def test_high_usage_answer_within_budget():

    llm = HighUsageLLM()

    agent = ResearchAgent(llm=llm, registry=SlowRegistry())

    result = await agent.run(question="What is RAG?")

    # 10,000 of the default 20,000 tokens.
    assert llm.calls == 1
    assert result.success is True
    assert result.answer == "answer"
    assert result.usage.input_tokens == 6_000
    assert result.usage.output_tokens == 4_000
    assert result.usage.total_tokens == 10_000


@pytest.mark.asyncio
async def test_high_usage_final_answer_over_budget_is_returned():

    llm = HighUsageLLM()

    agent = ResearchAgent(
        llm=llm,
        registry=SlowRegistry(),
        max_total_tokens=5_000,
    )

    result = await agent.run(question="What is RAG?")

    # The one call goes over the budget, but it is the final answer, and
    # its tokens are already spent, so the answer is kept.
    assert llm.calls == 1
    assert result.success is True
    assert result.answer == "answer"
    assert result.usage.total_tokens == 10_000


@pytest.mark.asyncio
async def test_agent_enforces_token_budget():
    llm = HighUsageLLM()
    registry = FakeRegistry()

    agent = ResearchAgent(
        llm=llm,
        registry=registry,
        max_total_tokens=10_000,
    )

    result = await agent.run(
        question="What is retrieval augmented generation?"
    )

    assert result.usage.total_tokens == 10_000
    assert llm.calls == 1


class BudgetLLM:
    def __init__(self):
        self.calls = 0

    async def generate_with_tools(
        self,
        input_items,
        tools,
        instructions=None,
    ):
        self.calls += 1

        if self.calls == 1:
            return FakeResponse(
                output=[
                    FakeFunctionCall(
                        type="function_call",
                        name="tavily_search",
                        arguments=(
                            '{"query": "retrieval augmented generation"}'
                        ),
                        call_id="budget_call_1",
                    )
                ],
                input_tokens=6_000,
                output_tokens=4_000,
            )

        return FakeResponse(
            output=[],
            output_text="final answer",
            input_tokens=6_000,
            output_tokens=4_000,
        )


@pytest.mark.asyncio
async def test_agent_stops_before_exceeding_token_budget():
    llm = BudgetLLM()
    registry = FakeRegistry()

    agent = ResearchAgent(
        llm=llm,
        registry=registry,
        max_total_tokens=10_000,
    )

    # run() doesn't raise: it returns a failed result that still carries the
    # token usage, so the workflow can save it to agent_runs.
    result = await agent.run(
        question="What is retrieval augmented generation?"
    )

    assert result.success is False
    assert "token budget exceeded" in result.error
    assert result.usage.total_tokens == 10_000

    # No second call, and the first call's search was never run.
    assert llm.calls == 1
    assert result.tool_results == []
