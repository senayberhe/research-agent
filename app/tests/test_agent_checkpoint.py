import json

import pytest
from openai.types.responses import ResponseFunctionToolCall, ResponseReasoningItem
from sqlalchemy import select

from app.agents.research_agent import (
    CHECKPOINT_VERSION,
    AgentCheckpoint,
    ResearchAgent,
    TokenUsage,
    serialize_input_item,
)
from app.db.models import AgentRun, AgentRunStatus, ResearchStep, TaskStatus
from app.services import workflow_service
from app.services.workflow_service import execute_research_workflow
from app.tests.fake_agent_llm import FakeAgentLLM, FakeFunctionCall, FakeResponse
from app.tests.fake_agent_tools import FakeRegistry
from app.tests.test_agent_token_budget import SearchForeverLLM
from app.tests.test_workflow_service import create_task


class RecordingCheckpoints:
    """An on_checkpoint callback that keeps every checkpoint.

    The agent passes a JSON string; it's turned back into AgentCheckpoint
    here (which also validates it) so tests can read its fields.
    """

    def __init__(self):
        self.raw: list[str] = []
        self.checkpoints: list[AgentCheckpoint] = []

    async def __call__(self, checkpoint: str) -> None:
        self.raw.append(checkpoint)
        self.checkpoints.append(
            AgentCheckpoint.from_dict(json.loads(checkpoint))
        )


class CrashOnThirdCallLLM:
    """Searches in rounds 1 and 2, then the API fails on the 3rd call."""

    def __init__(self):
        self.calls = 0

    async def generate_with_tools(self, input_items, tools, instructions=None):

        self.calls += 1

        if self.calls == 3:
            raise RuntimeError("OpenAI connection reset")

        return FakeResponse(
            output=[
                FakeFunctionCall(
                    type="function_call",
                    name="tavily_search",
                    arguments=f'{{"query": "rag {self.calls}"}}',
                    call_id=f"call_{self.calls}",
                )
            ],
        )


# -------------------------
# Agent: taking checkpoints
# -------------------------


@pytest.mark.asyncio
async def test_checkpoint_after_each_round_with_tool_calls():

    recorder = RecordingCheckpoints()

    await ResearchAgent(
        llm=FakeAgentLLM(),
        registry=FakeRegistry(),
        on_checkpoint=recorder,
    ).run(question="What is RAG?")

    # One after round 1 (the search), one after round 2 (the answer).
    checkpoint, final = recorder.checkpoints

    assert final.iteration == 2
    assert final.usage.total_tokens == 300

    assert checkpoint.version == CHECKPOINT_VERSION
    assert checkpoint.iteration == 1
    assert checkpoint.executed_tool_calls == 1
    assert checkpoint.tool_call_count == 1
    assert checkpoint.usage.total_tokens == 150

    assert checkpoint.input_items == [
        {"role": "user", "content": "What is RAG?"},
        {
            "type": "function_call",
            "name": "tavily_search",
            "arguments": '{"query": "retrieval augmented generation"}',
            "call_id": "call_1",
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "Fake Tavily research result.",
        },
    ]


@pytest.mark.asyncio
async def test_answer_without_searching_still_gets_a_final_checkpoint():

    from app.tests.test_agent_reliability import HighUsageLLM

    recorder = RecordingCheckpoints()

    await ResearchAgent(
        llm=HighUsageLLM(),
        registry=FakeRegistry(),
        on_checkpoint=recorder,
    ).run(question="What is RAG?")

    [final] = recorder.checkpoints

    assert final.iteration == 1
    assert final.tool_call_count == 0
    assert final.usage.total_tokens == 10_000


@pytest.mark.asyncio
async def test_checkpoints_track_progress_over_rounds():

    recorder = RecordingCheckpoints()

    await ResearchAgent(
        llm=SearchForeverLLM(),
        registry=FakeRegistry(),
        on_checkpoint=recorder,
        max_iterations=3,
    ).run(question="What is RAG?")

    assert [c.iteration for c in recorder.checkpoints] == [1, 2, 3]
    assert [c.executed_tool_calls for c in recorder.checkpoints] == [1, 2, 3]
    assert [c.usage.total_tokens for c in recorder.checkpoints] == [
        1000,
        2000,
        3000,
    ]

    # The question, then a call and its output per round.
    assert [len(c.input_items) for c in recorder.checkpoints] == [3, 5, 7]


@pytest.mark.asyncio
async def test_checkpoint_usage_is_a_snapshot():

    recorder = RecordingCheckpoints()

    result = await ResearchAgent(
        llm=FakeAgentLLM(),
        registry=FakeRegistry(),
        on_checkpoint=recorder,
    ).run(question="What is RAG?")

    # The run went on to use more tokens; the checkpoint kept its value.
    assert result.usage.total_tokens == 300
    assert recorder.checkpoints[0].usage.total_tokens == 150


@pytest.mark.asyncio
async def test_checkpoint_is_json_serializable():

    recorder = RecordingCheckpoints()

    await ResearchAgent(
        llm=FakeAgentLLM(),
        registry=FakeRegistry(),
        on_checkpoint=recorder,
    ).run(question="What is RAG?")

    raw = recorder.raw[0]

    # What the agent hands over is a JSON string of the flat layout.
    assert isinstance(raw, str)
    assert set(json.loads(raw)) == {
        "iteration",
        "input_items",
        "tool_call_count",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "estimated_cost",
        "executed_tool_calls",
        "version",
    }


# -------------------------
# Serialization
# -------------------------


def test_serializes_real_openai_output_items():

    function_call = ResponseFunctionToolCall(
        arguments='{"query": "rag"}',
        call_id="call_1",
        name="tavily_search",
        type="function_call",
        id="fc_1",
        status="completed",
    )

    reasoning = ResponseReasoningItem(id="rs_1", summary=[], type="reasoning")

    assert serialize_input_item(function_call) == {
        "arguments": '{"query": "rag"}',
        "call_id": "call_1",
        "name": "tavily_search",
        "type": "function_call",
        "id": "fc_1",
        "status": "completed",
    }
    assert serialize_input_item(reasoning) == {
        "id": "rs_1",
        "summary": [],
        "type": "reasoning",
    }


def test_dict_items_pass_through_unchanged():

    item = {"role": "user", "content": "q"}

    assert serialize_input_item(item) is item


def test_unknown_item_type_raises():

    with pytest.raises(TypeError):
        serialize_input_item(object())


def test_checkpoint_round_trips_through_dict():

    checkpoint = AgentCheckpoint(
        iteration=2,
        input_items=[{"role": "user", "content": "q"}],
        tool_call_count=4,
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        estimated_cost=None,
        executed_tool_calls=3,
    )

    assert AgentCheckpoint.from_dict(checkpoint.to_dict()) == checkpoint


def test_checkpoint_with_other_version_is_refused():

    data = valid_checkpoint_data()

    data["version"] = 99

    with pytest.raises(ValueError, match="Unsupported agent checkpoint version 99"):
        AgentCheckpoint.from_dict(data)


# -------------------------
# Workflow: saving checkpoints
# -------------------------


async def load_run(db_session, task_id: int) -> AgentRun:

    db_session.expire_all()

    return (
        await db_session.execute(
            select(AgentRun).where(AgentRun.task_id == task_id)
        )
    ).scalar_one()


@pytest.mark.asyncio
async def test_completed_workflow_clears_checkpoint(db_session, monkeypatch):

    monkeypatch.setattr(workflow_service, "OpenAIProvider", FakeAgentLLM)
    monkeypatch.setattr(workflow_service, "ToolRegistry", FakeRegistry)

    task = await create_task(db_session)
    task_id = task.id

    await execute_research_workflow(db=db_session, task=task)

    run = await load_run(db_session, task_id)

    # Checkpoints were saved during the run (see test_agent_state), but a
    # completed run has nothing to resume, so its checkpoint is cleared.
    # The totals stay on the run itself.
    assert run.status == AgentRunStatus.COMPLETED
    assert run.state is None
    assert run.state_updated_at is None
    assert run.iteration_count == 2
    assert run.total_tokens == 300


@pytest.mark.asyncio
async def test_checkpoint_survives_crash_mid_run(db_session, monkeypatch):

    monkeypatch.setattr(workflow_service, "OpenAIProvider", CrashOnThirdCallLLM)
    monkeypatch.setattr(workflow_service, "ToolRegistry", FakeRegistry)

    task = await create_task(db_session)
    task_id = task.id

    await execute_research_workflow(db=db_session, task=task)

    assert task.status == TaskStatus.FAILED

    run = await load_run(db_session, task_id)

    assert run.status == AgentRunStatus.FAILED
    assert run.error == "OpenAI connection reset"

    # The rollback after the crash didn't undo the checkpoints: the run's
    # position at the end of round 2 is still there.
    checkpoint = AgentCheckpoint.from_dict(run.state)

    assert checkpoint.iteration == 2
    assert checkpoint.executed_tool_calls == 2
    assert checkpoint.tool_call_count == 2
    assert checkpoint.usage.total_tokens == 300

    calls = [
        item
        for item in checkpoint.input_items
        if item.get("type") == "function_call"
    ]
    outputs = [
        item
        for item in checkpoint.input_items
        if item.get("type") == "function_call_output"
    ]

    assert [c["arguments"] for c in calls] == [
        '{"query": "rag 1"}',
        '{"query": "rag 2"}',
    ]
    assert [o["call_id"] for o in outputs] == ["call_1", "call_2"]

    # And it matches the steps that were saved.
    steps = (
        await db_session.execute(
            select(ResearchStep)
            .where(ResearchStep.task_id == task_id)
            .order_by(ResearchStep.id)
        )
    ).scalars().all()

    assert [step.query for step in steps] == ["rag 1", "rag 2"]


def test_checkpoint_round_trip():
    agent = ResearchAgent(
        llm=None,
        registry=None,
    )

    input_items = [
        {
            "role": "user",
            "content": "What is RAG?",
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "RAG retrieves external information.",
        },
    ]

    checkpoint = agent._serialize_checkpoint(
        input_items=input_items,
        iteration=2,
    )

    restored = agent._deserialize_checkpoint(
        checkpoint
    )

    restored_items, restored_iteration = (
        restored.input_items,
        restored.iteration,
    )

    assert restored_items == input_items
    assert restored_iteration == 2


def test_serialize_checkpoint_keeps_counts_and_usage():

    agent = ResearchAgent(llm=None, registry=None)

    usage = TokenUsage(
        input_tokens=300,
        output_tokens=100,
        total_tokens=400,
        estimated_cost=0.002,
    )

    data = agent._serialize_checkpoint(
        input_items=[{"role": "user", "content": "q"}],
        iteration=3,
        tool_call_count=5,
        usage=usage,
        executed_tool_calls=4,
    )

    assert isinstance(data, str)

    restored = agent._deserialize_checkpoint(data)

    assert restored.iteration == 3
    assert restored.tool_call_count == 5
    assert restored.executed_tool_calls == 4
    assert restored.usage == usage


def test_serialize_checkpoint_counts_all_calls_as_run_by_default():

    agent = ResearchAgent(llm=None, registry=None)

    data = agent._serialize_checkpoint(
        input_items=[{"role": "user", "content": "q"}],
        iteration=1,
        tool_call_count=2,
    )

    assert json.loads(data)["executed_tool_calls"] == 2


def test_deserialize_checkpoint_accepts_object_dict_and_json_string():

    agent = ResearchAgent(llm=None, registry=None)

    data = agent._serialize_checkpoint(
        input_items=[{"role": "user", "content": "q"}],
        iteration=1,
    )
    # data is the JSON string; the JSONB column holds it parsed.
    stored = json.loads(data)
    checkpoint = AgentCheckpoint.from_dict(stored)

    assert agent._deserialize_checkpoint(checkpoint) is checkpoint
    assert agent._deserialize_checkpoint(stored) == checkpoint
    assert agent._deserialize_checkpoint(data) == checkpoint


@pytest.mark.asyncio
async def test_run_resumes_from_json_string_checkpoint():

    from app.tests.test_resume_workflow import AnswerLLM

    agent = ResearchAgent(llm=AnswerLLM(), registry=FakeRegistry())

    checkpoint = agent._serialize_checkpoint(
        input_items=[
            {"role": "user", "content": "What is RAG?"},
            {"type": "function_call", "name": "tavily_search",
             "arguments": '{"query": "rag"}', "call_id": "call_1"},
            {"type": "function_call_output", "call_id": "call_1",
             "output": "result"},
        ],
        iteration=1,
        tool_call_count=1,
    )

    result = await agent.run(
        question="What is RAG?",
        checkpoint=checkpoint,
    )

    assert result.answer == "Resumed answer."
    assert result.iteration_count == 2
    assert agent.llm.requests[0] == json.loads(checkpoint)["input_items"]


# -------------------------
# Invalid checkpoints
# -------------------------


def test_invalid_checkpoint_raises_error():
    agent = ResearchAgent(
        llm=None,
        registry=None,
    )

    with pytest.raises(
        ValueError,
        match="Invalid agent checkpoint",
    ):
        agent._deserialize_checkpoint(
            "not valid json"
        )


def valid_checkpoint_data() -> dict:
    return AgentCheckpoint(
        iteration=2,
        input_items=[{"role": "user", "content": "What is RAG?"}],
        tool_call_count=2,
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        estimated_cost=0.001,
        executed_tool_calls=1,
    ).to_dict()


def test_valid_checkpoint_data_is_accepted():

    # The baseline the cases below break one field of.
    assert AgentCheckpoint.from_dict(valid_checkpoint_data()).iteration == 2


def changed(**fields):
    """Valid checkpoint data with some fields replaced."""
    return lambda d: {**d, **fields}


def without(name):
    return lambda d: {k: v for k, v in d.items() if k != name}


@pytest.mark.parametrize(
    ("make_data", "message"),
    [
        # Not an object at all.
        (lambda d: None, "Invalid agent checkpoint: expected an object, got NoneType"),
        (lambda d: [d], "Invalid agent checkpoint: expected an object, got list"),
        # Missing fields.
        (without("input_items"), r"Checkpoint missing fields: \['input_items'\]"),
        (without("total_tokens"), r"Checkpoint missing fields: \['total_tokens'\]"),
        # Bad counters.
        (changed(iteration="2"), "Checkpoint iteration must be an integer"),
        (changed(iteration=True), "Checkpoint iteration must be an integer"),
        (changed(iteration=0), "Checkpoint iteration must be at least 1"),
        (changed(tool_call_count=-1), "Checkpoint tool_call_count must not be negative"),
        (changed(executed_tool_calls=5), "Checkpoint executed_tool_calls must not be greater than tool_call_count"),
        # Bad usage.
        (changed(total_tokens=1.5), "Checkpoint total_tokens must be an integer"),
        (changed(input_tokens=-10), "Checkpoint input_tokens must not be negative"),
        (changed(estimated_cost="free"), "Checkpoint estimated_cost must be numeric"),
        (changed(estimated_cost=True), "Checkpoint estimated_cost must be numeric"),
        (changed(estimated_cost=-1), "Checkpoint estimated_cost must not be negative"),
        # Bad conversation.
        (changed(input_items="What is RAG?"), "Checkpoint input_items must be a list"),
        (changed(input_items=[]), "Checkpoint input_items must not be empty"),
        (
            lambda d: {**d, "input_items": [*d["input_items"], "oops"]},
            r"Checkpoint input_items\[1\] must be an object",
        ),
    ],
)
def test_invalid_checkpoint_fields_raise_value_error(make_data, message):

    data = make_data(valid_checkpoint_data())

    with pytest.raises(ValueError, match=message):
        AgentCheckpoint.from_dict(data)


def test_missing_resume_safety_fields_get_safe_defaults():

    # A checkpoint written without version / executed_tool_calls (like the
    # one in test_resume_from_checkpoint_after_crash) still loads; every
    # requested call is assumed to have run, so no searches are gained.
    data = valid_checkpoint_data()
    del data["version"]
    del data["executed_tool_calls"]

    checkpoint = AgentCheckpoint.from_dict(data)

    assert checkpoint.version == CHECKPOINT_VERSION
    assert checkpoint.executed_tool_calls == checkpoint.tool_call_count == 2


def test_unknown_version_is_still_reported_as_unsupported():

    # Version 1 was the earlier nested layout; it must not be misread.
    data = valid_checkpoint_data()
    data["version"] = 1

    with pytest.raises(ValueError, match="Unsupported agent checkpoint version 1"):
        AgentCheckpoint.from_dict(data)


def test_unpriced_usage_is_valid():

    data = valid_checkpoint_data()
    data["estimated_cost"] = None

    assert AgentCheckpoint.from_dict(data).estimated_cost is None


def test_non_object_json_is_invalid():

    agent = ResearchAgent(llm=None, registry=None)

    # Valid JSON, but not a checkpoint object.
    with pytest.raises(ValueError, match="Invalid agent checkpoint: expected an object"):
        agent._deserialize_checkpoint("[1, 2, 3]")


@pytest.mark.asyncio
async def test_checkpoint_counts_only_searches_that_ran():

    from app.tests.test_research_agent import AlwaysCallsToolLLM

    recorder = RecordingCheckpoints()

    # Calls a tool that doesn't exist: requested, but never run.
    await ResearchAgent(
        llm=AlwaysCallsToolLLM("google_search"),
        registry=FakeRegistry(),
        on_checkpoint=recorder,
        max_iterations=2,
    ).run(question="What is RAG?")

    assert [c.tool_call_count for c in recorder.checkpoints] == [1, 2]
    assert [c.executed_tool_calls for c in recorder.checkpoints] == [0, 0]


# -------------------------
# Crash / resume
# -------------------------


class ResumeLLM:
    def __init__(self):
        self.calls = 0
        self.requests = []

    async def generate_with_tools(
        self,
        input_items,
        tools,
        instructions=None,
    ):
        self.calls += 1
        self.requests.append(list(input_items))

        if self.calls == 1:
            return FakeResponse(
                output=[
                    FakeFunctionCall(
                        type="function_call",
                        name="tavily_search",
                        arguments=(
                            '{"query": '
                            '"retrieval augmented generation"}'
                        ),
                        call_id="resume_call_1",
                    )
                ],
                input_tokens=100,
                output_tokens=50,
            )

        return FakeResponse(
            output=[],
            output_text="Final resumed answer.",
            input_tokens=100,
            output_tokens=50,
        )


@pytest.mark.asyncio
async def test_checkpoint_contains_full_execution_state():
    llm = ResumeLLM()
    registry = FakeRegistry()

    checkpoints = []

    async def save_checkpoint(
        checkpoint,
    ):
        checkpoints.append(
            checkpoint
        )

    agent = ResearchAgent(
        llm=llm,
        registry=registry,
        on_checkpoint=save_checkpoint,
    )

    result = await agent.run(
        question="What is RAG?"
    )

    assert result.answer == "Final resumed answer."

    assert checkpoints

    # The last checkpoint is the end-of-run one.
    checkpoint_json = checkpoints[-1]

    data = json.loads(checkpoint_json)

    assert data["iteration"] == result.iteration_count == 2
    assert data["tool_call_count"] == 1
    assert data["executed_tool_calls"] == 1
    assert data["input_tokens"] == 200
    assert data["output_tokens"] == 100
    assert data["total_tokens"] == 300

    # The search and its output are in the saved conversation.
    assert [item.get("call_id") for item in data["input_items"][1:]] == [
        "resume_call_1",
        "resume_call_1",
    ]


@pytest.mark.asyncio
async def test_resume_from_checkpoint_after_crash():

    # The LLM had already made its search call (call 1) when the process
    # crashed after round 1; the next call it makes is the final answer.
    llm = ResumeLLM()
    llm.calls = 1

    agent = ResearchAgent(
        llm=llm,
        registry=FakeRegistry(),
    )

    # The round-1 checkpoint, as a JSON string.
    checkpoint = (
        agent._serialize_checkpoint(
            input_items=[
                {
                    "role": "user",
                    "content": "What is RAG?",
                },
                {
                    "type": "function_call",
                    "name": "tavily_search",
                    "arguments": '{"query": "retrieval augmented generation"}',
                    "call_id": "resume_call_1",
                },
                {
                    "type": "function_call_output",
                    "call_id": "resume_call_1",
                    "output": (
                        "RAG retrieves external information."
                    ),
                },
            ],
            iteration=1,
            tool_call_count=1,
            usage=TokenUsage(
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
                estimated_cost=0.000035,
            ),
        )
    )

    result = await agent.run(
        question="What is RAG?",
        checkpoint=checkpoint,
    )

    assert result.answer == "Final resumed answer."
    assert result.iteration_count == 2
    assert result.tool_call_count == 1
    assert result.usage.input_tokens == 200
    assert result.usage.output_tokens == 100
    assert result.usage.total_tokens == 300

    # Most important: resuming made exactly one new LLM call, and round 1
    # (the search) was not repeated.
    new_calls = llm.calls - 1
    assert new_calls == 1
    assert result.tool_results == []

    # The iteration count and token totals above only add up if the run
    # continued from the checkpoint; starting over would give 1 and 150.
    assert json.loads(checkpoint)["iteration"] == 1

    # The one new call was sent the saved conversation, not a fresh one.
    assert llm.requests[0] == json.loads(checkpoint)["input_items"]
