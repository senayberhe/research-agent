import pytest
from sqlalchemy import select

from app.agents.research_agent import (
    AgentCheckpoint,
    ResearchAgent,
)
from app.db.models import (
    AgentRun,
    AgentRunStatus,
    ResearchStep,
    TaskStatus,
)
from app.services import workflow_service
from app.services.agent_run_service import (
    create_agent_run,
    load_agent_checkpoint,
)
from app.services.workflow_service import (
    execute_research_workflow,
    resume_research_workflow,
)
from app.tests.fake_agent_llm import FakeResponse
from app.tests.fake_agent_tools import FakeRegistry
from app.tests.test_agent_checkpoint import CrashOnThirdCallLLM
from app.tests.test_workflow_service import create_task


class AnswerLLM:
    """Answers straight away (150 tokens) and records what it was sent."""

    instances: list["AnswerLLM"] = []

    def __init__(self):
        self.calls = 0
        self.requests = []
        self.tools_offered = []
        AnswerLLM.instances.append(self)

    async def generate_with_tools(self, input_items, tools, instructions=None):
        self.calls += 1
        self.requests.append(list(input_items))
        self.tools_offered.append(len(tools))
        return FakeResponse(output=[], output_text="Resumed answer.")


class NeverAnswersLLM:
    async def generate_with_tools(self, input_items, tools, instructions=None):
        return FakeResponse(output=[], output_text="")


def round_two_checkpoint(**overrides) -> AgentCheckpoint:
    """Where CrashOnThirdCallLLM leaves a run: two rounds, two searches."""

    values = dict(
        iteration=2,
        executed_tool_calls=2,
        tool_call_count=2,
        input_tokens=200,
        output_tokens=100,
        total_tokens=300,
        estimated_cost=0.001,
        input_items=[
            {"role": "user", "content": "What is RAG?"},
            {"type": "function_call", "name": "tavily_search",
             "arguments": '{"query": "rag 1"}', "call_id": "call_1"},
            {"type": "function_call_output", "call_id": "call_1",
             "output": "result 1"},
            {"type": "function_call", "name": "tavily_search",
             "arguments": '{"query": "rag 2"}', "call_id": "call_2"},
            {"type": "function_call_output", "call_id": "call_2",
             "output": "result 2"},
        ],
    )
    values.update(overrides)
    return AgentCheckpoint(**values)


# -------------------------
# Agent: continuing from a checkpoint
# -------------------------


@pytest.mark.asyncio
async def test_agent_continues_from_checkpoint():

    llm = AnswerLLM()
    checkpoint = round_two_checkpoint()

    result = await ResearchAgent(llm=llm, registry=FakeRegistry()).run(
        question="What is RAG?",
        checkpoint=checkpoint,
    )

    # One new call (round 3), sent the saved conversation as-is.
    assert llm.calls == 1
    assert llm.requests[0] == checkpoint.input_items

    assert result.success is True
    assert result.answer == "Resumed answer."

    # Counts and usage are for the whole job.
    assert result.iteration_count == 3
    assert result.tool_call_count == 2
    assert result.usage.total_tokens == 300 + 150

    # Only this attempt's searches (none); earlier ones are in the database.
    assert result.tool_results == []


@pytest.mark.asyncio
async def test_resume_does_not_change_the_checkpoint():

    checkpoint = round_two_checkpoint()
    items_before = list(checkpoint.input_items)

    await ResearchAgent(llm=AnswerLLM(), registry=FakeRegistry()).run(
        question="q",
        checkpoint=checkpoint,
    )

    assert checkpoint.input_items == items_before
    assert checkpoint.usage.total_tokens == 300


@pytest.mark.asyncio
async def test_iteration_limit_covers_the_whole_job():

    llm = AnswerLLM()

    result = await ResearchAgent(
        llm=llm,
        registry=FakeRegistry(),
        max_iterations=6,
    ).run(question="q", checkpoint=round_two_checkpoint(iteration=6))

    # All 6 rounds were used before the crash: nothing left to run.
    assert llm.calls == 0
    assert result.success is False
    assert result.error == "No final answer after 6 iterations."


@pytest.mark.asyncio
async def test_token_budget_covers_the_whole_job():

    llm = AnswerLLM()

    result = await ResearchAgent(
        llm=llm,
        registry=FakeRegistry(),
        max_total_tokens=20_000,
    ).run(
        question="q",
        checkpoint=round_two_checkpoint(total_tokens=20_000),
    )

    # The crashed attempt already spent the budget: no new LLM call.
    assert llm.calls == 0
    assert result.success is False
    assert result.error == "Agent token budget exceeded: 20000 >= 20000"


@pytest.mark.asyncio
async def test_tool_call_limit_covers_the_whole_job():

    llm = AnswerLLM()

    await ResearchAgent(
        llm=llm,
        registry=FakeRegistry(),
        max_tool_calls=8,
    ).run(question="q", checkpoint=round_two_checkpoint(executed_tool_calls=8))

    # All 8 searches were used before the crash: no tools offered.
    assert llm.tools_offered == [0]


# -------------------------
# load_agent_checkpoint
# -------------------------


@pytest.mark.asyncio
async def test_load_agent_checkpoint(db_session):

    task = await create_task(db_session)

    run = await create_agent_run(db=db_session, task_id=task.id)
    await db_session.commit()

    assert await load_agent_checkpoint(db=db_session, run=run) == (None, 0)

    run.state = round_two_checkpoint().to_dict()
    await db_session.commit()

    checkpoint, last_iteration = await load_agent_checkpoint(
        db=db_session,
        run=run,
    )

    assert checkpoint == round_two_checkpoint()
    assert last_iteration == 2


# -------------------------
# Workflow: crash, then resume
# -------------------------


async def load_runs(db_session, task_id: int) -> list[AgentRun]:

    db_session.expire_all()

    return list(
        (
            await db_session.execute(
                select(AgentRun)
                .where(AgentRun.task_id == task_id)
                .order_by(AgentRun.id)
            )
        ).scalars().all()
    )


async def load_step_queries(db_session, task_id: int) -> list[str]:

    return [
        step.query
        for step in (
            await db_session.execute(
                select(ResearchStep)
                .where(ResearchStep.task_id == task_id)
                .order_by(ResearchStep.id)
            )
        ).scalars().all()
    ]


async def crash_a_run(db_session, monkeypatch):
    """A task whose run crashed after round 2; returns (task, run)."""

    monkeypatch.setattr(workflow_service, "OpenAIProvider", CrashOnThirdCallLLM)
    monkeypatch.setattr(workflow_service, "ToolRegistry", FakeRegistry)

    task = await create_task(db_session)

    await execute_research_workflow(db=db_session, task=task)

    [crashed_run] = await load_runs(db_session, task.id)

    await db_session.refresh(task)

    return task, crashed_run


@pytest.mark.asyncio
async def test_crashed_run_keeps_totals_from_its_checkpoint(
    db_session,
    monkeypatch,
):

    task, crashed_run = await crash_a_run(db_session, monkeypatch)

    # No AgentResult after the crash, but the last checkpoint (after
    # iteration 2) says how far the run got, so it isn't recorded as 0.
    assert crashed_run.status == AgentRunStatus.FAILED
    assert crashed_run.iteration_count == 2
    assert crashed_run.tool_call_count == 2
    assert crashed_run.total_tokens == 300

    # And the checkpoint stays, ready to resume.
    assert crashed_run.state["iteration"] == 2


@pytest.mark.asyncio
async def test_resume_finishes_a_crashed_task(db_session, monkeypatch):

    task, crashed_run = await crash_a_run(db_session, monkeypatch)
    task_id = task.id
    crashed_run_id = crashed_run.id

    assert task.status == TaskStatus.FAILED

    AnswerLLM.instances.clear()
    monkeypatch.setattr(workflow_service, "OpenAIProvider", AnswerLLM)

    returned = await resume_research_workflow(
        db=db_session,
        task=task,
        agent_run=crashed_run,
    )

    assert returned is task
    assert task.status == TaskStatus.COMPLETED
    assert task.summary == "Resumed answer."

    # The model picked up the conversation where the crash left it.
    [llm] = AnswerLLM.instances
    assert len(llm.requests[0]) == 5
    assert llm.requests[0][-1]["call_id"] == "call_2"

    # The two searches from before the crash were not run again.
    assert await load_step_queries(db_session, task_id) == ["rag 1", "rag 2"]

    # Still one run: the same one, resumed in place. It records the whole
    # job, and its checkpoint is cleared now that it has completed.
    [run] = await load_runs(db_session, task_id)

    assert run.id == crashed_run_id
    assert run.status == AgentRunStatus.COMPLETED
    assert run.error is None
    assert run.iteration_count == 3
    assert run.tool_call_count == 2
    assert run.total_tokens == 300 + 150
    assert run.state is None
    assert run.state_updated_at is None


@pytest.mark.asyncio
async def test_failed_resume_can_be_resumed_again(db_session, monkeypatch):

    task, crashed_run = await crash_a_run(db_session, monkeypatch)
    task_id = task.id
    run_id = crashed_run.id

    # The resume crashes too (on its own 3rd call, after 2 more searches).
    await resume_research_workflow(
        db=db_session,
        task=task,
        agent_run=crashed_run,
    )

    assert task.status == TaskStatus.FAILED

    [run] = await load_runs(db_session, task_id)

    assert run.id == run_id
    assert run.status == AgentRunStatus.FAILED
    assert run.error == "OpenAI connection reset"

    # The same run's checkpoint moved on to the end of iteration 4.
    checkpoint, last_iteration = await load_agent_checkpoint(
        db=db_session,
        run=run,
    )

    assert last_iteration == 4
    assert checkpoint.executed_tool_calls == 4
    assert run.total_tokens == 600

    # Resume it once more.
    monkeypatch.setattr(workflow_service, "OpenAIProvider", AnswerLLM)

    await db_session.refresh(task)

    await resume_research_workflow(db=db_session, task=task, agent_run=run)

    assert task.status == TaskStatus.COMPLETED

    [run] = await load_runs(db_session, task_id)

    assert run.id == run_id
    assert run.status == AgentRunStatus.COMPLETED
    assert run.iteration_count == 5


@pytest.mark.asyncio
async def test_resume_without_answer_fails_task_again(db_session, monkeypatch):

    task, crashed_run = await crash_a_run(db_session, monkeypatch)

    monkeypatch.setattr(workflow_service, "OpenAIProvider", NeverAnswersLLM)

    await resume_research_workflow(
        db=db_session,
        task=task,
        agent_run=crashed_run,
    )

    assert task.status == TaskStatus.FAILED
    assert task.summary is None


# -------------------------
# Workflow: refused resumes
# -------------------------


@pytest.mark.asyncio
async def test_completed_task_cannot_be_resumed(db_session, monkeypatch):

    task, crashed_run = await crash_a_run(db_session, monkeypatch)
    task_id = task.id

    monkeypatch.setattr(workflow_service, "OpenAIProvider", AnswerLLM)

    await resume_research_workflow(db=db_session, task=task, agent_run=crashed_run)

    with pytest.raises(ValueError, match="only failed tasks can be resumed"):
        await resume_research_workflow(
            db=db_session,
            task=task,
            agent_run=crashed_run,
        )

    # Nothing changed: still completed, still one completed run.
    assert task.status == TaskStatus.COMPLETED

    [run] = await load_runs(db_session, task_id)

    assert run.status == AgentRunStatus.COMPLETED


@pytest.mark.asyncio
async def test_run_without_checkpoint_cannot_be_resumed(db_session, monkeypatch):

    monkeypatch.setattr(workflow_service, "ToolRegistry", FakeRegistry)

    # Fails on the very first LLM call, so no round ever finished.
    from app.tests.test_workflow_service import BrokenLLM

    monkeypatch.setattr(workflow_service, "OpenAIProvider", BrokenLLM)

    task = await create_task(db_session)
    task_id = task.id

    await execute_research_workflow(db=db_session, task=task)

    [run] = await load_runs(db_session, task_id)
    await db_session.refresh(task)

    with pytest.raises(ValueError, match="no checkpoint to resume"):
        await resume_research_workflow(db=db_session, task=task, agent_run=run)

    assert task.status == TaskStatus.FAILED
    assert len(await load_runs(db_session, task_id)) == 1


@pytest.mark.asyncio
async def test_run_from_another_task_is_refused(db_session, monkeypatch):

    _, crashed_run = await crash_a_run(db_session, monkeypatch)

    other_task, _ = await crash_a_run(db_session, monkeypatch)

    with pytest.raises(ValueError, match="belongs to task"):
        await resume_research_workflow(
            db=db_session,
            task=other_task,
            agent_run=crashed_run,
        )


@pytest.mark.asyncio
async def test_corrupted_checkpoint_is_refused(db_session, monkeypatch):

    task, crashed_run = await crash_a_run(db_session, monkeypatch)
    task_id = task.id

    # Damage the saved state, e.g. a bad manual edit in the database.
    state = dict(crashed_run.state)
    state["input_items"] = "not a conversation"
    crashed_run.state = state
    await db_session.commit()

    monkeypatch.setattr(workflow_service, "OpenAIProvider", AnswerLLM)

    with pytest.raises(ValueError, match="Checkpoint input_items must be a list"):
        await resume_research_workflow(
            db=db_session,
            task=task,
            agent_run=crashed_run,
        )

    # Refused before anything changed: still failed, no new run.
    await db_session.refresh(task)

    assert task.status == TaskStatus.FAILED
    assert len(await load_runs(db_session, task_id)) == 1


@pytest.mark.asyncio
async def test_completed_run_cannot_be_resumed(db_session, monkeypatch):

    # A task can fail after its run completed (e.g. saving the answer
    # failed); the run itself has nothing left to resume.
    task, crashed_run = await crash_a_run(db_session, monkeypatch)

    monkeypatch.setattr(workflow_service, "OpenAIProvider", AnswerLLM)
    await resume_research_workflow(db=db_session, task=task, agent_run=crashed_run)

    task.status = TaskStatus.FAILED
    await db_session.commit()

    with pytest.raises(ValueError, match="only failed runs can be resumed"):
        await resume_research_workflow(
            db=db_session,
            task=task,
            agent_run=crashed_run,
        )
