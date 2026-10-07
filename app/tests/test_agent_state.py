"""The agent_runs.state column across a run's life.

How checkpoints are taken and validated is in test_agent_checkpoint.py,
status transitions in test_state_machine.py, and resuming in
test_resume_workflow.py. This file covers the stored state itself: when it
is empty, how it is replaced, that it survives the database unchanged, and
that each run keeps its own.
"""

import pytest
from sqlalchemy import select

from app.agents.research_agent import AgentCheckpoint
from app.db.models import AgentRun, AgentRunStatus
from app.services import workflow_service
from app.services.agent_run_service import (
    create_agent_run,
    load_agent_checkpoint,
    save_agent_checkpoint,
)
from app.services.workflow_service import (
    execute_research_workflow,
    resume_research_workflow,
)
from app.tests.fake_agent_tools import FakeRegistry
from app.tests.test_agent_checkpoint import CrashOnThirdCallLLM
from app.tests.test_agent_token_budget import SearchForeverLLM
from app.tests.test_resume_workflow import AnswerLLM
from app.tests.test_workflow_service import create_task


def make_checkpoint(iteration: int, **usage) -> AgentCheckpoint:

    items = [{"role": "user", "content": "What is RAG?"}]

    for n in range(1, iteration + 1):
        items += [
            {
                "type": "function_call",
                "name": "tavily_search",
                "arguments": f'{{"query": "rag {n}"}}',
                "call_id": f"call_{n}",
            },
            {
                "type": "function_call_output",
                "call_id": f"call_{n}",
                "output": f"result {n}",
            },
        ]

    return AgentCheckpoint(
        iteration=iteration,
        executed_tool_calls=iteration,
        tool_call_count=iteration,
        input_items=items,
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        total_tokens=usage.get("total_tokens", 0),
        estimated_cost=usage.get("estimated_cost", 0.0),
    )


async def reload(db_session, run_id: int) -> AgentRun:

    # Fresh SELECT, so values come from the database, not the session.
    db_session.expire_all()

    return (
        await db_session.execute(select(AgentRun).where(AgentRun.id == run_id))
    ).scalar_one()


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


# -------------------------
# Saving and loading state
# -------------------------


@pytest.mark.asyncio
async def test_new_run_has_no_state(db_session):

    task = await create_task(db_session)

    run = await create_agent_run(db=db_session, task_id=task.id)
    await db_session.commit()

    run = await reload(db_session, run.id)

    assert run.state is None
    assert run.state_updated_at is None


@pytest.mark.asyncio
async def test_state_survives_the_database_unchanged(db_session):

    task = await create_task(db_session)

    run = await create_agent_run(db=db_session, task_id=task.id)

    # Things JSONB must keep exactly: unknown cost (null), non-ASCII text,
    # quotes and newlines inside tool output, nested structures.
    checkpoint = make_checkpoint(
        1,
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        estimated_cost=None,
    )
    checkpoint.input_items[2]["output"] = 'Résumé: "RAG"\nretrieves → evidence 📚'
    checkpoint.input_items.append(
        {"id": "rs_1", "type": "reasoning", "summary": [{"type": "summary_text", "text": "t"}]}
    )

    await save_agent_checkpoint(
        db=db_session,
        run=run,
        state=checkpoint.to_dict(),
    )
    await db_session.commit()

    run = await reload(db_session, run.id)

    assert run.state == checkpoint.to_dict()
    assert AgentCheckpoint.from_dict(run.state) == checkpoint


@pytest.mark.asyncio
async def test_each_save_replaces_state_and_moves_timestamp(db_session):

    task = await create_task(db_session)

    run = await create_agent_run(db=db_session, task_id=task.id)
    run_id = run.id

    await save_agent_checkpoint(
        db=db_session,
        run=run,
        state=make_checkpoint(1).to_dict(),
    )
    await db_session.commit()

    first_saved_at = (await reload(db_session, run_id)).state_updated_at

    run = await reload(db_session, run_id)

    await save_agent_checkpoint(
        db=db_session,
        run=run,
        state=make_checkpoint(2).to_dict(),
    )
    await db_session.commit()

    run = await reload(db_session, run_id)

    # Only the latest checkpoint is kept.
    assert run.state["iteration"] == 2
    assert len(run.state["input_items"]) == 5
    assert run.state_updated_at >= first_saved_at


@pytest.mark.asyncio
async def test_load_returns_none_until_first_round(db_session):

    task = await create_task(db_session)

    run = await create_agent_run(db=db_session, task_id=task.id)
    await db_session.commit()

    assert await load_agent_checkpoint(db=db_session, run=run) == (None, 0)


# -------------------------
# State during a workflow
# -------------------------


@pytest.mark.asyncio
async def test_state_follows_the_run_round_by_round(db_session, monkeypatch):

    monkeypatch.setattr(workflow_service, "OpenAIProvider", SearchForeverLLM)
    monkeypatch.setattr(workflow_service, "ToolRegistry", FakeRegistry)
    monkeypatch.setattr(workflow_service.settings, "agent_max_iterations", 3)

    # Record the state as it was committed after each round.
    saved_rounds = []
    real_persist = workflow_service.persist_agent_checkpoint

    async def recording_persist(db, run, checkpoint):
        await real_persist(db=db, run=run, checkpoint=checkpoint)

        # Reads the column straight from the database without expiring the
        # session's objects, which the running workflow is still using.
        state = await db_session.scalar(
            select(AgentRun.state).where(AgentRun.id == run.id)
        )
        saved_rounds.append(state["iteration"])

    monkeypatch.setattr(
        workflow_service,
        "persist_agent_checkpoint",
        recording_persist,
    )

    task = await create_task(db_session)
    task_id = task.id

    await execute_research_workflow(db=db_session, task=task)

    # Saved and committed after every round, not only at the end.
    assert saved_rounds == [1, 2, 3]

    [run] = await load_runs(db_session, task_id)

    # Out of iterations: the run failed, its last state is round 3.
    assert run.status == AgentRunStatus.FAILED
    assert run.state["iteration"] == 3
    assert run.state["total_tokens"] == 3000


@pytest.mark.asyncio
async def test_state_is_kept_after_the_run_ends(db_session, monkeypatch):

    monkeypatch.setattr(workflow_service, "OpenAIProvider", CrashOnThirdCallLLM)
    monkeypatch.setattr(workflow_service, "ToolRegistry", FakeRegistry)

    task = await create_task(db_session)
    task_id = task.id

    await execute_research_workflow(db=db_session, task=task)

    [run] = await load_runs(db_session, task_id)

    # Finishing (here: failing) doesn't clear the state, so it can still be
    # resumed or inspected afterwards.
    assert run.status == AgentRunStatus.FAILED
    assert run.state is not None
    assert run.state_updated_at <= run.completed_at


@pytest.mark.asyncio
async def test_resume_continues_the_same_runs_state(db_session, monkeypatch):

    monkeypatch.setattr(workflow_service, "OpenAIProvider", CrashOnThirdCallLLM)
    monkeypatch.setattr(workflow_service, "ToolRegistry", FakeRegistry)

    task = await create_task(db_session)
    task_id = task.id

    await execute_research_workflow(db=db_session, task=task)

    [run] = await load_runs(db_session, task_id)
    first_state = run.state

    assert first_state["iteration"] == 2

    # The resume crashes too, after two more iterations.
    await db_session.refresh(task)
    await resume_research_workflow(db=db_session, task=task, agent_run=run)

    [run] = await load_runs(db_session, task_id)

    # Same run; its state moved on and still starts with the earlier
    # conversation.
    assert run.status == AgentRunStatus.FAILED
    assert run.state["iteration"] == 4
    assert run.state["input_items"][:5] == first_state["input_items"]


@pytest.mark.asyncio
async def test_successful_run_clears_its_checkpoint(db_session, monkeypatch):

    monkeypatch.setattr(workflow_service, "OpenAIProvider", CrashOnThirdCallLLM)
    monkeypatch.setattr(workflow_service, "ToolRegistry", FakeRegistry)

    task = await create_task(db_session)
    task_id = task.id

    await execute_research_workflow(db=db_session, task=task)

    [run] = await load_runs(db_session, task_id)

    monkeypatch.setattr(workflow_service, "OpenAIProvider", AnswerLLM)

    await db_session.refresh(task)
    await resume_research_workflow(db=db_session, task=task, agent_run=run)

    [run] = await load_runs(db_session, task_id)

    # Completed: nothing left to resume, so the checkpoint is gone. The
    # run's own columns still hold the whole job's totals.
    assert run.status == AgentRunStatus.COMPLETED
    assert run.state is None
    assert run.state_updated_at is None
    assert run.total_tokens == 450


@pytest.mark.asyncio
async def test_resume_that_crashes_straight_away_can_be_resumed(
    db_session,
    monkeypatch,
):

    from app.tests.test_workflow_service import BrokenLLM

    monkeypatch.setattr(workflow_service, "OpenAIProvider", CrashOnThirdCallLLM)
    monkeypatch.setattr(workflow_service, "ToolRegistry", FakeRegistry)

    task = await create_task(db_session)
    task_id = task.id

    await execute_research_workflow(db=db_session, task=task)

    [run] = await load_runs(db_session, task_id)
    first_state = run.state

    # The resume fails on its very first LLM call, before any iteration of
    # its own finishes.
    monkeypatch.setattr(workflow_service, "OpenAIProvider", BrokenLLM)

    await db_session.refresh(task)
    await resume_research_workflow(db=db_session, task=task, agent_run=run)

    [run] = await load_runs(db_session, task_id)

    assert run.status == AgentRunStatus.FAILED
    assert run.error == "LLM service unavailable"

    # The checkpoint is untouched, and the totals still come from it (not
    # reset to 0 by the crash).
    assert run.state == first_state
    assert run.total_tokens == 300

    # So it can be resumed again.
    monkeypatch.setattr(workflow_service, "OpenAIProvider", AnswerLLM)

    await db_session.refresh(task)
    await resume_research_workflow(db=db_session, task=task, agent_run=run)

    [run] = await load_runs(db_session, task_id)

    assert run.status == AgentRunStatus.COMPLETED


# -------------------------
# Run status (agent phases)
# -------------------------


@pytest.mark.asyncio
async def test_agent_reports_its_phases_in_order():

    from app.agents.research_agent import AgentPhase, ResearchAgent
    from app.tests.fake_agent_llm import FakeAgentLLM

    phases = []

    async def record(phase):
        phases.append(phase)

    await ResearchAgent(
        llm=FakeAgentLLM(),
        registry=FakeRegistry(),
        on_status_change=record,
    ).run(question="What is RAG?")

    # Iteration 1 searches; iteration 2 answers.
    assert phases == [
        AgentPhase.RUNNING,
        AgentPhase.WAITING_FOR_TOOL,
        AgentPhase.PROCESSING_RESULT,
        AgentPhase.RUNNING,
    ]


@pytest.mark.asyncio
async def test_workflow_saves_each_status_as_it_happens(db_session, monkeypatch):

    from app.tests.fake_agent_llm import FakeAgentLLM

    monkeypatch.setattr(workflow_service, "OpenAIProvider", FakeAgentLLM)
    monkeypatch.setattr(workflow_service, "ToolRegistry", FakeRegistry)

    # The status as committed to the database after each change.
    saved = []
    real_persist = workflow_service.persist_agent_status

    async def recording_persist(db, run, phase):
        await real_persist(db=db, run=run, phase=phase)
        saved.append(
            await db_session.scalar(
                select(AgentRun.status).where(AgentRun.id == run.id)
            )
        )

    monkeypatch.setattr(
        workflow_service,
        "persist_agent_status",
        recording_persist,
    )

    task = await create_task(db_session)
    task_id = task.id

    await execute_research_workflow(db=db_session, task=task)

    assert saved == [
        AgentRunStatus.RUNNING,
        AgentRunStatus.WAITING_FOR_TOOL,
        AgentRunStatus.PROCESSING_RESULT,
        AgentRunStatus.RUNNING,
    ]

    [run] = await load_runs(db_session, task_id)

    assert run.status == AgentRunStatus.COMPLETED


@pytest.mark.asyncio
async def test_new_run_starts_as_created(db_session, monkeypatch):

    from app.tests.test_workflow_service import BrokenLLM

    monkeypatch.setattr(workflow_service, "OpenAIProvider", BrokenLLM)
    monkeypatch.setattr(workflow_service, "ToolRegistry", FakeRegistry)

    # The status the run had when the agent first reported in.
    first_seen = []
    real_persist = workflow_service.persist_agent_status

    async def recording_persist(db, run, phase):
        if not first_seen:
            first_seen.append(
                await db_session.scalar(
                    select(AgentRun.status).where(AgentRun.id == run.id)
                )
            )
        await real_persist(db=db, run=run, phase=phase)

    monkeypatch.setattr(
        workflow_service,
        "persist_agent_status",
        recording_persist,
    )

    task = await create_task(db_session)
    task_id = task.id

    await execute_research_workflow(db=db_session, task=task)

    # CREATED until the agent starts; the LLM then failed while RUNNING.
    assert first_seen == [AgentRunStatus.CREATED]

    [run] = await load_runs(db_session, task_id)

    assert run.status == AgentRunStatus.FAILED
