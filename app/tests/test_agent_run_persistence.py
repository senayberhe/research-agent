import pytest
from sqlalchemy import select

from app.core.config import settings
from app.db.models import AgentRun, AgentRunStatus, TaskStatus
from app.services import workflow_service
from app.services.workflow_service import execute_research_workflow
from app.tests.fake_agent_llm import FakeAgentLLM
from app.tests.fake_agent_tools import FakeRegistry
from app.tests.test_agent_token_budget import SearchForeverLLM
from app.tests.test_workflow_service import BrokenLLM, create_task


async def load_runs(db_session, task_id: int) -> list[AgentRun]:

    # Fresh SELECT, so the values come from the database, not the session.
    db_session.expire_all()

    result = await db_session.execute(
        select(AgentRun)
        .where(AgentRun.task_id == task_id)
        .order_by(AgentRun.id)
    )

    return list(result.scalars().all())


@pytest.mark.asyncio
async def test_successful_run_saves_token_usage(db_session, monkeypatch):

    monkeypatch.setattr(workflow_service, "OpenAIProvider", FakeAgentLLM)
    monkeypatch.setattr(workflow_service, "ToolRegistry", FakeRegistry)

    task = await create_task(db_session)
    task_id = task.id

    await execute_research_workflow(db=db_session, task=task)

    runs = await load_runs(db_session, task_id)

    assert len(runs) == 1

    run = runs[0]

    assert run.status == AgentRunStatus.COMPLETED
    assert run.iteration_count == 2
    assert run.tool_call_count == 1

    # Two FakeAgentLLM calls of 100 input + 50 output tokens.
    assert run.input_tokens == 200
    assert run.output_tokens == 100
    assert run.total_tokens == 300

    # FakeAgentLLM is gpt-5.4-mini: 200 * $0.75/1M + 100 * $4.50/1M.
    assert run.estimated_cost_usd == pytest.approx(0.0006)

    assert run.error is None
    assert run.completed_at is not None
    assert run.completed_at >= run.started_at


@pytest.mark.asyncio
async def test_over_budget_run_is_saved_as_failed_with_usage(
    db_session,
    monkeypatch,
):

    monkeypatch.setattr(workflow_service, "OpenAIProvider", SearchForeverLLM)
    monkeypatch.setattr(workflow_service, "ToolRegistry", FakeRegistry)
    monkeypatch.setattr(settings, "agent_max_total_tokens", 2500)

    task = await create_task(db_session)
    task_id = task.id

    await execute_research_workflow(db=db_session, task=task)

    assert task.status == TaskStatus.FAILED

    [run] = await load_runs(db_session, task_id)

    # 3 calls of 800 input + 200 output before the budget stopped the run.
    assert run.status == AgentRunStatus.FAILED
    assert run.input_tokens == 2400
    assert run.output_tokens == 600
    assert run.total_tokens == 3000
    assert run.error == "Agent token budget exceeded: 3000 >= 2500"
    assert run.completed_at is not None


@pytest.mark.asyncio
async def test_run_that_raises_is_saved_as_failed(db_session, monkeypatch):

    monkeypatch.setattr(workflow_service, "OpenAIProvider", BrokenLLM)
    monkeypatch.setattr(workflow_service, "ToolRegistry", FakeRegistry)

    task = await create_task(db_session)
    task_id = task.id

    await execute_research_workflow(db=db_session, task=task)

    [run] = await load_runs(db_session, task_id)

    assert run.status == AgentRunStatus.FAILED
    assert run.error == "LLM service unavailable"
    assert run.total_tokens == 0
    assert run.completed_at is not None


@pytest.mark.asyncio
async def test_failure_after_agent_finishes_keeps_token_usage(
    db_session,
    monkeypatch,
):

    # The agent finishes normally, then saving the completed run fails.
    async def broken_complete_agent_run(**kwargs):
        raise RuntimeError("database write failed")

    monkeypatch.setattr(workflow_service, "OpenAIProvider", FakeAgentLLM)
    monkeypatch.setattr(workflow_service, "ToolRegistry", FakeRegistry)
    monkeypatch.setattr(
        workflow_service,
        "complete_agent_run",
        broken_complete_agent_run,
    )

    task = await create_task(db_session)
    task_id = task.id

    await execute_research_workflow(db=db_session, task=task)

    [run] = await load_runs(db_session, task_id)

    # The agent's real counts and tokens are saved, not zeros.
    assert run.status == AgentRunStatus.FAILED
    assert run.error == "database write failed"
    assert run.iteration_count == 2
    assert run.tool_call_count == 1
    assert run.input_tokens == 200
    assert run.output_tokens == 100
    assert run.total_tokens == 300
