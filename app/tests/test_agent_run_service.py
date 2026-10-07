import pytest
from sqlalchemy import select

from app.db.models import AgentRun, AgentRunStatus
from app.services.agent_run_service import (
    clear_agent_checkpoint,
    complete_agent_run,
    create_agent_run,
    fail_agent_run,
    reopen_agent_run,
    save_agent_checkpoint,
    update_agent_run_status,
)
from app.services.state_machine import InvalidStateTransition
from app.tests.test_workflow_service import create_task


async def load_run(db_session, run_id: int) -> AgentRun:

    # Fresh SELECT, so the values come from the database, not the session.
    db_session.expire_all()

    result = await db_session.execute(
        select(AgentRun).where(AgentRun.id == run_id)
    )

    return result.scalar_one()


@pytest.mark.asyncio
async def test_create_agent_run(db_session):

    task = await create_task(db_session)
    task_id = task.id

    run = await create_agent_run(db=db_session, task_id=task_id)
    await db_session.commit()

    run = await load_run(db_session, run.id)

    assert run.task_id == task_id
    assert run.status == AgentRunStatus.CREATED
    assert run.completed_at is None


@pytest.mark.asyncio
async def test_complete_agent_run_saves_token_usage(db_session):

    task = await create_task(db_session)

    run = await create_agent_run(db=db_session, task_id=task.id)
    await update_agent_run_status(
        db=db_session,
        run=run,
        status=AgentRunStatus.RUNNING,
    )

    await complete_agent_run(
        db=db_session,
        run=run,
        iteration_count=3,
        tool_call_count=2,
        input_tokens=1000,
        output_tokens=500,
        total_tokens=1500,
        estimated_cost_usd=0.00525,
    )
    await db_session.commit()

    run = await load_run(db_session, run.id)

    assert run.status == AgentRunStatus.COMPLETED
    assert run.iteration_count == 3
    assert run.tool_call_count == 2
    assert run.input_tokens == 1000
    assert run.output_tokens == 500
    assert run.total_tokens == 1500
    assert run.estimated_cost_usd == pytest.approx(0.00525)
    assert run.error is None
    assert run.completed_at is not None


@pytest.mark.asyncio
async def test_fail_agent_run_saves_error_and_token_usage(db_session):

    task = await create_task(db_session)

    run = await create_agent_run(db=db_session, task_id=task.id)

    await fail_agent_run(
        db=db_session,
        run=run,
        error="Token budget of 20000 exceeded (used 21000 tokens).",
        iteration_count=4,
        tool_call_count=5,
        input_tokens=1000,
        output_tokens=500,
        total_tokens=1500,
        estimated_cost_usd=0.00525,
    )
    await db_session.commit()

    run = await load_run(db_session, run.id)

    assert run.status == AgentRunStatus.FAILED
    assert run.error == "Token budget of 20000 exceeded (used 21000 tokens)."
    assert run.iteration_count == 4
    assert run.tool_call_count == 5
    assert run.input_tokens == 1000
    assert run.output_tokens == 500
    assert run.total_tokens == 1500
    assert run.estimated_cost_usd == pytest.approx(0.00525)
    assert run.completed_at is not None


@pytest.mark.asyncio
async def test_complete_agent_run_saves_unknown_cost_as_null(db_session):

    task = await create_task(db_session)

    run = await create_agent_run(db=db_session, task_id=task.id)
    await update_agent_run_status(
        db=db_session,
        run=run,
        status=AgentRunStatus.RUNNING,
    )

    await complete_agent_run(
        db=db_session,
        run=run,
        iteration_count=1,
        tool_call_count=0,
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        estimated_cost_usd=None,
    )
    await db_session.commit()

    run = await load_run(db_session, run.id)

    assert run.total_tokens == 150
    assert run.estimated_cost_usd is None


@pytest.mark.asyncio
async def test_update_agent_run_status_follows_the_state_machine(db_session):

    task = await create_task(db_session)

    run = await create_agent_run(db=db_session, task_id=task.id)

    for status in [
        AgentRunStatus.RUNNING,
        # Already RUNNING: allowed, no change.
        AgentRunStatus.RUNNING,
        AgentRunStatus.WAITING_FOR_TOOL,
        AgentRunStatus.PROCESSING_RESULT,
    ]:
        await update_agent_run_status(db=db_session, run=run, status=status)

    await db_session.commit()

    assert (await load_run(db_session, run.id)).status == (
        AgentRunStatus.PROCESSING_RESULT
    )

    with pytest.raises(InvalidStateTransition):
        await update_agent_run_status(
            db=db_session,
            run=run,
            status=AgentRunStatus.COMPLETED,
        )


@pytest.mark.asyncio
async def test_reopen_agent_run_keeps_checkpoint_and_totals(db_session):

    task = await create_task(db_session)

    run = await create_agent_run(db=db_session, task_id=task.id)

    await save_agent_checkpoint(db=db_session, run=run, state={"iteration": 2})

    await fail_agent_run(
        db=db_session,
        run=run,
        error="OpenAI connection reset",
        iteration_count=2,
        tool_call_count=2,
        input_tokens=200,
        output_tokens=100,
        total_tokens=300,
        estimated_cost_usd=0.001,
    )

    await reopen_agent_run(db=db_session, run=run)
    await db_session.commit()

    run = await load_run(db_session, run.id)

    # Back to RUNNING; the failed attempt's error and end time are gone.
    assert run.status == AgentRunStatus.RUNNING
    assert run.error is None
    assert run.completed_at is None

    # Everything needed to continue is still there.
    assert run.state == {"iteration": 2}
    assert run.total_tokens == 300
    assert run.iteration_count == 2


@pytest.mark.asyncio
async def test_only_failed_runs_can_be_reopened(db_session):

    task = await create_task(db_session)

    run = await create_agent_run(db=db_session, task_id=task.id)

    # A run that is still going can't be reopened.
    await update_agent_run_status(
        db=db_session,
        run=run,
        status=AgentRunStatus.RUNNING,
    )

    with pytest.raises(InvalidStateTransition):
        await reopen_agent_run(db=db_session, run=run)


@pytest.mark.asyncio
async def test_clear_agent_checkpoint(db_session):

    task = await create_task(db_session)

    run = await create_agent_run(db=db_session, task_id=task.id)

    await save_agent_checkpoint(db=db_session, run=run, state={"iteration": 1})
    await clear_agent_checkpoint(db=db_session, run=run)
    await db_session.commit()

    run = await load_run(db_session, run.id)

    assert run.state is None
    assert run.state_updated_at is None
