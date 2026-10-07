import pytest
from sqlalchemy import select

from app.db.models import AgentRun, AgentRunStatus
from app.tests.test_workflow_service import create_task


async def load_run(db_session, run_id: int) -> AgentRun:

    # Fresh SELECT, so the values come from the database, not the session.
    db_session.expire_all()

    result = await db_session.execute(
        select(AgentRun).where(AgentRun.id == run_id)
    )

    return result.scalar_one()


@pytest.mark.asyncio
async def test_agent_run_saves_token_usage_and_cost(db_session):

    task = await create_task(db_session)

    # Read before load_run expires everything in the session.
    task_id = task.id

    run = AgentRun(
        task_id=task_id,
        status=AgentRunStatus.COMPLETED,
        iteration_count=2,
        tool_call_count=1,
        input_tokens=300,
        output_tokens=70,
        total_tokens=370,
        estimated_cost_usd=0.00116,
    )

    db_session.add(run)
    await db_session.commit()

    saved = await load_run(db_session, run.id)

    assert saved.task_id == task_id
    assert saved.status == AgentRunStatus.COMPLETED
    assert saved.input_tokens == 300
    assert saved.output_tokens == 70
    assert saved.total_tokens == 370
    assert saved.estimated_cost_usd == pytest.approx(0.00116)


@pytest.mark.asyncio
async def test_agent_run_token_fields_default_to_zero(db_session):

    task = await create_task(db_session)

    run = AgentRun(task_id=task.id)

    db_session.add(run)
    await db_session.commit()

    saved = await load_run(db_session, run.id)

    assert saved.status == AgentRunStatus.CREATED
    assert saved.input_tokens == 0
    assert saved.output_tokens == 0
    assert saved.total_tokens == 0
    # NULL until a cost is saved; NULL also means "pricing unknown".
    assert saved.estimated_cost_usd is None
    assert saved.completed_at is None
