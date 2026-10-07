import pytest

from app.db.models import ResearchStep, ResearchTask, StepStatus
from app.services.step_service import create_agent_step, update_step_status


@pytest.mark.asyncio
async def test_create_agent_step(db_session):

    task = ResearchTask(question="What is RAG?")

    db_session.add(task)

    await db_session.flush()

    step = await create_agent_step(
        db=db_session,
        task_id=task.id,
        tool="tavily",
        query="retrieval augmented generation",
        iteration=2,
    )

    # flush() has run, so the step has an id without a commit.
    assert step.id is not None

    saved = await db_session.get(ResearchStep, step.id)

    assert saved.task_id == task.id
    assert saved.tool == "tavily"
    assert saved.query == "retrieval augmented generation"
    assert saved.iteration == 2
    assert saved.status == StepStatus.RUNNING


@pytest.mark.asyncio
async def test_update_step_status(db_session):

    task = ResearchTask(question="What is RAG?")

    db_session.add(task)

    await db_session.flush()

    step = await create_agent_step(
        db=db_session,
        task_id=task.id,
        tool="arxiv",
        query="rag",
        iteration=1,
    )

    await update_step_status(
        db=db_session,
        step=step,
        status=StepStatus.COMPLETED,
    )

    await db_session.refresh(step)

    assert step.status == StepStatus.COMPLETED
