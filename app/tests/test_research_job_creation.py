import pytest
from sqlalchemy import select

from app.db.models import (
    JobStatus,
    ResearchJob,
)
from app.services.research_service import (
    create_research_task,
)


@pytest.mark.asyncio
async def test_creating_research_task_creates_job(db_session):

    db = db_session

    task = await create_research_task(
        db=db,
        question="What is retrieval augmented generation?",
    )

    result = await db.execute(
        select(ResearchJob).where(
            ResearchJob.task_id == task.id
        )
    )

    job = result.scalar_one()

    assert job.task_id == task.id
    assert job.status == JobStatus.PENDING
    assert job.attempts == 0
