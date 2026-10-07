import pytest
from sqlalchemy import select

from app.db.models import ResearchTask
from app.jobs.models import JobStatus, ResearchJob
from app.tests.test_workflow_service import create_task


@pytest.mark.asyncio
async def test_research_job_defaults(db_session):

    task = await create_task(db_session)
    task_id = task.id

    job = ResearchJob(task_id=task_id)
    db_session.add(job)
    await db_session.commit()
    job_id = job.id

    # Fresh SELECT, so the defaults come from what was stored.
    db_session.expire_all()

    job = await db_session.scalar(
        select(ResearchJob).where(ResearchJob.id == job_id)
    )

    assert job.task_id == task_id
    assert job.status == JobStatus.PENDING
    assert job.attempts == 0
    assert job.worker_id is None
    assert job.lease_token is None
    assert job.lease_expires_at is None
    assert job.error is None
    assert job.created_at is not None
    assert job.started_at is None
    assert job.completed_at is None


@pytest.mark.asyncio
async def test_task_jobs_relationship(db_session):

    task = await create_task(db_session)

    db_session.add_all([
        ResearchJob(task_id=task.id),
        ResearchJob(task_id=task.id, status=JobStatus.FAILED, attempts=1),
    ])
    await db_session.commit()

    # Load the relationship explicitly (no lazy loading on async sessions).
    await db_session.refresh(task, attribute_names=["jobs"])

    assert [job.status for job in task.jobs] == [
        JobStatus.PENDING,
        JobStatus.FAILED,
    ]
    assert all(job.task is task for job in task.jobs)


@pytest.mark.asyncio
async def test_deleting_task_deletes_its_jobs(db_session):

    task = await create_task(db_session)

    db_session.add(ResearchJob(task_id=task.id))
    await db_session.commit()

    await db_session.refresh(task, attribute_names=["jobs", "steps", "agent_runs"])
    await db_session.delete(task)
    await db_session.commit()

    assert (await db_session.scalar(select(ResearchJob))) is None
    assert (await db_session.scalar(select(ResearchTask))) is None


@pytest.mark.asyncio
async def test_research_job_lease_is_stored(db_session):

    from datetime import timedelta

    from app.db.models import utc_now

    task = await create_task(db_session)

    expires = utc_now() + timedelta(minutes=5)

    job = ResearchJob(
        task_id=task.id,
        lease_token="3f6c0a52-lease",
        lease_expires_at=expires,
    )
    db_session.add(job)
    await db_session.commit()
    job_id = job.id

    db_session.expire_all()

    job = await db_session.scalar(
        select(ResearchJob).where(ResearchJob.id == job_id)
    )

    assert job.lease_token == "3f6c0a52-lease"
    assert job.lease_expires_at == expires
