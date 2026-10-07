import pytest
import pytest_asyncio
from sqlalchemy import select

from app.db.models import ResearchTask
from app.jobs.models import JobStatus, ResearchJob
from app.jobs.service import (
    claim_next_job,
    create_research_job,
    get_next_pending_job,
    get_research_job,
    mark_job_completed,
    mark_job_failed,
    mark_job_running,
)
from app.services.state_machine import InvalidStateTransition
from app.tests.conftest import TestSessionLocal
from app.tests.test_workflow_service import create_task


# Jobs have a foreign key to research_tasks, so each test needs a real task
# (the test database has no task 1).
@pytest_asyncio.fixture
async def task_id(db_session) -> int:
    task = await create_task(db_session)
    return task.id


@pytest.mark.asyncio
async def test_create_research_job(db_session, task_id):

    job = await create_research_job(
        db=db_session,
        task_id=task_id,
    )

    assert job.id is not None
    assert job.task_id == task_id
    assert job.status == JobStatus.PENDING
    assert job.attempts == 0


@pytest.mark.asyncio
async def test_get_research_job(db_session, task_id):

    created = await create_research_job(
        db=db_session,
        task_id=task_id,
    )

    await db_session.commit()

    job = await get_research_job(
        db=db_session,
        job_id=created.id,
    )

    assert job is not None
    assert job.id == created.id


@pytest.mark.asyncio
async def test_get_missing_research_job(db_session):

    assert await get_research_job(db=db_session, job_id=999_999) is None


@pytest.mark.asyncio
async def test_mark_job_running(db_session, task_id):

    job = await create_research_job(
        db=db_session,
        task_id=task_id,
    )

    await mark_job_running(
        db=db_session,
        job=job,
        worker_id="worker-1",
    )

    assert job.status == JobStatus.RUNNING
    assert job.worker_id == "worker-1"
    assert job.attempts == 1
    assert job.started_at is not None


@pytest.mark.asyncio
async def test_mark_job_completed(db_session, task_id):

    job = await create_research_job(
        db=db_session,
        task_id=task_id,
    )

    await mark_job_running(
        db=db_session,
        job=job,
        worker_id="worker-1",
    )

    await mark_job_completed(
        db=db_session,
        job=job,
        lease_token=job.lease_token,
    )

    assert job.status == JobStatus.COMPLETED
    assert job.completed_at is not None


@pytest.mark.asyncio
async def test_mark_job_failed(db_session, task_id):

    job = await create_research_job(
        db=db_session,
        task_id=task_id,
    )

    await mark_job_running(
        db=db_session,
        job=job,
        worker_id="worker-1",
    )

    await mark_job_failed(
        db=db_session,
        job=job,
        error="Tool execution failed",
    )

    assert job.status == JobStatus.FAILED
    assert job.error == "Tool execution failed"
    assert job.completed_at is not None


# -------------------------
# Retries and invalid transitions
# -------------------------


@pytest.mark.asyncio
async def test_failed_job_can_be_retried(db_session, task_id):

    job = await create_research_job(db=db_session, task_id=task_id)

    await mark_job_running(db=db_session, job=job, worker_id="worker-1")
    await mark_job_failed(db=db_session, job=job, error="boom")

    await mark_job_running(db=db_session, job=job, worker_id="worker-2")

    # Second attempt, by another worker, with the old error cleared.
    assert job.status == JobStatus.RUNNING
    assert job.attempts == 2
    assert job.worker_id == "worker-2"
    assert job.error is None
    assert job.completed_at is None


@pytest.mark.asyncio
async def test_pending_job_cannot_complete(db_session, task_id):

    job = await create_research_job(db=db_session, task_id=task_id)

    # Not RUNNING (and no lease): nothing to complete, nothing changed.
    assert await mark_job_completed(
        db=db_session,
        job=job,
        lease_token="any-token",
    ) is False

    assert job.status == JobStatus.PENDING


@pytest.mark.asyncio
async def test_completed_job_is_final(db_session, task_id):

    job = await create_research_job(db=db_session, task_id=task_id)

    await mark_job_running(db=db_session, job=job, worker_id="worker-1")
    await mark_job_completed(
        db=db_session,
        job=job,
        lease_token=job.lease_token,
    )

    with pytest.raises(InvalidStateTransition):
        await mark_job_running(db=db_session, job=job, worker_id="worker-2")

    with pytest.raises(InvalidStateTransition):
        await mark_job_failed(db=db_session, job=job, error="late")


@pytest.mark.asyncio
async def test_running_job_cannot_be_started_twice(db_session, task_id):

    job = await create_research_job(db=db_session, task_id=task_id)

    await mark_job_running(db=db_session, job=job, worker_id="worker-1")

    with pytest.raises(InvalidStateTransition):
        await mark_job_running(db=db_session, job=job, worker_id="worker-2")


# -------------------------
# claim_next_job
# -------------------------


@pytest.mark.asyncio
async def test_claim_next_job_takes_oldest_pending(db_session, task_id):

    first = await create_research_job(db=db_session, task_id=task_id)
    second = await create_research_job(db=db_session, task_id=task_id)

    claimed = await claim_next_job(db=db_session, worker_id="worker-1")

    assert claimed.id == first.id
    assert claimed.status == JobStatus.RUNNING
    assert claimed.worker_id == "worker-1"
    assert claimed.attempts == 1

    # The next claim gets the next job; then there are none left.
    assert (await claim_next_job(db=db_session, worker_id="worker-2")).id == (
        second.id
    )
    assert await claim_next_job(db=db_session, worker_id="worker-3") is None


@pytest.mark.asyncio
async def test_claim_skips_job_locked_by_another_worker(clean_tables):

    # Two real sessions, like two workers (the rolled-back db_session can't
    # show locking between connections).
    async with TestSessionLocal() as setup:

        task = ResearchTask(question="What is RAG?", status="pending")
        setup.add(task)
        await setup.flush()

        locked = ResearchJob(task_id=task.id)
        free = ResearchJob(task_id=task.id)
        setup.add_all([locked, free])
        await setup.commit()

        locked_id, free_id = locked.id, free.id

    async with TestSessionLocal() as worker_a, TestSessionLocal() as worker_b:

        # Worker A is in the middle of claiming the oldest job: it holds
        # the row lock and hasn't committed yet.
        await worker_a.scalar(
            select(ResearchJob)
            .where(ResearchJob.id == locked_id)
            .with_for_update()
        )

        # Worker B doesn't wait for it and doesn't take it twice: it skips
        # to the next pending job.
        claimed = await claim_next_job(db=worker_b, worker_id="worker-b")

        assert claimed.id == free_id

        await worker_a.rollback()


# -------------------------
# create_research_task
# -------------------------


@pytest.mark.asyncio
async def test_create_research_task_saves_task_and_job_together(db_session):

    from app.db.models import TaskStatus
    from app.services.research_service import create_research_task

    task = await create_research_task(
        db=db_session,
        question="How does RAG work?",
    )

    assert task.status == TaskStatus.PENDING

    jobs = (
        await db_session.execute(
            select(ResearchJob).where(ResearchJob.task_id == task.id)
        )
    ).scalars().all()

    [job] = jobs

    assert job.status == JobStatus.PENDING

    # A worker would claim exactly this job.
    claimed = await claim_next_job(db=db_session, worker_id="worker-1")

    assert claimed.id == job.id
    assert claimed.task_id == task.id


# -------------------------
# get_next_pending_job
# -------------------------


@pytest.mark.asyncio
async def test_get_next_pending_job_returns_oldest(db_session, task_id):

    assert await get_next_pending_job(db=db_session) is None

    first = await create_research_job(db=db_session, task_id=task_id)
    await create_research_job(db=db_session, task_id=task_id)

    job = await get_next_pending_job(db=db_session)

    assert job.id == first.id


@pytest.mark.asyncio
async def test_get_next_pending_job_skips_non_pending(db_session, task_id):

    running = await create_research_job(db=db_session, task_id=task_id)
    await mark_job_running(db=db_session, job=running, worker_id="worker-1")

    pending = await create_research_job(db=db_session, task_id=task_id)

    assert (await get_next_pending_job(db=db_session)).id == pending.id


@pytest.mark.asyncio
async def test_get_next_pending_job_does_not_claim(db_session, task_id):

    created = await create_research_job(db=db_session, task_id=task_id)

    job = await get_next_pending_job(db=db_session)

    # Still pending and unclaimed; the same job is what claim_next_job takes.
    assert job.status == JobStatus.PENDING
    assert job.worker_id is None
    assert job.attempts == 0

    claimed = await claim_next_job(db=db_session, worker_id="worker-1")

    assert claimed.id == created.id


@pytest.mark.asyncio
async def test_unlocked_read_does_not_protect_a_claim(clean_tables):

    # Why claim_next_job locks: without the lock, a second worker still
    # sees the job another worker is about to claim.
    async with TestSessionLocal() as setup:

        task = ResearchTask(question="What is RAG?", status="pending")
        setup.add(task)
        await setup.flush()

        job = ResearchJob(task_id=task.id)
        setup.add(job)
        await setup.commit()

        job_id = job.id

    async with TestSessionLocal() as worker_a, TestSessionLocal() as worker_b:

        # Worker A has locked the job (mid-claim).
        assert (await get_next_pending_job(worker_a, lock=True)).id == job_id

        # An unlocked read in worker B returns the same job...
        assert (await get_next_pending_job(worker_b)).id == job_id

        # ...a locked read skips it: no job left to take.
        await worker_b.rollback()
        assert await get_next_pending_job(worker_b, lock=True) is None

        await worker_a.rollback()


# -------------------------
# Lease
# -------------------------


@pytest.mark.asyncio
async def test_claim_gives_job_a_lease(db_session, task_id, monkeypatch):

    import uuid
    from datetime import timedelta

    from app.core.config import settings

    monkeypatch.setattr(settings, "worker_lease_seconds", 120)

    await create_research_job(db=db_session, task_id=task_id)

    job = await claim_next_job(db=db_session, worker_id="worker-1")

    # A real UUID...
    assert str(uuid.UUID(job.lease_token)) == job.lease_token

    # ...and a lease of WORKER_LEASE_SECONDS from the moment it started.
    assert job.lease_expires_at == job.started_at + timedelta(seconds=120)


@pytest.mark.asyncio
async def test_each_claim_gets_a_unique_lease_token(db_session, task_id):

    for _ in range(5):
        await create_research_job(db=db_session, task_id=task_id)

    tokens = [
        (await claim_next_job(db=db_session, worker_id="worker-1")).lease_token
        for _ in range(5)
    ]

    assert len(set(tokens)) == 5


@pytest.mark.asyncio
async def test_retry_gets_a_new_lease_token(db_session, task_id):

    job = await create_research_job(db=db_session, task_id=task_id)

    await mark_job_running(db=db_session, job=job, worker_id="worker-1")
    first_token = job.lease_token

    await mark_job_failed(db=db_session, job=job, error="boom")
    await mark_job_running(db=db_session, job=job, worker_id="worker-2")

    # Same job, new claim: the old worker's token no longer matches.
    assert job.lease_token != first_token
    assert job.attempts == 2


@pytest.mark.asyncio
async def test_requeue_clears_the_lease(db_session, task_id):

    from app.jobs.service import requeue_job

    job = await create_research_job(db=db_session, task_id=task_id)

    await mark_job_running(db=db_session, job=job, worker_id="worker-1")
    await mark_job_failed(db=db_session, job=job, error="boom")
    await requeue_job(db=db_session, job=job)

    assert job.status == JobStatus.PENDING
    assert job.worker_id is None
    assert job.lease_token is None
    assert job.lease_expires_at is None


@pytest.mark.asyncio
async def test_new_job_has_no_lease(db_session, task_id):

    job = await create_research_job(db=db_session, task_id=task_id)

    assert job.lease_token is None
    assert job.lease_expires_at is None
