import asyncio

import pytest
import pytest_asyncio
from sqlalchemy import select

from app.db.models import (
    JobStatus,
    ResearchJob,
    ResearchTask,
    TaskStatus,
)
from app.jobs.service import claim_next_job
from app.tests.conftest import TestSessionLocal


@pytest_asyncio.fixture
async def db(db_session):
    # These tests use one session, so the rolled-back db_session works.
    return db_session


@pytest.mark.asyncio
async def test_claim_next_job_marks_job_running(db):

    task = ResearchTask(
        question="What is RAG?",
        status=TaskStatus.PENDING,
    )

    db.add(task)

    await db.flush()

    job = ResearchJob(
        task_id=task.id,
        status=JobStatus.PENDING,
    )

    db.add(job)

    await db.commit()

    claimed = await claim_next_job(
        db=db,
        worker_id="worker-1",
    )

    assert claimed is not None
    assert claimed.id == job.id
    assert claimed.status == JobStatus.RUNNING
    assert claimed.worker_id == "worker-1"
    assert claimed.attempts == 1


@pytest.mark.asyncio
async def test_completed_job_is_not_claimed(db):

    task = ResearchTask(
        question="What is RAG?",
        status=TaskStatus.PENDING,
    )

    db.add(task)

    await db.flush()

    job = ResearchJob(
        task_id=task.id,
        status=JobStatus.COMPLETED,
    )

    db.add(job)

    await db.commit()

    claimed = await claim_next_job(
        db=db,
        worker_id="worker-1",
    )

    assert claimed is None


@pytest.mark.asyncio
async def test_failed_job_is_not_claimed(db):

    task = ResearchTask(
        question="What is RAG?",
        status=TaskStatus.PENDING,
    )

    db.add(task)

    await db.flush()

    job = ResearchJob(
        task_id=task.id,
        status=JobStatus.FAILED,
        error="Previous execution failed.",
    )

    db.add(job)

    await db.commit()

    claimed = await claim_next_job(
        db=db,
        worker_id="worker-1",
    )

    assert claimed is None


# -------------------------
# Real concurrency
# -------------------------


@pytest.mark.asyncio
async def test_concurrent_workers_claim_each_job_exactly_once(clean_tables):

    jobs_count = 5
    workers_count = 10

    async with TestSessionLocal() as setup:

        task = ResearchTask(question="What is RAG?", status=TaskStatus.PENDING)
        setup.add(task)
        await setup.flush()

        setup.add_all(
            [ResearchJob(task_id=task.id) for _ in range(jobs_count)]
        )
        await setup.commit()

    # Every worker waits here until all of them are ready, then they claim
    # at the same moment, each on its own database connection.
    start = asyncio.Event()
    ready = 0

    async def worker(n: int) -> int | None:

        nonlocal ready

        async with TestSessionLocal() as db:

            # Open the connection first, so all claims hit the database
            # together instead of one after another as connections open.
            await db.execute(select(1))

            ready += 1
            if ready == workers_count:
                start.set()

            await start.wait()

            job = await claim_next_job(db=db, worker_id=f"worker-{n}")

            return job.id if job is not None else None

    claimed = await asyncio.wait_for(
        asyncio.gather(*(worker(n) for n in range(workers_count))),
        timeout=10,
    )

    claimed_ids = [job_id for job_id in claimed if job_id is not None]

    # Every job went to exactly one worker; the other workers got nothing
    # (they didn't wait on locks or take a job twice).
    assert len(claimed_ids) == jobs_count
    assert len(set(claimed_ids)) == jobs_count
    assert claimed.count(None) == workers_count - jobs_count

    async with TestSessionLocal() as db:

        jobs = (await db.execute(select(ResearchJob))).scalars().all()

        assert all(job.status == JobStatus.RUNNING for job in jobs)
        assert all(job.attempts == 1 for job in jobs)

        # Each job is recorded against the one worker that got it, with a
        # lease token of its own.
        assert len({job.worker_id for job in jobs}) == jobs_count
        assert len({job.lease_token for job in jobs}) == jobs_count
        assert None not in {job.lease_token for job in jobs}
