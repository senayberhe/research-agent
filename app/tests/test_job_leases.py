from datetime import timedelta

import pytest
import pytest_asyncio

from app.agents.research_agent import AgentCheckpoint
from app.core.config import settings
from app.db.models import (
    AgentRun,
    AgentRunStatus,
    JobStatus,
    ResearchJob,
    ResearchTask,
    TaskStatus,
    utc_now,
)
from app.jobs.service import (
    MAX_ATTEMPTS_ERROR,
    STALE_LEASE_ERROR,
    claim_next_job,
    create_research_job,
    mark_job_completed,
    mark_job_failed,
    recover_stale_jobs,
)
from app.tests.test_workflow_service import create_task


@pytest_asyncio.fixture
async def db(db_session):
    return db_session


# Jobs have a foreign key to research_tasks, so they need a real task (the
# test database has no task 1).
@pytest_asyncio.fixture
async def task_id(db) -> int:
    task = await create_task(db)
    return task.id


async def expire_lease(db, job: ResearchJob) -> None:
    """As if the worker died: its lease ran out a second ago."""
    job.lease_expires_at = utc_now() - timedelta(seconds=1)
    await db.commit()


# -------------------------
# Claiming
# -------------------------


@pytest.mark.asyncio
async def test_claim_creates_lease(db, task_id):

    job = await create_research_job(
        db=db,
        task_id=task_id,
    )

    claimed = await claim_next_job(
        db=db,
        worker_id="worker-1",
    )

    assert claimed is not None
    assert claimed.status == JobStatus.RUNNING
    assert claimed.worker_id == "worker-1"
    assert claimed.lease_token is not None
    assert claimed.lease_expires_at is not None


# -------------------------
# Stale job recovered
# -------------------------


@pytest.mark.asyncio
async def test_stale_job_is_returned_to_pending(db, task_id):

    job = await create_research_job(db=db, task_id=task_id)

    claimed = await claim_next_job(db=db, worker_id="worker-1")
    old_token = claimed.lease_token

    await expire_lease(db, claimed)

    assert await recover_stale_jobs(db) == 1

    refreshed = await db.get(ResearchJob, job.id)

    # Back in the queue, owned by nobody, saying why.
    assert refreshed.status == JobStatus.PENDING
    assert refreshed.worker_id is None
    assert refreshed.lease_token is None
    assert refreshed.lease_expires_at is None
    assert refreshed.error == STALE_LEASE_ERROR

    # Another worker can take it, with a lease of its own.
    reclaimed = await claim_next_job(db=db, worker_id="worker-2")

    assert reclaimed.id == job.id
    assert reclaimed.worker_id == "worker-2"
    assert reclaimed.lease_token != old_token
    assert reclaimed.attempts == 2


@pytest.mark.asyncio
async def test_job_with_live_lease_is_not_recovered(db, task_id):

    await create_research_job(db=db, task_id=task_id)

    claimed = await claim_next_job(db=db, worker_id="worker-1")

    # Its worker is still within the lease: leave it alone.
    assert await recover_stale_jobs(db) == 0

    assert claimed.status == JobStatus.RUNNING
    assert claimed.worker_id == "worker-1"


@pytest.mark.asyncio
async def test_finished_and_pending_jobs_are_not_recovered(db, task_id):

    await create_research_job(db=db, task_id=task_id)

    db.add_all([
        ResearchJob(task_id=task_id, status=JobStatus.COMPLETED),
        ResearchJob(task_id=task_id, status=JobStatus.FAILED),
    ])
    await db.commit()

    assert await recover_stale_jobs(db) == 0


# -------------------------
# Maximum attempts
# -------------------------


@pytest.mark.asyncio
async def test_stale_job_fails_after_max_attempts(db, task_id):

    job = await create_research_job(
        db=db,
        task_id=task_id,
    )

    claimed = await claim_next_job(
        db=db,
        worker_id="worker-1",
    )

    claimed.attempts = settings.worker_max_attempts

    claimed.lease_expires_at = (
        utc_now() - timedelta(seconds=1)
    )

    await db.commit()

    recovered = await recover_stale_jobs(db)

    assert recovered == 1

    refreshed = await db.get(
        ResearchJob,
        job.id,
    )

    assert refreshed.status == JobStatus.FAILED
    assert refreshed.error is not None
    assert refreshed.error == MAX_ATTEMPTS_ERROR
    assert refreshed.completed_at is not None
    assert refreshed.lease_token is None


# -------------------------
# Wrong worker cannot complete
# -------------------------


@pytest.mark.asyncio
async def test_wrong_lease_cannot_complete_job(db, task_id):

    job = await create_research_job(
        db=db,
        task_id=task_id,
    )

    claimed = await claim_next_job(
        db=db,
        worker_id="worker-1",
    )

    result = await mark_job_completed(
        db=db,
        job=claimed,
        lease_token="wrong-token",
    )

    assert result is False

    refreshed = await db.get(
        ResearchJob,
        job.id,
    )

    assert refreshed.status == JobStatus.RUNNING


@pytest.mark.asyncio
async def test_right_lease_completes_job(db, task_id):

    await create_research_job(db=db, task_id=task_id)

    claimed = await claim_next_job(db=db, worker_id="worker-1")

    assert await mark_job_completed(
        db=db,
        job=claimed,
        lease_token=claimed.lease_token,
    ) is True

    # Done: completed, and no lease left.
    assert claimed.status == JobStatus.COMPLETED
    assert claimed.completed_at is not None
    assert claimed.lease_token is None
    assert claimed.lease_expires_at is None


@pytest.mark.asyncio
async def test_wrong_lease_cannot_fail_job(db, task_id):

    await create_research_job(db=db, task_id=task_id)

    claimed = await claim_next_job(db=db, worker_id="worker-1")

    assert await mark_job_failed(
        db=db,
        job=claimed,
        error="late failure",
        lease_token="wrong-token",
    ) is False

    assert claimed.status == JobStatus.RUNNING
    assert claimed.error is None


@pytest.mark.asyncio
async def test_worker_that_lost_its_lease_cannot_overwrite(db, task_id):

    await create_research_job(db=db, task_id=task_id)

    # Worker 1 claims the job, stalls past its lease, and the job is
    # recovered and claimed by worker 2.
    first = await claim_next_job(db=db, worker_id="worker-1")
    worker_1_token = first.lease_token

    await expire_lease(db, first)
    await recover_stale_jobs(db)
    await db.commit()

    second = await claim_next_job(db=db, worker_id="worker-2")

    # Worker 1 wakes up and tries to record its result: refused.
    assert await mark_job_completed(
        db=db,
        job=first,
        lease_token=worker_1_token,
    ) is False

    assert await mark_job_failed(
        db=db,
        job=first,
        error="worker 1 failed",
        lease_token=worker_1_token,
    ) is False

    # Still worker 2's job, still running.
    assert second.status == JobStatus.RUNNING
    assert second.worker_id == "worker-2"
    assert second.error is None


# -------------------------
# The dead worker's run
# -------------------------


@pytest.mark.asyncio
async def test_stale_recovery_fails_dead_workers_run_so_task_can_resume(
    db,
    task_id,
):

    await create_research_job(db=db, task_id=task_id)

    claimed = await claim_next_job(db=db, worker_id="worker-1")

    # The dead worker was mid-run, with a checkpoint after iteration 2.
    task = await db.get(ResearchTask, task_id)
    task.status = TaskStatus.RESEARCHING

    checkpoint = AgentCheckpoint(
        iteration=2,
        input_items=[{"role": "user", "content": "What is RAG?"}],
        tool_call_count=2,
        input_tokens=200,
        output_tokens=100,
        total_tokens=300,
        estimated_cost=None,
        executed_tool_calls=2,
    ).to_dict()

    run = AgentRun(
        task_id=task_id,
        status=AgentRunStatus.WAITING_FOR_TOOL,
        state=checkpoint,
    )
    db.add(run)

    await expire_lease(db, claimed)

    await recover_stale_jobs(db)

    # The run is failed with its checkpoint totals (not 0) and kept
    # checkpoint, and the task is failed: the next worker to claim the job
    # resumes it from iteration 2.
    assert run.status == AgentRunStatus.FAILED
    assert run.error == STALE_LEASE_ERROR
    assert run.iteration_count == 2
    assert run.total_tokens == 300
    assert run.state == checkpoint

    assert task.status == TaskStatus.FAILED


# -------------------------
# Completing and recovering at the same moment
# -------------------------


@pytest.mark.asyncio
async def test_recovery_skips_job_a_worker_is_completing(clean_tables):

    from app.jobs.service import _lock_if_still_ours
    from app.tests.conftest import TestSessionLocal

    async with TestSessionLocal() as setup:

        task = ResearchTask(question="What is RAG?", status=TaskStatus.PENDING)
        setup.add(task)
        await setup.flush()

        await create_research_job(db=setup, task_id=task.id)

        job = await claim_next_job(db=setup, worker_id="worker-1")
        token = job.lease_token

        # Expired, but its worker is just now finishing.
        await expire_lease(setup, job)
        job_id = job.id

    async with TestSessionLocal() as worker, TestSessionLocal() as recoverer:

        # The worker has the job locked, mid-completion.
        worker_job = await worker.get(ResearchJob, job_id)
        assert await _lock_if_still_ours(worker, worker_job, token) is not None

        # Recovery in another session skips the locked row instead of
        # requeueing a job that is being completed.
        assert await recover_stale_jobs(recoverer) == 0
        await recoverer.rollback()

        # The worker's completion goes through.
        assert await mark_job_completed(
            db=worker,
            job=worker_job,
            lease_token=token,
        ) is True
        await worker.commit()

    async with TestSessionLocal() as check:
        assert (await check.get(ResearchJob, job_id)).status == JobStatus.COMPLETED


@pytest.mark.asyncio
async def test_running_job_without_lease_is_recovered(db, task_id):

    # Claimed before leases existed: RUNNING, but no lease to expire.
    db.add(
        ResearchJob(
            task_id=task_id,
            status=JobStatus.RUNNING,
            attempts=1,
            worker_id="old-worker",
        )
    )
    await db.commit()

    assert await recover_stale_jobs(db) == 1
