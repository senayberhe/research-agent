"""System health checks: database, workers (running jobs' leases) and the
job queue."""

from datetime import timedelta

import pytest

from app.db.models import JobStatus, ResearchJob, utc_now
from app.schemas.research import SystemHealthResponse
from app.services.health_service import (
    build_system_health,
    check_database,
    check_jobs,
    check_workers,
)
from app.tests.test_workflow_service import create_task


async def add_job(db, status, worker_id=None, lease_seconds=None):
    """A job; lease_seconds from now (negative: already expired)."""

    task = await create_task(db)

    job = ResearchJob(
        task_id=task.id,
        status=status,
        attempts=1 if status != JobStatus.PENDING else 0,
        worker_id=worker_id,
        lease_expires_at=(
            utc_now() + timedelta(seconds=lease_seconds)
            if lease_seconds is not None
            else None
        ),
    )

    db.add(job)
    await db.commit()

    return job


@pytest.mark.asyncio
async def test_healthy_when_idle(db_session):

    health = await build_system_health(db_session)

    assert health == {
        "status": "healthy",
        "database": {
            "status": "healthy",
            "message": "Database connection is healthy.",
        },
        "workers": {
            "status": "healthy",
            "message": "No running jobs; worker queue is idle.",
        },
        "jobs": {
            "status": "healthy",
            "message": "Job system is healthy.",
        },
    }

    SystemHealthResponse(**health)


@pytest.mark.asyncio
async def test_healthy_with_live_workers(db_session):

    await add_job(db_session, JobStatus.RUNNING, "worker-1", lease_seconds=300)
    await add_job(db_session, JobStatus.RUNNING, "worker-2", lease_seconds=300)
    # Pending and failed jobs alone don't make the system unhealthy.
    await add_job(db_session, JobStatus.PENDING)
    await add_job(db_session, JobStatus.FAILED)

    health = await build_system_health(db_session)

    assert health["status"] == "healthy"
    assert health["workers"]["message"] == (
        "2 active worker lease(s) are healthy."
    )

    jobs = await check_jobs(db_session)

    assert jobs["pending"] == 1
    assert jobs["running"] == 2
    assert jobs["failed"] == 1
    assert jobs["stale"] == 0


@pytest.mark.asyncio
async def test_degraded_when_a_lease_has_expired(db_session):

    await add_job(db_session, JobStatus.RUNNING, "worker-1", lease_seconds=300)
    await add_job(db_session, JobStatus.RUNNING, "worker-dead", lease_seconds=-1)

    health = await build_system_health(db_session)

    assert health["status"] == "degraded"
    assert health["workers"] == {
        "status": "degraded",
        "message": "1 running job(s) have unhealthy worker leases.",
    }
    assert health["jobs"] == {
        "status": "degraded",
        "message": "1 stale running job(s) detected.",
    }


@pytest.mark.asyncio
async def test_running_job_without_lease_is_stale(db_session):

    # Nothing will ever renew it (as recover_stale_jobs sees it): both
    # checks agree it's a problem.
    await add_job(db_session, JobStatus.RUNNING, "worker-1", lease_seconds=None)

    workers = await check_workers(db_session)
    jobs = await check_jobs(db_session)

    assert workers["status"] == "degraded"
    assert jobs["status"] == "degraded"
    assert jobs["stale"] == 1


class BrokenSession:
    """A session whose database is unreachable."""

    async def execute(self, *args, **kwargs):
        raise ConnectionRefusedError(
            "connect to postgres:secret@db.internal:5432 refused"
        )


@pytest.mark.asyncio
async def test_unhealthy_when_database_is_down(caplog):

    health = await build_system_health(BrokenSession())

    assert health["status"] == "unhealthy"

    for component in ("database", "workers", "jobs"):
        assert health[component]["status"] == "unhealthy"

    assert health["database"]["message"] == (
        "Database check failed: ConnectionRefusedError"
    )

    # The details are logged, not returned.
    assert "db.internal" not in str(health)
    assert "db.internal" in caplog.text

    SystemHealthResponse(**health)


@pytest.mark.asyncio
async def test_check_database_healthy(db_session):

    assert (await check_database(db_session))["status"] == "healthy"
