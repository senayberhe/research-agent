"""System health: is the database reachable, are running jobs held by live
workers, is the job queue moving. Each check returns a status (healthy,
degraded or unhealthy) and a message; build_system_health combines them."""

import logging

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import JobStatus, ResearchJob, utc_now


logger = logging.getLogger(__name__)


def _lease_expired(now):
    """A RUNNING job's lease has run out, or it never had one: either way
    no live worker is renewing it (as in recover_stale_jobs)."""

    return or_(
        ResearchJob.lease_expires_at.is_(None),
        ResearchJob.lease_expires_at < now,
    )


def _failed_check(name: str, exc: Exception) -> dict:
    """The details go to the log, not the response: a health endpoint is
    usually public, and database errors can name hosts and databases."""

    logger.exception("%s check failed", name)

    return {
        "status": "unhealthy",
        "message": (
            f"{name} check failed: {type(exc).__name__}"
        ),
    }


async def check_database(
    db: AsyncSession,
) -> dict:
    """
    Verify that PostgreSQL is reachable.
    """

    try:
        await db.execute(
            select(func.count()).select_from(
                ResearchJob
            )
        )

        return {
            "status": "healthy",
            "message": "Database connection is healthy.",
        }

    except Exception as exc:
        return _failed_check("Database", exc)


async def check_jobs(
    db: AsyncSession,
) -> dict:
    """
    Check the current state of the job queue.
    """

    try:
        result = await db.execute(
            select(
                ResearchJob.status,
                func.count(ResearchJob.id),
            )
            .group_by(ResearchJob.status)
        )

        counts = {
            status: count
            for status, count in result.all()
        }

        pending = counts.get(
            JobStatus.PENDING,
            0,
        )

        running = counts.get(
            JobStatus.RUNNING,
            0,
        )

        failed = counts.get(
            JobStatus.FAILED,
            0,
        )

        # A failed job does not automatically mean
        # the whole system is unhealthy.
        #
        # We only flag the queue when there are
        # stale running jobs.

        now = utc_now()

        stale_result = await db.execute(
            select(func.count(ResearchJob.id))
            .where(
                ResearchJob.status == JobStatus.RUNNING,
                _lease_expired(now),
            )
        )

        stale_jobs = stale_result.scalar_one()

        if stale_jobs > 0:
            return {
                "status": "degraded",
                "message": (
                    f"{stale_jobs} stale running "
                    "job(s) detected."
                ),
                "pending": pending,
                "running": running,
                "failed": failed,
                "stale": stale_jobs,
            }

        return {
            "status": "healthy",
            "message": "Job system is healthy.",
            "pending": pending,
            "running": running,
            "failed": failed,
            "stale": stale_jobs,
        }

    except Exception as exc:
        return _failed_check("Job", exc)


async def check_workers(
    db: AsyncSession,
) -> dict:
    """
    Check whether active jobs have valid worker leases.
    """

    try:
        now = utc_now()

        result = await db.execute(
            select(ResearchJob)
            .where(
                ResearchJob.status
                == JobStatus.RUNNING
            )
        )

        running_jobs = list(
            result.scalars().all()
        )

        if not running_jobs:
            return {
                "status": "healthy",
                "message": (
                    "No running jobs; worker queue "
                    "is idle."
                ),
            }

        healthy_workers = 0
        unhealthy_workers = 0

        for job in running_jobs:
            if (
                job.worker_id
                and job.lease_expires_at
                and job.lease_expires_at > now
            ):
                healthy_workers += 1
            else:
                unhealthy_workers += 1

        if unhealthy_workers:
            return {
                "status": "degraded",
                "message": (
                    f"{unhealthy_workers} running "
                    "job(s) have unhealthy worker "
                    "leases."
                ),
            }

        return {
            "status": "healthy",
            "message": (
                f"{healthy_workers} active worker "
                "lease(s) are healthy."
            ),
        }

    except Exception as exc:
        return _failed_check("Worker", exc)


async def build_system_health(
    db: AsyncSession,
) -> dict:
    """
    Build the complete system health response.
    """

    database = await check_database(db)
    workers = await check_workers(db)
    jobs = await check_jobs(db)

    component_statuses = [
        database["status"],
        workers["status"],
        jobs["status"],
    ]

    if "unhealthy" in component_statuses:
        overall_status = "unhealthy"

    elif "degraded" in component_statuses:
        overall_status = "degraded"

    else:
        overall_status = "healthy"

    return {
        "status": overall_status,
        "database": {
            "status": database["status"],
            "message": database["message"],
        },
        "workers": {
            "status": workers["status"],
            "message": workers["message"],
        },
        "jobs": {
            "status": jobs["status"],
            "message": jobs["message"],
        },
    }
