"""Service level indicators measured over a time window [start, end]: the
good and bad events the SLO report and the error budgets are built from,
so both count the same events.

- Job success: jobs that finished (completed or failed) inside the window,
  by when they finished (completed_at), not when they were created. A job
  still pending or running hasn't succeeded or failed yet.
- Tool success: finished tool calls (tool_completed / tool_failed
  JobEvents) recorded inside the window.

- Latency (for the error budgets): finished jobs, and finished tool calls
  with a duration, inside the window, and how many were slower than the
  SLO's threshold: the same jobs and calls latency_service's percentiles
  are taken over.
"""

from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.slo import JOB_LATENCY, TOOL_LATENCY
from app.jobs.models import JobStatus, ResearchJob
from app.services.system_metrics_service import finished_tool_calls


async def job_success_counts(
    db: AsyncSession,
    start: datetime,
    end: datetime,
) -> tuple[int, int]:
    """(finished jobs, failed jobs) that finished between start and end."""

    finished, failed = (
        await db.execute(
            select(
                func.count(),
                func.count().filter(ResearchJob.status == JobStatus.FAILED),
            ).where(
                ResearchJob.status.in_([JobStatus.COMPLETED, JobStatus.FAILED]),
                ResearchJob.completed_at >= start,
                ResearchJob.completed_at <= end,
            )
        )
    ).one()

    return finished, failed


async def tool_success_counts(
    db: AsyncSession,
    start: datetime,
    end: datetime,
) -> tuple[int, int]:
    """(finished tool calls, failed ones) recorded between start and end."""

    calls = finished_tool_calls(start, until=end)

    total, failed = (
        await db.execute(
            select(
                func.count(),
                func.count().filter(calls.c.success.is_not(True)),
            )
        )
    ).one()

    return total, failed


async def job_latency_counts(
    db: AsyncSession,
    start: datetime,
    end: datetime,
) -> tuple[int, int]:
    """(timed jobs that finished between start and end, those that took
    longer than the job latency target)."""

    seconds = func.extract(
        "epoch",
        ResearchJob.completed_at - ResearchJob.started_at,
    )

    total, slow = (
        await db.execute(
            select(
                func.count(),
                func.count().filter(seconds > JOB_LATENCY.target),
            ).where(
                ResearchJob.status.in_([JobStatus.COMPLETED, JobStatus.FAILED]),
                ResearchJob.started_at.is_not(None),
                ResearchJob.completed_at >= start,
                ResearchJob.completed_at <= end,
                seconds >= 0,
            )
        )
    ).one()

    return total, slow


async def tool_latency_counts(
    db: AsyncSession,
    start: datetime,
    end: datetime,
) -> tuple[int, int]:
    """(timed tool calls recorded between start and end, those slower than
    the tool latency target)."""

    calls = finished_tool_calls(start, until=end)

    total, slow = (
        await db.execute(
            select(
                func.count(),
                func.count().filter(
                    calls.c.duration_ms > TOOL_LATENCY.target * 1000
                ),
            ).where(
                calls.c.tool.is_not(None),
                calls.c.duration_ms.is_not(None),
            )
        )
    ).one()

    return total, slow

