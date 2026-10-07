"""Success rates over rows already loaded for a window: the same rules as
sli_service's SQL counts (which the SLO report and error budgets use),
for code that has the jobs or events in hand.

- Job success: completed / (completed + failed); jobs still pending or
  running haven't succeeded or failed yet, so they don't count.
- Tool success: finished tool calls (tool_completed / tool_failed events)
  that succeeded, out of all of them. A call failed if its metadata says
  success: false; without that key, tool_failed means failed.

An empty window has nothing failing: 1.0.

Latency percentiles here (calculate_percentile) interpolate linearly, like
Postgres' percentile_cont, which latency_service uses for the SLO report:
the same values either way.
"""

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    JobEvent,
    JobEventType,
    JobStatus,
    ResearchJob,
)
from app.jobs.event_service import parse_event_metadata
from app.services.latency_service import calculate_percentile


FINISHED_TOOL_EVENTS = (
    JobEventType.TOOL_COMPLETED.value,
    JobEventType.TOOL_FAILED.value,
)


def calculate_job_success_rate(jobs: list[ResearchJob]) -> float:

    statuses = [JobStatus(job.status) for job in jobs]

    completed = statuses.count(JobStatus.COMPLETED)
    finished = completed + statuses.count(JobStatus.FAILED)

    return completed / finished if finished else 1.0


def _tool_call_succeeded(event: JobEvent) -> bool:

    return parse_event_metadata(event).get(
        "success",
        event.event_type == JobEventType.TOOL_COMPLETED.value,
    ) is True


def calculate_tool_success_rate(events: list[JobEvent]) -> float:

    calls = [
        event
        for event in events
        if event.event_type in FINISHED_TOOL_EVENTS
    ]

    if not calls:
        return 1.0

    succeeded = sum(_tool_call_succeeded(event) for event in calls)

    return succeeded / len(calls)


# -------------------------------------------------------------------
# Loading a window
# -------------------------------------------------------------------


async def get_jobs_in_window(
    db: AsyncSession,
    start: datetime,
    end: datetime,
) -> list[ResearchJob]:
    """Jobs that finished (completed or failed) between start and end, by
    when they finished."""

    result = await db.execute(
        select(ResearchJob)
        .where(
            ResearchJob.status.in_([JobStatus.COMPLETED, JobStatus.FAILED]),
            ResearchJob.completed_at >= start,
            ResearchJob.completed_at <= end,
        )
        .order_by(ResearchJob.completed_at, ResearchJob.id)
    )

    return list(result.scalars().all())


async def get_tool_events_in_window(
    db: AsyncSession,
    start: datetime,
    end: datetime,
) -> list[JobEvent]:
    """Finished tool calls (tool_completed / tool_failed events) recorded
    between start and end."""

    result = await db.execute(
        select(JobEvent)
        .where(
            JobEvent.event_type.in_(FINISHED_TOOL_EVENTS),
            JobEvent.created_at >= start,
            JobEvent.created_at <= end,
        )
        .order_by(JobEvent.created_at, JobEvent.id)
    )

    return list(result.scalars().all())


# -------------------------------------------------------------------
# Percentiles
# -------------------------------------------------------------------


def _percentiles(values: list[float]) -> dict:
    """{"p50", "p95", "p99"}, like latency_service's."""

    return {
        "p50": calculate_percentile(values, 0.50),
        "p95": calculate_percentile(values, 0.95),
        "p99": calculate_percentile(values, 0.99),
    }


def calculate_job_latency_percentiles(jobs: list[ResearchJob]) -> dict:
    """Seconds from each finished job's last start to its completion. (Only
    the last attempt: a job's start time is reset when it's retried.)"""

    return _percentiles(
        [
            (job.completed_at - job.started_at).total_seconds()
            for job in jobs
            if job.started_at is not None and job.completed_at is not None
        ]
    )


def calculate_tool_latency_percentiles(events: list[JobEvent]) -> dict:
    """Seconds per finished tool call, from the events' duration_ms."""

    durations = []

    for event in events:
        if event.event_type not in FINISHED_TOOL_EVENTS:
            continue

        duration_ms = parse_event_metadata(event).get("duration_ms")

        if duration_ms is not None:
            durations.append(float(duration_ms) / 1000)

    return _percentiles(durations)

