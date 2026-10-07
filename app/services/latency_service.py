"""Latency percentiles (p50, p95, p99), in seconds, over the SLO's rolling
window [window_start, window_end]:

- Jobs: finished (completed or failed) inside the window, by completed_at;
  each job's duration from its (last) start to its completion. A job's
  started_at is reset when it's retried, so a retried job counts its last
  attempt.
- Tools: finished tool calls (tool_completed and tool_failed events)
  recorded inside the window, per tool. Failed calls count too: timeouts
  are usually the slowest calls of all.

Percentiles interpolate linearly (calculate_percentile), the same as
Postgres' percentile_cont.
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


FINISHED_TOOL_EVENTS = (
    JobEventType.TOOL_COMPLETED.value,
    JobEventType.TOOL_FAILED.value,
)


def calculate_percentile(
    values: list[float],
    percentile: float,
) -> float | None:
    """
    Calculate a percentile using linear interpolation.
    """

    if not values:
        return None

    if not 0 <= percentile <= 1:
        raise ValueError(
            "percentile must be between 0 and 1"
        )

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    position = (
        percentile
        * (len(values) - 1)
    )

    lower_index = int(position)
    upper_index = min(
        lower_index + 1,
        len(values) - 1,
    )

    fraction = (
        position - lower_index
    )

    lower_value = values[lower_index]
    upper_value = values[upper_index]

    return (
        lower_value
        + fraction
        * (upper_value - lower_value)
    )


def percentiles(durations: list[float]) -> dict:
    """{"p50", "p95", "p99"} of the durations (None each if empty)."""

    return {
        "p50": calculate_percentile(
            durations,
            0.50,
        ),
        "p95": calculate_percentile(
            durations,
            0.95,
        ),
        "p99": calculate_percentile(
            durations,
            0.99,
        ),
    }


def calculate_histogram_quantile(
    buckets: list[dict],
    quantile: float,
) -> float | None:
    """Estimates a quantile from cumulative histogram buckets
    ([{"le": upper bound, "count": observations <= le}], sorted by le),
    interpolating linearly inside the bucket it falls in, as Prometheus'
    histogram_quantile does. None if there are no observations.

    For bucket data such as GET /metrics' histograms; the SLO report uses
    the exact percentiles below (calculate_percentile) instead.
    """

    if not buckets:
        return None

    total = buckets[-1]["count"]

    if total <= 0:
        return None

    target = total * quantile

    previous_count = 0.0
    previous_bound = 0.0

    for bucket in buckets:
        current_count = bucket["count"]

        if current_count >= target:
            bucket_count = current_count - previous_count

            if bucket_count <= 0:
                return bucket["le"]

            position = (
                target - previous_count
            ) / bucket_count

            return (
                previous_bound
                + position
                * (bucket["le"] - previous_bound)
            )

        previous_count = current_count
        previous_bound = bucket["le"]

    return buckets[-1]["le"]


async def get_job_latency_percentiles_in_window(
    db: AsyncSession,
    window_start: datetime,
    window_end: datetime,
) -> dict:
    """
    Calculate job latency percentiles using jobs
    completed inside the SLO window.
    """

    result = await db.execute(
        select(ResearchJob)
        .where(
            ResearchJob.status.in_(
                [
                    JobStatus.COMPLETED,
                    JobStatus.FAILED,
                ]
            ),
            ResearchJob.started_at.is_not(None),
            ResearchJob.completed_at.is_not(None),
            ResearchJob.completed_at >= window_start,
            ResearchJob.completed_at <= window_end,
        )
        .order_by(ResearchJob.completed_at.asc())
    )

    jobs = list(result.scalars().all())

    durations = []

    for job in jobs:
        if (
            job.started_at is None
            or job.completed_at is None
        ):
            continue

        duration = (
            job.completed_at
            - job.started_at
        ).total_seconds()

        if duration >= 0:
            durations.append(duration)

    return percentiles(durations)


async def get_tool_latency_percentiles_in_window(
    db: AsyncSession,
    window_start: datetime,
    window_end: datetime,
) -> dict:
    """
    Calculate tool latency percentiles, per tool, from
    finished tool calls (tool_completed and tool_failed
    events) inside the SLO window.
    """

    result = await db.execute(
        select(JobEvent)
        .where(
            JobEvent.event_type.in_(FINISHED_TOOL_EVENTS),
            JobEvent.created_at >= window_start,
            JobEvent.created_at <= window_end,
        )
        .order_by(JobEvent.created_at.asc())
    )

    events = list(result.scalars().all())

    tool_durations: dict[str, list[float]] = {}

    for event in events:

        # {} if there's no metadata or it isn't valid JSON.
        metadata = parse_event_metadata(event)

        tool = metadata.get("tool")
        duration_ms = metadata.get(
            "duration_ms"
        )

        if (
            not tool
            or duration_ms is None
        ):
            continue

        duration_seconds = (
            float(duration_ms) / 1000
        )

        tool_durations.setdefault(
            tool,
            [],
        ).append(duration_seconds)

    return {
        tool: percentiles(durations)
        for tool, durations in sorted(tool_durations.items())
    }
