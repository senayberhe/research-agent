"""Job throughput and latency over time, in fixed buckets, for charts.

Jobs are placed by when they finished (completed_at), as in the SLOs;
latency is each job's start-to-completion time. Every bucket in the range
is returned, empty ones included, so a chart's x-axis has no gaps.
"""

from datetime import datetime, timedelta

from sqlalchemy import func, literal, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.jobs.models import JobStatus, ResearchJob


# The bucket size each range is shown at: about 24-30 points per chart.
RANGES = {
    "24h": (timedelta(hours=24), timedelta(hours=1)),
    "7d": (timedelta(days=7), timedelta(hours=6)),
    "30d": (timedelta(days=30), timedelta(days=1)),
}


def _bucket_start(moment: datetime, size: timedelta) -> datetime:
    """The start of the bucket `moment` falls in (buckets are aligned to
    the Unix epoch, as Postgres' date_bin below)."""

    epoch = datetime(1970, 1, 1)

    return epoch + ((moment - epoch) // size) * size


async def build_job_timeseries(
    db: AsyncSession,
    range_name: str,
    now: datetime,
) -> dict:
    """{range, bucket_seconds, start, end, buckets: [{start, completed,
    failed, p95_latency_seconds}]} for the last `range_name`."""

    span, size = RANGES[range_name]

    # Whole buckets, so the first one isn't partial.
    first = _bucket_start(now - span, size) + size
    last = _bucket_start(now, size)

    bucket = func.date_bin(
        literal(size),
        ResearchJob.completed_at,
        literal(datetime(1970, 1, 1)),
    ).label("bucket")

    seconds = func.extract(
        "epoch",
        ResearchJob.completed_at - ResearchJob.started_at,
    )

    rows = (
        await db.execute(
            select(
                bucket,
                func.count().filter(ResearchJob.status == JobStatus.COMPLETED),
                func.count().filter(ResearchJob.status == JobStatus.FAILED),
                func.percentile_cont(0.95).within_group(seconds.asc()),
            )
            .where(
                ResearchJob.status.in_([JobStatus.COMPLETED, JobStatus.FAILED]),
                ResearchJob.completed_at >= first,
                ResearchJob.completed_at < last + size,
            )
            .group_by(bucket)
        )
    ).all()

    found = {
        start: (completed, failed, p95)
        for start, completed, failed, p95 in rows
    }

    buckets = []
    start = first

    while start <= last:
        completed, failed, p95 = found.get(start, (0, 0, None))

        buckets.append(
            {
                "start": start,
                "completed": completed,
                "failed": failed,
                # None: no job finished in the bucket (not 0 seconds).
                "p95_latency_seconds": (
                    round(float(p95), 3) if p95 is not None else None
                ),
            }
        )

        start += size

    return {
        "range": range_name,
        "bucket_seconds": int(size.total_seconds()),
        "start": first,
        "end": last + size,
        "buckets": buckets,
    }
