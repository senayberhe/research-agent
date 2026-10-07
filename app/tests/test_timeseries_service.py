"""Job throughput and latency per time bucket (for the dashboard charts)."""

from datetime import datetime, timedelta

import pytest

from app.db.models import JobStatus, ResearchJob
from app.services.timeseries_service import build_job_timeseries
from app.tests.test_research_api import client
from app.tests.test_workflow_service import create_task


NOW = datetime(2026, 10, 7, 12, 30, 0)


async def add_job(db, status, finished_at, seconds=10.0):
    task = await create_task(db)

    db.add(
        ResearchJob(
            task_id=task.id,
            status=status,
            attempts=1,
            started_at=finished_at - timedelta(seconds=seconds),
            completed_at=finished_at,
        )
    )
    await db.flush()


@pytest.mark.asyncio
async def test_hourly_buckets_with_gaps_filled(db_session):

    # 11:00-12:00 bucket: 2 completed (10s, 30s) and 1 failed (50s).
    await add_job(db_session, JobStatus.COMPLETED, datetime(2026, 10, 7, 11, 5), 10)
    await add_job(db_session, JobStatus.COMPLETED, datetime(2026, 10, 7, 11, 40), 30)
    await add_job(db_session, JobStatus.FAILED, datetime(2026, 10, 7, 11, 59), 50)

    # 08:00 bucket: 1 completed.
    await add_job(db_session, JobStatus.COMPLETED, datetime(2026, 10, 7, 8, 15), 20)

    series = await build_job_timeseries(db_session, "24h", now=NOW)

    assert series["range"] == "24h"
    assert series["bucket_seconds"] == 3600

    buckets = series["buckets"]

    # Whole hours, oldest first, every hour present (13:00 yesterday to
    # 12:00 today, the current hour last).
    assert len(buckets) == 24
    assert buckets[0]["start"] == datetime(2026, 10, 6, 13, 0)
    assert buckets[-1]["start"] == datetime(2026, 10, 7, 12, 0)

    by_start = {bucket["start"]: bucket for bucket in buckets}

    eleven = by_start[datetime(2026, 10, 7, 11, 0)]

    assert eleven["completed"] == 2
    assert eleven["failed"] == 1
    # p95 of 10, 30, 50 (interpolated).
    assert eleven["p95_latency_seconds"] == pytest.approx(48)

    assert by_start[datetime(2026, 10, 7, 8, 0)]["completed"] == 1

    # An empty hour: zero jobs, and no latency (not 0 seconds).
    empty = by_start[datetime(2026, 10, 7, 9, 0)]

    assert empty["completed"] == 0
    assert empty["failed"] == 0
    assert empty["p95_latency_seconds"] is None


@pytest.mark.asyncio
async def test_only_finished_jobs_inside_the_range(db_session):

    # Too old, still running, and pending: none of them count.
    await add_job(db_session, JobStatus.COMPLETED, NOW - timedelta(days=2))

    task = await create_task(db_session)
    db_session.add_all(
        [
            ResearchJob(
                task_id=task.id,
                status=JobStatus.RUNNING,
                attempts=1,
                started_at=NOW - timedelta(minutes=5),
            ),
            ResearchJob(task_id=task.id, status=JobStatus.PENDING, attempts=0),
        ]
    )
    await db_session.flush()

    series = await build_job_timeseries(db_session, "24h", now=NOW)

    assert sum(bucket["completed"] for bucket in series["buckets"]) == 0
    assert sum(bucket["failed"] for bucket in series["buckets"]) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "range_name, bucket_seconds, count",
    [("24h", 3600, 24), ("7d", 6 * 3600, 28), ("30d", 86400, 30)],
)
async def test_bucket_size_per_range(db_session, range_name, bucket_seconds, count):

    series = await build_job_timeseries(db_session, range_name, now=NOW)

    assert series["bucket_seconds"] == bucket_seconds
    assert len(series["buckets"]) == count


@pytest.mark.asyncio
async def test_timeseries_endpoint(clean_tables):

    response = client.get("/research/metrics/timeseries?range=7d")

    assert response.status_code == 200

    body = response.json()

    assert body["range"] == "7d"
    assert len(body["buckets"]) == 28
    assert body["buckets"][0]["start"].endswith("Z")

    assert client.get("/research/metrics/timeseries").json()["range"] == "24h"
    assert client.get("/research/metrics/timeseries?range=1y").status_code == 422
