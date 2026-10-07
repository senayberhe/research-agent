from datetime import datetime, timedelta

import pytest

from app.db.models import JobEvent, JobStatus, ResearchJob
from app.services.latency_service import (
    calculate_histogram_quantile,
    calculate_percentile,
    get_job_latency_percentiles_in_window,
    get_tool_latency_percentiles_in_window,
)


def test_histogram_quantile_empty():
    result = calculate_histogram_quantile(
        [],
        0.95,
    )

    assert result is None


def test_histogram_quantile_returns_value():
    buckets = [
        {
            "le": 1.0,
            "count": 10,
        },
        {
            "le": 2.0,
            "count": 20,
        },
        {
            "le": 5.0,
            "count": 30,
        },
    ]

    result = calculate_histogram_quantile(
        buckets,
        0.95,
    )

    assert result is not None
    assert 2.0 <= result <= 5.0

    # Exactly: the 28.5th of 30 observations is 8.5/10 of the way through
    # the 2-5s bucket.
    assert result == pytest.approx(4.55)


def test_histogram_quantile_with_no_observations():
    buckets = [
        {"le": 1.0, "count": 0},
        {"le": 5.0, "count": 0},
    ]

    assert calculate_histogram_quantile(buckets, 0.5) is None


def test_histogram_quantile_in_the_first_bucket():
    # Interpolates from 0 up to the first bound.
    buckets = [
        {"le": 10.0, "count": 100},
        {"le": 20.0, "count": 100},
    ]

    assert calculate_histogram_quantile(buckets, 0.5) == pytest.approx(5.0)


# -------------------------
# calculate_percentile
# -------------------------


def test_calculate_percentile_empty():
    result = calculate_percentile(
        [],
        0.95,
    )

    assert result is None


def test_calculate_percentile_single_value():
    result = calculate_percentile(
        [10.0],
        0.95,
    )

    assert result == 10.0


def test_calculate_percentile_median():
    result = calculate_percentile(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        0.50,
    )

    assert result == 3.0


def test_calculate_percentile_interpolation():
    result = calculate_percentile(
        [10.0, 20.0],
        0.50,
    )

    assert result == 15.0



# -------------------------
# Rolling-window latency
# -------------------------


NOW = datetime(2026, 10, 7, 12, 0, 0)
WINDOW_START = NOW - timedelta(days=30)


async def add_finished_job(db, task_id, seconds, finished_at, status="completed"):
    db.add(
        ResearchJob(
            task_id=task_id,
            status=status,
            attempts=1,
            started_at=finished_at - timedelta(seconds=seconds),
            completed_at=finished_at,
        )
    )


@pytest.mark.asyncio
async def test_job_latency_in_window(db_session):

    from app.tests.test_workflow_service import create_task

    task = await create_task(db_session)

    # Inside: 10s, 20s (completed) and 30s (failed: still a duration).
    await add_finished_job(db_session, task.id, 10, NOW - timedelta(days=1))
    await add_finished_job(db_session, task.id, 20, NOW - timedelta(days=29))
    await add_finished_job(
        db_session, task.id, 30, NOW - timedelta(days=2), status="failed"
    )

    # Outside: finished 31 days ago, or after the window's end.
    await add_finished_job(db_session, task.id, 999, NOW - timedelta(days=31))
    await add_finished_job(db_session, task.id, 999, NOW + timedelta(minutes=1))

    # Running: not finished.
    db_session.add(
        ResearchJob(
            task_id=task.id,
            status=JobStatus.RUNNING,
            attempts=1,
            started_at=NOW - timedelta(hours=5),
        )
    )
    await db_session.flush()

    latency = await get_job_latency_percentiles_in_window(
        db_session,
        window_start=WINDOW_START,
        window_end=NOW,
    )

    assert latency["p50"] == pytest.approx(20)
    assert latency["p95"] == pytest.approx(29)
    assert latency["p99"] == pytest.approx(29.8)


@pytest.mark.asyncio
async def test_tool_latency_in_window(db_session):

    from app.tests.test_workflow_service import create_task

    task = await create_task(db_session)
    job = ResearchJob(task_id=task.id, status=JobStatus.COMPLETED, attempts=1)
    db_session.add(job)
    await db_session.flush()

    def event(event_type, metadata_json, days_ago=1):
        return JobEvent(
            job_id=job.id,
            event_type=event_type,
            metadata_json=metadata_json,
            created_at=NOW - timedelta(days=days_ago),
        )

    db_session.add_all(
        [
            event("tool_completed", '{"tool": "tavily", "duration_ms": 200}'),
            event("tool_completed", '{"tool": "tavily", "duration_ms": 400}'),
            # A timeout: failed, and the slowest. It counts.
            event("tool_failed", '{"tool": "tavily", "duration_ms": 30000}'),
            event("tool_completed", '{"tool": "arxiv", "duration_ms": 1000}'),
            # Not counted: outside the window, not a finished call, no
            # duration, or unreadable metadata.
            event("tool_completed", '{"tool": "tavily", "duration_ms": 99999}', 40),
            event("tool_started", '{"tool": "tavily"}'),
            event("tool_completed", '{"tool": "tavily"}'),
            event("tool_completed", "{not json"),
        ]
    )
    await db_session.flush()

    latency = await get_tool_latency_percentiles_in_window(
        db_session,
        window_start=WINDOW_START,
        window_end=NOW,
    )

    assert set(latency) == {"arxiv", "tavily"}

    assert latency["tavily"]["p50"] == pytest.approx(0.4)
    # 0.95 * 2 = position 1.9: 90% of the way from 0.4s to 30s (0.4 + 0.9 * 29.6).
    assert latency["tavily"]["p95"] == pytest.approx(27.04)

    assert latency["arxiv"] == {"p50": 1.0, "p95": 1.0, "p99": 1.0}


@pytest.mark.asyncio
async def test_empty_window_has_no_latency(db_session):

    assert await get_job_latency_percentiles_in_window(
        db_session,
        window_start=WINDOW_START,
        window_end=NOW,
    ) == {"p50": None, "p95": None, "p99": None}

    assert await get_tool_latency_percentiles_in_window(
        db_session,
        window_start=WINDOW_START,
        window_end=NOW,
    ) == {}


@pytest.mark.asyncio
async def test_latency_budget_counts_the_same_jobs(db_session):
    """The error budget's slow-job count and the percentiles cover the same
    jobs: 2 of 20 finished jobs took over 60s, so the p95 is over 60s and
    2 jobs spent the latency budget."""

    from app.services.error_budget_service import build_error_budget_report
    from app.services.slo_service import build_slo_report
    from app.tests.test_workflow_service import create_task

    task = await create_task(db_session)

    for seconds in [10] * 18 + [120] * 2:
        await add_finished_job(db_session, task.id, seconds, NOW - timedelta(days=3))

    await db_session.flush()

    report = await build_slo_report(db_session, now=NOW)
    budget = (await build_error_budget_report(db_session, now=NOW))["budgets"][
        "job_latency"
    ]

    assert report["job_latency"]["actual_ms"] > 60_000
    assert report["job_latency"]["status"] == "breached"

    assert budget["total_events"] == 20
    assert budget["bad_events"] == 2
    # 2 slow of 1 allowed (5% of 20): overspent.
    assert budget["exhausted"] is True
