from datetime import datetime, timedelta

import pytest

from app.db.models import (
    JobEvent,
    JobStatus,
    ResearchJob,
)

from app.services.slo_window_service import (
    calculate_job_latency_percentiles,
    calculate_job_success_rate,
    calculate_percentile,
    calculate_tool_latency_percentiles,
    calculate_tool_success_rate,
    get_jobs_in_window,
    get_tool_events_in_window,
)


def test_job_success_rate():
    jobs = [
        ResearchJob(
            status=JobStatus.COMPLETED,
        ),
        ResearchJob(
            status=JobStatus.COMPLETED,
        ),
        ResearchJob(
            status=JobStatus.FAILED,
        ),
    ]

    result = calculate_job_success_rate(jobs)

    assert result == 2 / 3


def test_empty_job_window():
    result = calculate_job_success_rate([])

    assert result == 1.0


def test_tool_success_rate():
    events = [
        JobEvent(
            event_type="tool_completed",
            metadata_json='{"success": true}',
        ),
        JobEvent(
            event_type="tool_completed",
            metadata_json='{"success": true}',
        ),
        JobEvent(
            event_type="tool_completed",
            metadata_json='{"success": false}',
        ),
    ]

    result = calculate_tool_success_rate(events)

    assert result == 2 / 3


def test_empty_tool_window():
    result = calculate_tool_success_rate([])

    assert result == 1.0


# -------------------------
# The same rules as sli_service
# -------------------------


def test_unfinished_jobs_dont_count():
    jobs = [
        ResearchJob(status=JobStatus.COMPLETED),
        ResearchJob(status=JobStatus.PENDING),
        ResearchJob(status=JobStatus.RUNNING),
    ]

    # Only the finished job counts, and it succeeded.
    assert calculate_job_success_rate(jobs) == 1.0
    assert calculate_job_success_rate(jobs[1:]) == 1.0


def test_tool_failed_events_and_other_events():
    events = [
        JobEvent(event_type="tool_completed", metadata_json='{"success": true}'),
        # No success key: the event type decides.
        JobEvent(event_type="tool_failed", metadata_json='{"tool": "tavily"}'),
        JobEvent(event_type="tool_completed", metadata_json=None),
        # Not finished tool calls: ignored.
        JobEvent(event_type="tool_started", metadata_json='{"tool": "tavily"}'),
        JobEvent(event_type="claimed", metadata_json='{"worker_id": "w-1"}'),
    ]

    assert calculate_tool_success_rate(events) == 2 / 3


def test_corrupt_metadata_counts_by_event_type():
    events = [
        JobEvent(event_type="tool_completed", metadata_json="{not json"),
        JobEvent(event_type="tool_failed", metadata_json="{not json"),
    ]

    assert calculate_tool_success_rate(events) == 0.5


@pytest.mark.asyncio
async def test_matches_the_sql_counts(db_session):
    """The same window measured in Python and in SQL (sli_service)."""

    from sqlalchemy import select

    from app.jobs.events import record_tool_event
    from app.services.sli_service import (
        job_success_counts,
        tool_success_counts,
    )
    from app.tests.test_workflow_service import create_task

    now = datetime(2026, 10, 7, 12, 0, 0)
    start = now - timedelta(days=30)

    task = await create_task(db_session)

    for status, finished_days_ago in (
        (JobStatus.COMPLETED, 1),
        (JobStatus.COMPLETED, 2),
        (JobStatus.FAILED, 3),
        (JobStatus.FAILED, 40),  # outside the window
        (JobStatus.PENDING, None),
    ):
        db_session.add(
            ResearchJob(
                task_id=task.id,
                status=status,
                attempts=1,
                completed_at=(
                    now - timedelta(days=finished_days_ago)
                    if finished_days_ago is not None
                    else None
                ),
            )
        )

    await db_session.flush()

    [job, *_] = (await db_session.execute(select(ResearchJob))).scalars().all()

    for event_type, success, recorded_days_ago in (
        ("tool_completed", True, 1),
        ("tool_completed", True, 2),
        ("tool_failed", False, 3),
        ("tool_completed", True, 3),
        ("tool_failed", False, 45),  # outside the window
    ):
        event = await record_tool_event(
            db=db_session,
            job_id=job.id,
            event_type=event_type,
            tool="tavily",
            query="RAG",
            success=success,
            duration_ms=500,
        )
        event.created_at = now - timedelta(days=recorded_days_ago)

    await db_session.flush()

    # The rows in the window, loaded...
    jobs_in_window = [
        job
        for job in (await db_session.execute(select(ResearchJob))).scalars()
        if job.completed_at is not None and start <= job.completed_at <= now
    ]
    events_in_window = [
        event
        for event in (await db_session.execute(select(JobEvent))).scalars()
        if start <= event.created_at <= now
    ]

    # ...give the same rates as the SQL counts.
    finished, failed = await job_success_counts(db_session, start, now)
    assert calculate_job_success_rate(jobs_in_window) == pytest.approx(
        (finished - failed) / finished
    )

    total, failed = await tool_success_counts(db_session, start, now)
    assert calculate_tool_success_rate(events_in_window) == pytest.approx(
        (total - failed) / total
    )


# -------------------------
# calculate_percentile
# -------------------------


def test_percentile_empty():
    assert calculate_percentile([], 0.95) is None


def test_percentile_single_value():
    assert calculate_percentile([42.0], 0.99) == 42.0


def test_percentile_interpolates():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]

    assert calculate_percentile(values, 0.0) == 1.0
    assert calculate_percentile(values, 0.5) == 3.0
    assert calculate_percentile(values, 1.0) == 5.0
    # 0.95 * 4 = position 3.8: 80% of the way from 4 to 5.
    assert calculate_percentile(values, 0.95) == pytest.approx(4.8)


def test_percentile_sorts_its_input():
    assert calculate_percentile([5.0, 1.0, 3.0], 0.5) == 3.0


def test_percentile_out_of_range():
    for bad in (-0.1, 1.5, 95):
        with pytest.raises(ValueError):
            calculate_percentile([1.0, 2.0], bad)


@pytest.mark.asyncio
async def test_percentile_matches_postgres(db_session):
    """Same values, same answer as percentile_cont (latency_service)."""

    from sqlalchemy import column, func, select, values as sql_values
    from sqlalchemy import Float

    numbers = [0.4, 12.0, 3.3, 7.7, 1.1, 30.0, 2.2, 9.9, 15.5, 0.8, 4.4]

    table = sql_values(column("x", Float), name="numbers").data(
        [(number,) for number in numbers]
    )

    for percentile in (0.0, 0.25, 0.5, 0.9, 0.95, 0.99, 1.0):

        postgres = await db_session.scalar(
            select(
                func.percentile_cont(percentile).within_group(table.c.x)
            )
        )

        assert calculate_percentile(numbers, percentile) == pytest.approx(
            postgres
        ), percentile


# -------------------------
# Loading a window and its latency
# -------------------------


@pytest.mark.asyncio
async def test_window_loaders_and_latency_percentiles(db_session):

    from app.jobs.events import record_tool_event
    from app.tests.test_workflow_service import create_task

    now = datetime(2026, 10, 7, 12, 0, 0)
    start = now - timedelta(days=30)

    task = await create_task(db_session)

    # Finished jobs that ran 10s, 20s ... 100s, inside the window; plus one
    # outside it and one still running.
    for seconds in range(10, 101, 10):
        finished = now - timedelta(days=1)
        db_session.add(
            ResearchJob(
                task_id=task.id,
                status=JobStatus.COMPLETED,
                attempts=1,
                started_at=finished - timedelta(seconds=seconds),
                completed_at=finished,
            )
        )

    db_session.add_all(
        [
            ResearchJob(
                task_id=task.id,
                status=JobStatus.FAILED,
                attempts=1,
                started_at=now - timedelta(days=40, seconds=999),
                completed_at=now - timedelta(days=40),
            ),
            ResearchJob(
                task_id=task.id,
                status=JobStatus.RUNNING,
                attempts=1,
                started_at=now - timedelta(minutes=5),
            ),
        ]
    )
    await db_session.flush()

    jobs = await get_jobs_in_window(db_session, start, now)

    assert len(jobs) == 10

    latency = calculate_job_latency_percentiles(jobs)

    assert latency["p50"] == pytest.approx(55)
    assert latency["p95"] == pytest.approx(95.5)
    assert latency["p99"] == pytest.approx(99.1)

    # Tool calls of 100ms ... 1000ms inside the window, one outside, and a
    # tool_started (not a finished call).
    job = jobs[0]

    for number in range(1, 11):
        event = await record_tool_event(
            db=db_session,
            job_id=job.id,
            event_type="tool_completed",
            tool="tavily",
            query="RAG",
            success=True,
            duration_ms=number * 100,
        )
        event.created_at = now - timedelta(hours=number)

    outside = await record_tool_event(
        db=db_session,
        job_id=job.id,
        event_type="tool_failed",
        tool="tavily",
        query="RAG",
        success=False,
        duration_ms=60_000,
    )
    outside.created_at = now - timedelta(days=31)

    started = await record_tool_event(
        db=db_session,
        job_id=job.id,
        event_type="tool_started",
        tool="tavily",
        query="RAG",
    )
    started.created_at = now - timedelta(hours=1)

    await db_session.flush()

    events = await get_tool_events_in_window(db_session, start, now)

    assert len(events) == 10
    assert calculate_tool_success_rate(events) == 1.0

    tool_latency = calculate_tool_latency_percentiles(events)

    assert tool_latency["p50"] == pytest.approx(0.55)
    assert tool_latency["p95"] == pytest.approx(0.955)


def test_latency_percentiles_with_nothing_measured():
    assert calculate_job_latency_percentiles([]) == {
        "p50": None,
        "p95": None,
        "p99": None,
    }
    assert calculate_tool_latency_percentiles([]) == {
        "p50": None,
        "p95": None,
        "p99": None,
    }
