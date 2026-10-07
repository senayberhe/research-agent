"""SLO report: each SLI over the rolling window against its SLO."""

from datetime import timedelta

import pytest

from app.core.slo import JOB_LATENCY, RESEARCH_JOB_SUCCESS, TOOL_SUCCESS
from app.db.models import JobStatus, ResearchJob, utc_now
from app.jobs.events import record_tool_event
from app.schemas.research import SLOReportResponse
from app.services.slo_service import build_slo_report
from app.tests.test_research_api import client
from app.tests.test_workflow_service import create_task


FINISHED = (JobStatus.COMPLETED, JobStatus.FAILED)


async def add_jobs(db, *statuses, created_days_ago=0, finished_days_ago=None):
    """Jobs with these statuses, created that many days ago. Completed and
    failed ones finished finished_days_ago (default: when created), as a
    real finished job has a completed_at."""

    task = await create_task(db)
    now = utc_now()
    created_at = now - timedelta(days=created_days_ago)

    finished_at = (
        now - timedelta(days=finished_days_ago)
        if finished_days_ago is not None
        else created_at
    )

    jobs = [
        ResearchJob(
            task_id=task.id,
            status=status,
            attempts=0 if status == JobStatus.PENDING else 1,
            created_at=created_at,
            completed_at=finished_at if status in FINISHED else None,
        )
        for status in statuses
    ]

    db.add_all(jobs)
    await db.flush()

    return jobs


async def add_timed_jobs(db, seconds: list[float], status=JobStatus.COMPLETED):
    """Finished jobs (inside the window) that each ran the given number of
    seconds, from started_at to completed_at."""

    task = await create_task(db)
    finished = utc_now() - timedelta(hours=1)

    db.add_all(
        [
            ResearchJob(
                task_id=task.id,
                status=status,
                attempts=1,
                started_at=finished - timedelta(seconds=duration),
                completed_at=finished,
            )
            for duration in seconds
        ]
    )
    await db.commit()


async def add_tool_calls(
    db,
    succeeded: int,
    failed: int,
    duration_ms=500,
    tool="tavily",
):

    [job] = await add_jobs(db, JobStatus.COMPLETED)

    for success in [True] * succeeded + [False] * failed:
        await record_tool_event(
            db=db,
            job_id=job.id,
            event_type="tool_completed" if success else "tool_failed",
            tool=tool,
            query="RAG",
            success=success,
            duration_ms=duration_ms,
        )

    await db.commit()


@pytest.mark.asyncio
async def test_report_with_no_data(db_session):

    report = await build_slo_report(db_session)

    assert report["api_availability"] == {
        "target": 0.999,
        "actual": None,
        "status": "unknown",
        "error_budget": None,
    }

    # Nothing measured: no data, not healthy (an outage that stops all
    # work also produces no failures).
    for name in ("research_job_success", "tool_success"):
        assert report[name]["actual"] is None
        assert report[name]["status"] == "no_data"

    for name in ("job_latency", "tool_latency"):
        assert report[name]["actual_ms"] is None
        assert report[name]["status"] == "no_data"

    # No budget figures either: not "100% left".
    for name in ("research_job_success", "tool_success", "job_latency", "tool_latency"):
        budget = report[name]["error_budget"]

        assert budget["actual"] is None
        assert budget["budget_remaining"] is None
        assert budget["budget_remaining_percent"] is None
        assert budget["exhausted"] is False

    assert report["job_latency"]["percentiles"] == {
        "p50_ms": None,
        "p95_ms": None,
        "p99_ms": None,
    }
    assert report["tool_latency"]["by_tool"] == {}

    SLOReportResponse(**report)


@pytest.mark.asyncio
async def test_job_success_ignores_unfinished_jobs(db_session):

    # 99 completed, 1 failed: exactly the 99% target. Pending and running
    # jobs aren't failures.
    await add_jobs(
        db_session,
        *[JobStatus.COMPLETED] * 99,
        JobStatus.FAILED,
        *[JobStatus.PENDING] * 20,
        JobStatus.RUNNING,
    )

    report = await build_slo_report(db_session)

    job_success = report["research_job_success"]

    assert job_success["target"] == RESEARCH_JOB_SUCCESS.target
    assert job_success["actual"] == pytest.approx(0.99)
    assert job_success["status"] == "healthy"


@pytest.mark.asyncio
async def test_job_success_breached(db_session):

    await add_jobs(db_session, *[JobStatus.COMPLETED] * 97, *[JobStatus.FAILED] * 3)

    job_success = (await build_slo_report(db_session))["research_job_success"]

    assert job_success["actual"] == pytest.approx(0.97)
    assert job_success["status"] == "breached"


@pytest.mark.asyncio
async def test_failures_outside_the_window_stop_counting(db_session):

    # A bad stretch 40 days ago, all good since.
    await add_jobs(db_session, *[JobStatus.FAILED] * 50, created_days_ago=40)
    await add_jobs(db_session, *[JobStatus.COMPLETED] * 10)

    report = await build_slo_report(db_session)

    assert report["research_job_success"]["actual"] == 1.0
    assert report["research_job_success"]["status"] == "healthy"

    # A 60-day window still sees it.
    wide = await build_slo_report(db_session, window=timedelta(days=60))

    assert wide["research_job_success"]["status"] == "breached"


@pytest.mark.asyncio
async def test_tool_success(db_session):

    await add_tool_calls(db_session, succeeded=49, failed=1)

    tool_success = (await build_slo_report(db_session))["tool_success"]

    assert tool_success["target"] == TOOL_SUCCESS.target
    assert tool_success["actual"] == pytest.approx(0.98)
    assert tool_success["status"] == "healthy"

    await add_tool_calls(db_session, succeeded=0, failed=2)

    tool_success = (await build_slo_report(db_session))["tool_success"]

    assert tool_success["actual"] == pytest.approx(49 / 52)
    assert tool_success["status"] == "breached"


@pytest.mark.asyncio
async def test_job_latency_is_p95_not_average(db_session):

    # 18 jobs of 10s and 2 of 300s: the average (39s) is under the 60s
    # target, but the p95 isn't.
    await add_timed_jobs(db_session, [10] * 18 + [300] * 2)

    job_latency = (await build_slo_report(db_session))["job_latency"]

    assert job_latency["target"] == JOB_LATENCY.target
    assert job_latency["target_seconds"] == 60.0
    assert job_latency["actual_ms"] > 60_000
    assert job_latency["status"] == "breached"


@pytest.mark.asyncio
async def test_job_latency_counts_finished_jobs_only(db_session):

    await add_timed_jobs(db_session, [20] * 10)

    # Still running for an hour: not finished, so no latency yet.
    task = await create_task(db_session)
    db_session.add(
        ResearchJob(
            task_id=task.id,
            status=JobStatus.RUNNING,
            attempts=1,
            started_at=utc_now() - timedelta(hours=1),
        )
    )
    await db_session.commit()

    job_latency = (await build_slo_report(db_session))["job_latency"]

    assert job_latency["actual_ms"] == pytest.approx(20_000)
    assert job_latency["status"] == "healthy"


@pytest.mark.asyncio
async def test_retried_job_counts_its_last_attempt(db_session):

    # Failed after 5 minutes, retried, then completed in 30s: started_at is
    # reset by the retry, so the job's latency is 30s.
    from app.jobs.service import (
        create_research_job,
        mark_job_completed,
        mark_job_failed,
        mark_job_running,
        requeue_job,
    )

    task = await create_task(db_session)
    job = await create_research_job(db=db_session, task_id=task.id)

    await mark_job_running(db=db_session, job=job, worker_id="worker-1")
    job.started_at = utc_now() - timedelta(minutes=10)
    await mark_job_failed(
        db=db_session,
        job=job,
        error="timeout",
        lease_token=job.lease_token,
    )
    await db_session.commit()
    await requeue_job(db=db_session, job=job)

    await mark_job_running(db=db_session, job=job, worker_id="worker-2")
    job.started_at = utc_now() - timedelta(seconds=30)
    await mark_job_completed(db=db_session, job=job, lease_token=job.lease_token)
    await db_session.commit()

    job_latency = (await build_slo_report(db_session))["job_latency"]

    assert job_latency["actual_ms"] == pytest.approx(30_000, rel=0.05)


@pytest.mark.asyncio
async def test_tool_latency(db_session):

    await add_tool_calls(db_session, succeeded=20, failed=0, duration_ms=500)

    tool_latency = (await build_slo_report(db_session))["tool_latency"]

    assert tool_latency["by_tool"]["tavily"]["p95_ms"] == pytest.approx(500)
    assert tool_latency["by_tool"]["tavily"]["status"] == "healthy"
    assert tool_latency["status"] == "healthy"

    await add_tool_calls(db_session, succeeded=20, failed=0, duration_ms=20_000)

    tool_latency = (await build_slo_report(db_session))["tool_latency"]

    assert tool_latency["by_tool"]["tavily"]["p95_ms"] > 15_000
    assert tool_latency["by_tool"]["tavily"]["status"] == "breached"
    assert tool_latency["status"] == "breached"


@pytest.mark.asyncio
async def test_slo_endpoint(clean_tables):

    response = client.get("/research/slo")

    # Not mistaken for GET /research/{task_id}.
    assert response.status_code == 200

    body = response.json()

    assert set(body) == {
        "window_days",
        "window_start",
        "window_end",
        "api_availability",
        "research_job_success",
        "tool_success",
        "job_latency",
        "tool_latency",
    }
    # An empty database: no data, not healthy.
    assert body["research_job_success"]["status"] == "no_data"
    assert body["job_latency"]["target_seconds"] == 60.0


# -------------------------
# Percentiles
# -------------------------


@pytest.mark.asyncio
async def test_job_latency_percentiles(db_session):

    from app.services.latency_service import (
        get_job_latency_percentiles_in_window,
    )

    # 1s, 2s, ... 100s.
    await add_timed_jobs(db_session, list(range(1, 101)))

    now = utc_now()

    percentiles = await get_job_latency_percentiles_in_window(
        db_session,
        window_start=now - timedelta(days=1),
        window_end=now,
    )

    # Interpolated, like Postgres percentile_cont.
    assert percentiles["p50"] == pytest.approx(50.5)
    assert percentiles["p95"] == pytest.approx(95.05)
    assert percentiles["p99"] == pytest.approx(99.01)

    job_latency = (await build_slo_report(db_session))["job_latency"]

    assert job_latency["percentiles"] == {
        "p50_ms": pytest.approx(50_500),
        "p95_ms": pytest.approx(95_050),
        "p99_ms": pytest.approx(99_010),
    }
    # Judged on the p95.
    assert job_latency["actual_ms"] == job_latency["percentiles"]["p95_ms"]
    assert job_latency["status"] == "breached"


@pytest.mark.asyncio
async def test_tool_latency_is_reported_per_tool(db_session):

    # Fast Tavily calls, slow arXiv ones: each tool is judged on its own
    # p95, not on one number for all tools.
    await add_tool_calls(db_session, 95, 0, duration_ms=300, tool="tavily")
    await add_tool_calls(db_session, 5, 0, duration_ms=20_000, tool="arxiv")

    tool_latency = (await build_slo_report(db_session))["tool_latency"]

    assert tool_latency["by_tool"] == {
        "arxiv": {
            "p50_ms": pytest.approx(20_000),
            "p95_ms": pytest.approx(20_000),
            "p99_ms": pytest.approx(20_000),
            "status": "breached",
        },
        "tavily": {
            "p50_ms": pytest.approx(300),
            "p95_ms": pytest.approx(300),
            "p99_ms": pytest.approx(300),
            "status": "healthy",
        },
    }

    # No single "slowest tool" number.
    assert tool_latency["actual_ms"] is None

    # Overall: breached, because one tool is.
    assert tool_latency["status"] == "breached"


@pytest.mark.asyncio
async def test_tool_latency_healthy_only_when_every_tool_is(db_session):

    await add_tool_calls(db_session, 10, 0, duration_ms=300, tool="tavily")
    await add_tool_calls(db_session, 10, 0, duration_ms=900, tool="wikipedia")

    tool_latency = (await build_slo_report(db_session))["tool_latency"]

    assert {
        tool: entry["status"] for tool, entry in tool_latency["by_tool"].items()
    } == {"tavily": "healthy", "wikipedia": "healthy"}
    assert tool_latency["status"] == "healthy"


@pytest.mark.asyncio
async def test_slo_endpoint_includes_percentiles(clean_tables):

    from app.tests.conftest import TestSessionLocal

    async with TestSessionLocal() as db:
        await add_timed_jobs(db, [10, 20, 30])
        await add_tool_calls(db, 3, 0, duration_ms=400, tool="wikipedia")

    body = client.get("/research/slo").json()

    assert body["job_latency"]["percentiles"]["p50_ms"] == pytest.approx(20_000)
    assert body["tool_latency"]["by_tool"]["wikipedia"] == {
        "p50_ms": pytest.approx(400),
        "p95_ms": pytest.approx(400),
        "p99_ms": pytest.approx(400),
        "status": "healthy",
    }

    # Not part of the other SLOs.
    assert body["tool_success"]["percentiles"] is None
    assert body["tool_success"]["by_tool"] is None


# -------------------------
# Error budgets in the report
# -------------------------


@pytest.mark.asyncio
async def test_report_includes_each_error_budget(db_session):

    # 995 of 1,000 finished jobs: half the 1% budget used.
    await add_jobs(
        db_session,
        *[JobStatus.COMPLETED] * 995,
        *[JobStatus.FAILED] * 5,
    )

    report = await build_slo_report(db_session)

    budget = report["research_job_success"]["error_budget"]

    # Exactly the ErrorBudgetResponse fields.
    assert budget == {
        "slo_target": 0.99,
        "actual": pytest.approx(0.995),
        "allowed_failure_rate": pytest.approx(0.01),
        "actual_failure_rate": pytest.approx(0.005),
        "budget_remaining": pytest.approx(0.5),
        "budget_remaining_percent": pytest.approx(50.0),
        "exhausted": False,
    }

    for name in ("tool_success", "job_latency", "tool_latency"):
        assert report[name]["error_budget"]["exhausted"] is False

    SLOReportResponse(**report)


@pytest.mark.asyncio
async def test_report_budgets_match_the_error_budget_report(db_session):

    from app.services.error_budget_service import build_error_budget_report

    await add_jobs(db_session, *[JobStatus.COMPLETED] * 90, *[JobStatus.FAILED] * 10)
    await add_tool_calls(db_session, succeeded=40, failed=2, duration_ms=16_000)

    report = await build_slo_report(db_session)
    budgets = (await build_error_budget_report(db_session))["budgets"]

    # The same numbers as the detailed report (which adds counts, burn
    # rate and status).
    for name, budget in budgets.items():
        assert report[name]["error_budget"] == {
            field: budget[field]
            for field in report[name]["error_budget"]
        }


@pytest.mark.asyncio
async def test_slo_endpoint_includes_error_budgets(clean_tables):

    body = client.get("/research/slo").json()

    assert body["api_availability"]["error_budget"] is None

    budget = body["research_job_success"]["error_budget"]

    assert set(budget) == {
        "slo_target",
        "actual",
        "allowed_failure_rate",
        "actual_failure_rate",
        "budget_remaining",
        "budget_remaining_percent",
        "exhausted",
    }
    # Empty database: no data, not a full budget.
    assert budget["budget_remaining_percent"] is None
    assert budget["exhausted"] is False

