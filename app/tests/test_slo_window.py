"""The SLO rolling window: records inside it count, records outside it
don't, for job success (by when a job finished), tool success (by when the
JobEvent was recorded) and the error budgets built from them."""

from datetime import UTC, datetime, timedelta

import pytest

from app.core.config import Settings
from app.core.slo import SLO_WINDOW, SLO_WINDOW_DAYS
from app.db.models import JobStatus, ResearchJob
from app.jobs.events import record_tool_event
from app.services.error_budget_service import build_error_budget_report
from app.services.sli_service import job_success_counts, tool_success_counts
from app.services.slo_service import build_slo_report
from app.tests.test_research_api import client
from app.tests.test_workflow_service import create_task


# A fixed "now", so records can be placed exactly relative to the window.
NOW = datetime(2026, 10, 7, 12, 0, 0)
WINDOW_START = NOW - SLO_WINDOW


def days_ago(days: float) -> datetime:
    return NOW - timedelta(days=days)


async def add_job(db, status, created_at, completed_at=None):

    task = await create_task(db)

    job = ResearchJob(
        task_id=task.id,
        status=status,
        attempts=0 if status == JobStatus.PENDING else 1,
        created_at=created_at,
        completed_at=completed_at,
    )

    db.add(job)
    await db.flush()

    return job


async def add_tool_call(db, recorded_at, success=True):

    task = await create_task(db)
    job = ResearchJob(task_id=task.id, status=JobStatus.COMPLETED, attempts=1)
    db.add(job)
    await db.flush()

    event = await record_tool_event(
        db=db,
        job_id=job.id,
        event_type="tool_completed" if success else "tool_failed",
        tool="tavily",
        query="RAG",
        success=success,
        duration_ms=500,
    )
    event.created_at = recorded_at

    await db.flush()


# -------------------------
# The window setting
# -------------------------


def test_window_defaults_to_30_days():

    assert SLO_WINDOW_DAYS == 30
    assert SLO_WINDOW == timedelta(days=30)


def test_window_is_configurable(monkeypatch):

    monkeypatch.setenv("SLO_WINDOW_DAYS", "7")

    assert Settings().slo_window_days == 7


def test_window_must_be_at_least_a_day(monkeypatch):

    from pydantic import ValidationError

    monkeypatch.setenv("SLO_WINDOW_DAYS", "0")

    with pytest.raises(ValidationError):
        Settings()


# -------------------------
# Job success: by when the job finished
# -------------------------


@pytest.mark.asyncio
async def test_jobs_finished_inside_the_window_count(db_session):

    await add_job(db_session, JobStatus.COMPLETED, days_ago(5), days_ago(5))
    await add_job(db_session, JobStatus.FAILED, days_ago(29), days_ago(29))

    assert await job_success_counts(db_session, WINDOW_START, NOW) == (2, 1)


@pytest.mark.asyncio
async def test_jobs_finished_outside_the_window_dont_count(db_session):

    # Finished 31 and 60 days ago: outside a 30-day window.
    await add_job(db_session, JobStatus.FAILED, days_ago(31), days_ago(31))
    await add_job(db_session, JobStatus.COMPLETED, days_ago(60), days_ago(60))

    assert await job_success_counts(db_session, WINDOW_START, NOW) == (0, 0)


@pytest.mark.asyncio
async def test_job_counts_by_finish_time_not_creation_time(db_session):

    # Created before the window but finished inside it: counts.
    await add_job(db_session, JobStatus.FAILED, days_ago(45), days_ago(2))

    # Created inside the window but still pending or running: not finished,
    # so neither a success nor a failure yet.
    await add_job(db_session, JobStatus.PENDING, days_ago(1))
    await add_job(db_session, JobStatus.RUNNING, days_ago(1))

    assert await job_success_counts(db_session, WINDOW_START, NOW) == (1, 1)


@pytest.mark.asyncio
async def test_window_edges(db_session):

    # Exactly at the start and the end: inside. Just past either: outside.
    await add_job(db_session, JobStatus.COMPLETED, WINDOW_START, WINDOW_START)
    await add_job(db_session, JobStatus.COMPLETED, NOW, NOW)
    await add_job(
        db_session,
        JobStatus.FAILED,
        WINDOW_START,
        WINDOW_START - timedelta(seconds=1),
    )
    await add_job(
        db_session,
        JobStatus.FAILED,
        NOW,
        NOW + timedelta(seconds=1),
    )

    assert await job_success_counts(db_session, WINDOW_START, NOW) == (2, 0)


# -------------------------
# Tool success: by when the JobEvent was recorded
# -------------------------


@pytest.mark.asyncio
async def test_tool_calls_inside_vs_outside_the_window(db_session):

    await add_tool_call(db_session, days_ago(1), success=True)
    await add_tool_call(db_session, days_ago(20), success=False)

    # Outside: before the window, and after its end.
    await add_tool_call(db_session, days_ago(31), success=False)
    await add_tool_call(db_session, NOW + timedelta(minutes=5), success=False)

    assert await tool_success_counts(db_session, WINDOW_START, NOW) == (2, 1)


# -------------------------
# The SLO report and error budgets
# -------------------------


@pytest.mark.asyncio
async def test_old_failures_leave_the_slo_and_its_budget(db_session):

    # A bad stretch 40 days ago (outside), all good since (inside).
    for _ in range(20):
        await add_job(db_session, JobStatus.FAILED, days_ago(40), days_ago(40))
    for _ in range(100):
        await add_job(db_session, JobStatus.COMPLETED, days_ago(3), days_ago(3))
    for _ in range(5):
        await add_tool_call(db_session, days_ago(40), success=False)
    for _ in range(50):
        await add_tool_call(db_session, days_ago(3), success=True)

    report = await build_slo_report(db_session, now=NOW)

    assert report["research_job_success"]["actual"] == 1.0
    assert report["research_job_success"]["status"] == "healthy"
    assert report["research_job_success"]["error_budget"]["exhausted"] is False

    assert report["tool_success"]["actual"] == 1.0
    assert report["tool_success"]["error_budget"]["budget_remaining"] == 1.0


@pytest.mark.asyncio
async def test_failures_inside_the_window_spend_the_budget(db_session):

    # 3 of 100 jobs finished as failures inside the window: 3% > the 1%
    # allowed.
    for _ in range(97):
        await add_job(db_session, JobStatus.COMPLETED, days_ago(10), days_ago(10))
    for _ in range(3):
        await add_job(db_session, JobStatus.FAILED, days_ago(10), days_ago(10))

    budgets = (await build_error_budget_report(db_session, now=NOW))["budgets"]

    budget = budgets["research_job_success"]

    assert budget["total_events"] == 100
    assert budget["bad_events"] == 3
    assert budget["exhausted"] is True
    assert budget["status"] == "exhausted"


@pytest.mark.asyncio
async def test_slo_report_and_budgets_use_the_same_window(db_session):

    await add_job(db_session, JobStatus.FAILED, days_ago(10), days_ago(10))

    report = await build_slo_report(db_session, now=NOW)
    budgets = await build_error_budget_report(db_session, now=NOW)

    # The exact instants measured, as explicit UTC.
    for result in (report, budgets):
        assert result["window_start"] == WINDOW_START.replace(tzinfo=UTC)
        assert result["window_end"] == NOW.replace(tzinfo=UTC)
        assert result["window_days"] == 30


@pytest.mark.asyncio
async def test_a_shorter_window(db_session):

    # Failed 10 days ago: inside 30 days, outside 7.
    await add_job(db_session, JobStatus.FAILED, days_ago(10), days_ago(10))
    await add_job(db_session, JobStatus.COMPLETED, days_ago(1), days_ago(1))

    thirty = await build_slo_report(db_session, now=NOW)
    seven = await build_slo_report(
        db_session,
        window=timedelta(days=7),
        now=NOW,
    )

    assert thirty["research_job_success"]["actual"] == 0.5
    assert seven["research_job_success"]["actual"] == 1.0
    assert seven["window_start"] == days_ago(7).replace(tzinfo=UTC)
    assert seven["window_days"] == 7


# -------------------------
# The API
# -------------------------


@pytest.mark.asyncio
async def test_slo_api_returns_the_window(clean_tables):

    for path in ("/research/slo", "/research/slo/error-budget"):

        body = client.get(path).json()

        start = datetime.fromisoformat(body["window_start"])
        end = datetime.fromisoformat(body["window_end"])

        # Explicit UTC, not a bare local-looking time.
        assert body["window_start"].endswith("Z"), path
        assert start.utcoffset() == timedelta(0), path
        assert end.utcoffset() == timedelta(0), path

        assert end - start == timedelta(days=30), path
        assert body["window_days"] == 30, path


# -------------------------
# No data is not healthy
# -------------------------


@pytest.mark.asyncio
async def test_one_slo_without_data_while_others_have_it(db_session):

    # Jobs finished, but no tool was called in the window.
    await add_job(db_session, JobStatus.COMPLETED, days_ago(2), days_ago(2))

    report = await build_slo_report(db_session, now=NOW)

    assert report["research_job_success"]["status"] == "healthy"
    assert report["research_job_success"]["actual"] == 1.0

    assert report["tool_success"]["status"] == "no_data"
    assert report["tool_success"]["actual"] is None
    assert report["tool_latency"]["status"] == "no_data"
    assert report["tool_latency"]["by_tool"] == {}


@pytest.mark.asyncio
async def test_burn_rate_is_unknown_without_recent_events(db_session):

    # Data in the window, nothing in the last hour: the budget is known,
    # the current burn rate isn't (rather than a reassuring 0x).
    await add_job(db_session, JobStatus.FAILED, days_ago(3), days_ago(3))
    await add_job(db_session, JobStatus.COMPLETED, days_ago(3), days_ago(3))

    budget = (await build_error_budget_report(db_session, now=NOW))["budgets"][
        "research_job_success"
    ]

    assert budget["total_events"] == 2
    assert budget["budget_remaining"] == 0.0
    assert budget["burn_rate_1h"] is None


def test_prometheus_leaves_out_no_data_samples():

    from app.core.prometheus import render_prometheus_metrics
    from app.tests.test_prometheus_config import exported_metric_names

    no_data = {
        "slo": "tool_success",
        "actual": None,
        "budget_remaining": None,
        "burn_rate_1h": None,
    }
    measured = {
        "slo": "research_job_success",
        "actual": 0.995,
        "budget_remaining": 0.5,
        "burn_rate_1h": None,
    }

    snapshot = {
        "queue": {
            "jobs": {},
            "oldest_pending_seconds": None,
            "expired_leases": 0,
            "active_workers": [],
        },
        "jobs_created": 0,
        "job_attempts": {},
        "job_attempt_durations": {},
        "tool_calls": {},
        "tool_latency": {},
        "agent_runs": {},
        "agent_iterations": 0,
        "llm_tokens": {},
        "llm_cost_usd": 0.0,
        "unpriced_agent_runs": 0,
        "error_budgets": {"tool_success": no_data, "research_job_success": measured},
    }

    text = render_prometheus_metrics(snapshot).decode()

    assert 'research_slo_sli{slo="research_job_success"} 0.995' in text
    assert 'research_slo_error_budget_remaining{slo="research_job_success"} 0.5' in text

    # No sample (rather than a healthy-looking value) without data.
    assert 'slo="tool_success"' not in text
    assert "research_slo_burn_rate{" not in text

    assert "research_slo_sli" in exported_metric_names()
