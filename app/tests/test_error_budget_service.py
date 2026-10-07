"""Error budgets: calculate_error_budget, and the report built from the
database (GET /research/slo/error-budget)."""

from datetime import timedelta

import pytest

from app.db.models import JobStatus, ResearchJob, utc_now
from app.jobs.events import record_tool_event
from app.schemas.research import ErrorBudgetReportResponse
from app.services.error_budget_service import (
    ErrorBudget,
    build_error_budget_report,
    calculate_error_budget,
)
from app.tests.test_research_api import client
from app.tests.test_workflow_service import create_task


# -------------------------
# calculate_error_budget
# -------------------------


def test_untouched_budget():

    budget = calculate_error_budget(slo_target=0.99, actual=1.0)

    assert budget == ErrorBudget(
        slo_target=0.99,
        actual=1.0,
        allowed_failure_rate=pytest.approx(0.01),
        actual_failure_rate=0.0,
        budget_remaining=1.0,
        budget_remaining_percent=100.0,
        exhausted=False,
    )


def test_partly_used_budget():

    # 0.4% failing of the 1% allowed: 60% left.
    budget = calculate_error_budget(slo_target=0.99, actual=0.996)

    assert budget.actual_failure_rate == pytest.approx(0.004)
    assert budget.budget_remaining == pytest.approx(0.6)
    assert budget.budget_remaining_percent == pytest.approx(60.0)
    assert budget.exhausted is False


def test_overspent_budget_stops_at_zero():

    budget = calculate_error_budget(slo_target=0.99, actual=0.95)

    assert budget.budget_remaining == 0.0
    assert budget.budget_remaining_percent == 0.0
    assert budget.exhausted is True


def test_perfect_slo_has_no_budget():

    # A 100% target allows no failures at all.
    assert calculate_error_budget(1.0, 1.0).exhausted is False
    assert calculate_error_budget(1.0, 0.999).exhausted is True
    assert calculate_error_budget(1.0, 1.0).budget_remaining == 0.0


# -------------------------
# The report
# -------------------------


async def add_jobs(db, completed: int, failed: int, created_ago=timedelta(0)):

    task = await create_task(db)

    # Finished jobs: they completed (or failed) when created, here.
    finished_at = utc_now() - created_ago

    db.add_all(
        ResearchJob(
            task_id=task.id,
            status=status,
            attempts=1,
            created_at=finished_at,
            completed_at=finished_at,
        )
        for status in (
            [JobStatus.COMPLETED] * completed + [JobStatus.FAILED] * failed
        )
    )

    await db.commit()


@pytest.mark.asyncio
async def test_report_with_no_data(db_session):

    report = await build_error_budget_report(db_session)

    assert report["window_days"] == 30

    assert set(report["budgets"]) == {
        "research_job_success",
        "tool_success",
        "job_latency",
        "tool_latency",
    }

    # Nothing measured: no data, not a healthy full budget.
    for budget in report["budgets"].values():
        assert budget["total_events"] == 0
        assert budget["actual"] is None
        assert budget["budget_remaining"] is None
        assert budget["budget_consumed"] is None
        assert budget["burn_rate_1h"] is None
        assert budget["exhausted"] is False
        assert budget["status"] == "no_data"

    ErrorBudgetReportResponse(**report)


@pytest.mark.asyncio
async def test_job_success_budget(db_session):

    # 1,000 finished jobs at 99% allow 10 failures; 4 used: 60% left.
    # All in the last hour, so failing at 0.4% / 1% = 0.4x the budget pace.
    await add_jobs(db_session, completed=996, failed=4)

    budget = (await build_error_budget_report(db_session))["budgets"][
        "research_job_success"
    ]

    assert budget["total_events"] == 1000
    assert budget["bad_events"] == 4
    assert budget["allowed_bad_events"] == pytest.approx(10)
    assert budget["budget_consumed"] == pytest.approx(0.4)
    assert budget["budget_remaining"] == pytest.approx(0.6)
    assert budget["budget_remaining_percent"] == pytest.approx(60.0)
    assert budget["burn_rate_1h"] == pytest.approx(0.4)
    assert budget["status"] == "healthy"


@pytest.mark.asyncio
async def test_job_success_budget_at_risk_and_exhausted(db_session):

    # 8 of 10 allowed failures: 20% left, under the 25% line.
    await add_jobs(db_session, completed=992, failed=8)

    budget = (await build_error_budget_report(db_session))["budgets"][
        "research_job_success"
    ]

    assert budget["budget_remaining"] == pytest.approx(0.2)
    assert budget["status"] == "at_risk"

    # 20 more failures: overspent.
    await add_jobs(db_session, completed=0, failed=20)

    budget = (await build_error_budget_report(db_session))["budgets"][
        "research_job_success"
    ]

    assert budget["budget_consumed"] > 1
    assert budget["budget_remaining"] == 0.0
    assert budget["status"] == "exhausted"


@pytest.mark.asyncio
async def test_burn_rate_only_counts_the_last_hour(db_session):

    # Failures a day ago spend the budget but don't burn it now.
    await add_jobs(
        db_session,
        completed=90,
        failed=10,
        created_ago=timedelta(days=1),
    )
    await add_jobs(db_session, completed=100, failed=0)

    budget = (await build_error_budget_report(db_session))["budgets"][
        "research_job_success"
    ]

    assert budget["bad_events"] == 10
    assert budget["status"] == "exhausted"
    assert budget["burn_rate_1h"] == 0.0


@pytest.mark.asyncio
async def test_fast_burn(db_session):

    # Failing 20% of the time in the last hour: 20x the 1% budget pace.
    await add_jobs(db_session, completed=80, failed=20)

    budget = (await build_error_budget_report(db_session))["budgets"][
        "research_job_success"
    ]

    assert budget["burn_rate_1h"] == pytest.approx(20)


@pytest.mark.asyncio
async def test_job_latency_budget(db_session):

    # p95 target: 5% of finished jobs may take over 60s. 100 jobs, 3 slow:
    # 3 of 5 allowed used, 40% left.
    task = await create_task(db_session)
    finished = utc_now() - timedelta(minutes=30)

    db_session.add_all(
        ResearchJob(
            task_id=task.id,
            status=JobStatus.COMPLETED,
            attempts=1,
            started_at=finished - timedelta(seconds=seconds),
            completed_at=finished,
        )
        for seconds in [10] * 97 + [120] * 3
    )
    await db_session.commit()

    budget = (await build_error_budget_report(db_session))["budgets"][
        "job_latency"
    ]

    assert budget["target"] == 60.0
    assert budget["total_events"] == 100
    assert budget["bad_events"] == 3
    assert budget["allowed_bad_events"] == pytest.approx(5)
    assert budget["budget_remaining"] == pytest.approx(0.4)
    assert budget["status"] == "healthy"


@pytest.mark.asyncio
async def test_tool_budgets(db_session):

    task = await create_task(db_session)
    job = ResearchJob(task_id=task.id, status=JobStatus.COMPLETED, attempts=1)
    db_session.add(job)
    await db_session.flush()

    # 50 calls: 1 failed (of 1 allowed at 98%), 2 slower than 15s.
    for number in range(50):
        await record_tool_event(
            db=db_session,
            job_id=job.id,
            event_type="tool_failed" if number == 0 else "tool_completed",
            tool="tavily",
            query="RAG",
            success=number != 0,
            duration_ms=20_000 if number in (1, 2) else 500,
        )

    await db_session.commit()

    budgets = (await build_error_budget_report(db_session))["budgets"]

    tool_success = budgets["tool_success"]

    assert tool_success["bad_events"] == 1
    assert tool_success["allowed_bad_events"] == pytest.approx(1)
    assert tool_success["status"] == "exhausted"

    tool_latency = budgets["tool_latency"]

    # 2 of 2.5 allowed.
    assert tool_latency["bad_events"] == 2
    assert tool_latency["budget_remaining"] == pytest.approx(0.2)
    assert tool_latency["status"] == "at_risk"


@pytest.mark.asyncio
async def test_error_budget_endpoint(clean_tables):

    response = client.get("/research/slo/error-budget")

    assert response.status_code == 200

    body = response.json()

    assert body["window_days"] == 30
    assert body["budgets"]["research_job_success"]["status"] == "no_data"
    assert body["budgets"]["job_latency"]["budget_remaining_percent"] is None


# -------------------------
# calculate_error_budget: more cases
# -------------------------


def test_error_budget_half_remaining():
    result = calculate_error_budget(
        slo_target=0.99,
        actual=0.995,
    )

    # approx: in floating point 1 - 0.99 is 0.010000000000000009.
    assert result.allowed_failure_rate == pytest.approx(0.01)
    assert result.actual_failure_rate == pytest.approx(0.005)
    assert result.budget_remaining_percent == 50.0
    assert result.exhausted is False


def test_error_budget_exhausted():
    result = calculate_error_budget(
        slo_target=0.99,
        actual=0.98,
    )

    assert result.exhausted is True
    assert result.budget_remaining == 0.0


def test_perfect_service():
    result = calculate_error_budget(
        slo_target=0.99,
        actual=1.0,
    )

    assert result.budget_remaining_percent == 100.0
    assert result.exhausted is False


# -------------------------
# Response schemas
# -------------------------


def test_error_budget_response_from_calculation():

    from dataclasses import asdict

    from app.schemas.research import ErrorBudgetResponse

    response = ErrorBudgetResponse(
        **asdict(calculate_error_budget(slo_target=0.99, actual=0.995))
    )

    assert response.budget_remaining_percent == 50.0
    assert response.exhausted is False


@pytest.mark.asyncio
async def test_report_entries_include_the_error_budget_fields(db_session):

    await add_jobs(db_session, completed=995, failed=5)

    report = await build_error_budget_report(db_session)
    budget = report["budgets"]["research_job_success"]

    assert budget["slo_target"] == 0.99
    assert budget["actual"] == pytest.approx(0.995)
    assert budget["allowed_failure_rate"] == pytest.approx(0.01)
    assert budget["actual_failure_rate"] == pytest.approx(0.005)
    assert budget["budget_remaining_percent"] == pytest.approx(50.0)
    assert budget["exhausted"] is False

    # Latency SLOs: target in seconds, slo_target the p95 share.
    latency = report["budgets"]["job_latency"]

    assert latency["target"] == 60.0
    assert latency["slo_target"] == pytest.approx(0.95)

    ErrorBudgetReportResponse(**report)

