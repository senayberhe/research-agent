"""Error budgets: how much failure each SLO still allows, and how fast it's
being used up.

An SLO's target leaves an error budget of bad events over the window:

    allowed bad events = (1 - target) * total events

99% job success over 30 days with 1,000 finished jobs allows 10 failures;
after 4, 60% of the budget remains. The latency SLOs are p95 targets, so
5% of events may exceed the threshold: each job (or tool call) slower than
it spends budget.

The burn rate is how fast the budget is going, from the last hour: the
bad-event ratio divided by the allowed ratio. At 1 the budget lasts
exactly the window; at 14.4, 2% of a 30-day budget goes in an hour (the
usual threshold for paging someone).

API availability isn't here: Prometheus measures it (up{job="research-api"}).
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.slo import (
    JOB_LATENCY,
    RESEARCH_JOB_SUCCESS,
    SLO,
    SLO_WINDOW,
    TOOL_LATENCY,
    TOOL_SUCCESS,
    window_bounds,
)
from app.db.models import utc_now
from app.services.sli_service import (
    job_latency_counts,
    job_success_counts,
    tool_latency_counts,
    tool_success_counts,
)


# The latency SLOs are p95 targets: 5% of events may exceed them.
LATENCY_PERCENTILE = 0.95

BURN_RATE_WINDOW = timedelta(hours=1)

# Below this share of budget left, the SLO is "at_risk".
AT_RISK_REMAINING = 0.25


@dataclass(frozen=True)
class ErrorBudget:
    slo_target: float
    actual: float
    allowed_failure_rate: float
    actual_failure_rate: float
    budget_remaining: float
    budget_remaining_percent: float
    exhausted: bool


def calculate_error_budget(
    slo_target: float,
    actual: float,
) -> ErrorBudget:
    """The error budget left when an SLO of slo_target (e.g. 0.99) is met
    at `actual` (the success ratio, e.g. 0.995). budget_remaining is 0-1
    (never below 0); exhausted once it's all used."""

    allowed_failure_rate = 1.0 - slo_target
    actual_failure_rate = 1.0 - actual

    if allowed_failure_rate <= 0:
        return ErrorBudget(
            slo_target=slo_target,
            actual=actual,
            allowed_failure_rate=0.0,
            actual_failure_rate=actual_failure_rate,
            budget_remaining=0.0,
            budget_remaining_percent=0.0,
            exhausted=actual_failure_rate > 0,
        )

    budget_used = actual_failure_rate / allowed_failure_rate

    budget_remaining = max(
        0.0,
        1.0 - budget_used,
    )

    budget_remaining_percent = (
        budget_remaining * 100
    )

    return ErrorBudget(
        slo_target=slo_target,
        actual=actual,
        allowed_failure_rate=allowed_failure_rate,
        actual_failure_rate=actual_failure_rate,
        budget_remaining=budget_remaining,
        budget_remaining_percent=budget_remaining_percent,
        exhausted=budget_remaining <= 0,
    )


def _budget(
    slo: SLO,
    allowed_bad_ratio: float,
    total: int,
    bad: int,
    recent_total: int,
    recent_bad: int,
) -> dict:

    burn_rate = (
        (recent_bad / recent_total) / allowed_bad_ratio
        if recent_total and allowed_bad_ratio
        # Nothing measured in the last hour: no rate, rather than "0x".
        else None
    )

    if total == 0:
        # Nothing measured in the window: no data, which isn't the same as
        # healthy (an outage that stops all jobs also produces no events).
        return {
            "slo": slo.name,
            "target": slo.target,
            "slo_target": round(1.0 - allowed_bad_ratio, 6),
            "actual": None,
            "allowed_failure_rate": round(allowed_bad_ratio, 6),
            "actual_failure_rate": None,
            "budget_remaining": None,
            "budget_remaining_percent": None,
            "exhausted": False,
            "total_events": 0,
            "bad_events": 0,
            "allowed_bad_events": 0.0,
            "budget_consumed": None,
            "burn_rate_1h": None,
            "status": "no_data",
        }

    # The success ratio over the window.
    actual = 1.0 - bad / total

    budget = calculate_error_budget(
        slo_target=1.0 - allowed_bad_ratio,
        actual=actual,
    )

    if budget.allowed_failure_rate:
        consumed = budget.actual_failure_rate / budget.allowed_failure_rate
    else:
        consumed = 0.0 if bad == 0 else float("inf")

    if budget.exhausted:
        status = "exhausted"
    elif budget.budget_remaining < AT_RISK_REMAINING:
        status = "at_risk"
    else:
        status = "healthy"

    return {
        "slo": slo.name,
        "target": slo.target,
        # The ErrorBudget's fields (ErrorBudgetResponse), rounded.
        "slo_target": round(budget.slo_target, 6),
        "actual": round(budget.actual, 6),
        "allowed_failure_rate": round(budget.allowed_failure_rate, 6),
        "actual_failure_rate": round(budget.actual_failure_rate, 6),
        # 0-1, never below 0.
        "budget_remaining": round(budget.budget_remaining, 4),
        "budget_remaining_percent": round(budget.budget_remaining_percent, 2),
        "exhausted": budget.exhausted,
        "total_events": total,
        "bad_events": bad,
        "allowed_bad_events": round(allowed_bad_ratio * total, 2),
        # Above 1 once overspent.
        "budget_consumed": round(consumed, 4),
        "burn_rate_1h": round(burn_rate, 2) if burn_rate is not None else None,
        "status": status,
    }


async def build_error_budget_report(
    db: AsyncSession,
    window: timedelta = SLO_WINDOW,
    now: datetime | None = None,
) -> dict:
    """Each SLO's budget over [now - window, now]. The success SLOs count
    jobs that finished, and tool calls recorded, inside the window
    (sli_service)."""

    now = now or utc_now()
    window_start = now - window
    recent = now - BURN_RATE_WINDOW

    budgets = {}

    for slo, allowed_bad_ratio, counts in (
        (RESEARCH_JOB_SUCCESS, 1 - RESEARCH_JOB_SUCCESS.target, job_success_counts),
        (TOOL_SUCCESS, 1 - TOOL_SUCCESS.target, tool_success_counts),
        (JOB_LATENCY, 1 - LATENCY_PERCENTILE, job_latency_counts),
        (TOOL_LATENCY, 1 - LATENCY_PERCENTILE, tool_latency_counts),
    ):
        total, bad = await counts(db, window_start, now)
        recent_total, recent_bad = await counts(db, recent, now)

        budgets[slo.name] = _budget(
            slo,
            allowed_bad_ratio,
            total,
            bad,
            recent_total,
            recent_bad,
        )

    return {
        **window_bounds(window_start, now),
        "budgets": budgets,
    }
