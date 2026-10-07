"""The SLO report: each SLI measured over a rolling window, against its SLO
(app/core/slo.py)."""

from datetime import datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.slo import (
    API_AVAILABILITY,
    JOB_LATENCY,
    RESEARCH_JOB_SUCCESS,
    SLO_WINDOW,
    TOOL_LATENCY,
    TOOL_SUCCESS,
    window_bounds,
)
from app.db.models import utc_now
from app.services.error_budget_service import (
    ErrorBudget,
    build_error_budget_report,
)
from app.services.latency_service import (
    get_job_latency_percentiles_in_window,
    get_tool_latency_percentiles_in_window,
)
from app.services.sli_service import job_success_counts, tool_success_counts


ERROR_BUDGET_FIELDS = tuple(ErrorBudget.__dataclass_fields__)


def _to_ms(seconds: float | None) -> float | None:
    return round(seconds * 1000, 2) if seconds is not None else None


def _percentiles_ms(percentiles: dict) -> dict:
    """{"p50", "p95", "p99"} in seconds -> {"p50_ms", "p95_ms", "p99_ms"}."""

    return {
        f"{name}_ms": _to_ms(value)
        for name, value in percentiles.items()
    }


def _ratio_status(actual: float | None, target: float) -> str:
    if actual is None:
        return "no_data"

    return "healthy" if actual >= target else "breached"


def _latency_status(p95_seconds: float | None, target: float) -> str:
    if p95_seconds is None:
        return "no_data"

    return "healthy" if p95_seconds <= target else "breached"


def _overall_status(statuses: list[str]) -> str:
    """One status for several (tools): breached if any is, healthy only if
    all with data are, no_data if none has data."""

    measured = [status for status in statuses if status != "no_data"]

    if not measured:
        return "no_data"

    return "breached" if "breached" in measured else "healthy"


async def build_slo_report(
    db: AsyncSession,
    window: timedelta = SLO_WINDOW,
    now: datetime | None = None,
) -> dict:
    """Each SLO over [now - window, now] (SLO_WINDOW_DAYS). An SLO with
    nothing measured in the window is "no_data", not healthy: an outage
    that stops all work also produces no failures."""

    window_end = now or utc_now()
    window_start = window_end - window

    # Jobs that finished (completed or failed) inside the window, by when
    # they finished; pending and running jobs aren't failures.
    finished_jobs, failed_jobs = await job_success_counts(
        db, window_start, window_end
    )

    job_success_rate = (
        (finished_jobs - failed_jobs) / finished_jobs
        if finished_jobs
        else None
    )

    # Tool calls recorded (as job events) inside the window.
    total_tool_calls, failed_tool_calls = await tool_success_counts(
        db, window_start, window_end
    )

    tool_success_rate = (
        (total_tool_calls - failed_tool_calls) / total_tool_calls
        if total_tool_calls
        else None
    )

    # Each SLO's error budget, over the same window: just the ErrorBudget
    # fields (ErrorBudgetResponse).
    budgets = {
        name: {field: budget[field] for field in ERROR_BUDGET_FIELDS}
        for name, budget in (
            await build_error_budget_report(db, window=window, now=window_end)
        )["budgets"].items()
    }

    # p95s (not averages: a few very slow jobs barely move an average) over
    # the same rolling window.
    job_latency = (
        await get_job_latency_percentiles_in_window(
            db=db,
            window_start=window_start,
            window_end=window_end,
        )
    )

    tool_latency = (
        await get_tool_latency_percentiles_in_window(
            db=db,
            window_start=window_start,
            window_end=window_end,
        )
    )

    job_p95 = job_latency["p95"]

    # Each tool is judged on its own p95 against the target.
    tools = {
        tool: {
            **_percentiles_ms(percentiles),
            "status": _latency_status(percentiles["p95"], TOOL_LATENCY.target),
        }
        for tool, percentiles in tool_latency.items()
    }

    return {
        **window_bounds(window_start, window_end),

        "api_availability": {
            "target": API_AVAILABILITY.target,
            # Measured by Prometheus (up{job="research-api"}), not here.
            "actual": None,
            "status": "unknown",
            # Measured by Prometheus, so no budget here either.
            "error_budget": None,
        },

        "research_job_success": {
            "target": RESEARCH_JOB_SUCCESS.target,
            "actual": job_success_rate,
            "status": _ratio_status(
                job_success_rate,
                RESEARCH_JOB_SUCCESS.target,
            ),
            "error_budget": budgets[RESEARCH_JOB_SUCCESS.name],
        },

        "tool_success": {
            "target": TOOL_SUCCESS.target,
            "actual": tool_success_rate,
            "status": _ratio_status(
                tool_success_rate,
                TOOL_SUCCESS.target,
            ),
            "error_budget": budgets[TOOL_SUCCESS.name],
        },

        "job_latency": {
            "target": JOB_LATENCY.target,
            "target_seconds": JOB_LATENCY.target,
            # p95.
            "actual_ms": _to_ms(job_p95),
            "percentiles": _percentiles_ms(job_latency),
            "status": _latency_status(job_p95, JOB_LATENCY.target),
            "error_budget": budgets[JOB_LATENCY.name],
        },

        "tool_latency": {
            "target": TOOL_LATENCY.target,
            "target_seconds": TOOL_LATENCY.target,
            # No single number: each tool has its own p95 and status.
            "actual_ms": None,
            "by_tool": tools,
            "status": _overall_status(
                [tool["status"] for tool in tools.values()]
            ),
            "error_budget": budgets[TOOL_LATENCY.name],
        },
    }
