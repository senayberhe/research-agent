"""Collects the numbers behind GET /metrics (Prometheus) from the database.

Everything comes from the database, not from counters kept in a process:
the totals survive worker restarts and are correct however many workers
run, and Prometheus only has to scrape the API. Rows are never deleted, so
the totals only go up and work as Prometheus counters.

Sources:
- research_jobs, plus the queue (get_queue_health): current state, and
  the number of jobs ever created.
- job_attempts: attempts finished, by outcome, and their duration.
- job_events (tool_completed / tool_failed): tool calls and latency.
- agent_runs: runs by status, iterations, tokens, cost.
- the error budget report: each SLO's budget left and burn rate.
"""

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AgentRun, JobAttempt
from app.jobs.models import AttemptOutcome
from app.jobs.service import get_queue_health
from app.services.error_budget_service import build_error_budget_report
from app.services.system_metrics_service import finished_tool_calls


# Histogram bucket upper bounds, in seconds.
JOB_DURATION_BUCKETS = (1, 5, 10, 30, 60, 120, 300, 600, 1200, 1800)
TOOL_LATENCY_BUCKETS = (0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60)


async def _histograms(db, label, seconds, source, buckets) -> dict:
    """{label value: {"buckets": [cumulative count per bucket], "count",
    "sum"}} for the rows of `source` with a value, computed in SQL."""

    has_value = seconds.is_not(None)

    rows = (
        await db.execute(
            select(
                label,
                func.count().filter(has_value),
                func.coalesce(func.sum(seconds), 0.0),
                *(
                    func.count().filter(seconds <= bound)
                    for bound in buckets
                ),
            )
            .select_from(source)
            .where(has_value)
            .group_by(label)
        )
    ).all()

    return {
        value: {
            "count": count,
            "sum": float(total),
            "buckets": list(bucket_counts),
        }
        for value, count, total, *bucket_counts in rows
    }


async def collect_prometheus_snapshot(db: AsyncSession) -> dict:
    """Everything render_prometheus_metrics needs, in one dict."""

    queue = await get_queue_health(db)

    # Job attempts that ended, by outcome (completed, failed,
    # lease_expired, released).
    finished_attempts = JobAttempt.ended_at.is_not(None)

    attempts = {
        AttemptOutcome(outcome).value: count
        for outcome, count in (
            await db.execute(
                select(JobAttempt.outcome, func.count())
                .where(finished_attempts)
                .group_by(JobAttempt.outcome)
            )
        ).all()
    }

    attempt_seconds = func.extract(
        "epoch",
        JobAttempt.ended_at - JobAttempt.started_at,
    )

    attempt_durations = await _histograms(
        db,
        label=JobAttempt.outcome,
        seconds=attempt_seconds,
        source=JobAttempt,
        buckets=JOB_DURATION_BUCKETS,
    )

    # Tool calls, from the job events.
    calls = finished_tool_calls(since=None)

    tool_calls = {
        (tool or "unknown", "success" if success else "failure"): count
        for tool, success, count in (
            await db.execute(
                select(
                    calls.c.tool,
                    calls.c.success.is_(True),
                    func.count(),
                )
                .group_by(calls.c.tool, calls.c.success.is_(True))
            )
        ).all()
    }

    tool_latency = await _histograms(
        db,
        label=func.coalesce(calls.c.tool, "unknown"),
        seconds=calls.c.duration_ms / 1000,
        source=calls,
        buckets=TOOL_LATENCY_BUCKETS,
    )

    # Agent runs.
    runs_by_status = dict(
        (
            await db.execute(
                select(AgentRun.status, func.count())
                .group_by(AgentRun.status)
            )
        ).all()
    )

    (
        iterations,
        input_tokens,
        output_tokens,
        cost,
        unpriced_runs,
    ) = (
        await db.execute(
            select(
                func.coalesce(func.sum(AgentRun.iteration_count), 0),
                func.coalesce(func.sum(AgentRun.input_tokens), 0),
                func.coalesce(func.sum(AgentRun.output_tokens), 0),
                func.coalesce(func.sum(AgentRun.estimated_cost_usd), 0.0),
                func.count().filter(AgentRun.estimated_cost_usd.is_(None)),
            )
        )
    ).one()

    return {
        "queue": queue,
        # Jobs are never deleted, so the total only goes up.
        "jobs_created": sum(queue["jobs"].values()),
        "job_attempts": attempts,
        "job_attempt_durations": attempt_durations,
        "tool_calls": tool_calls,
        "tool_latency": tool_latency,
        "agent_runs": runs_by_status,
        "agent_iterations": iterations,
        "llm_tokens": {
            "input": input_tokens,
            "output": output_tokens,
        },
        "llm_cost_usd": float(cost),
        "unpriced_agent_runs": unpriced_runs,
        "error_budgets": (await build_error_budget_report(db))["budgets"],
    }
