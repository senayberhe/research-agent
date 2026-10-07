"""Operational metrics for the whole research system: jobs, agent runs,
tools and workers, aggregated in SQL (never loading every row), over a
time window or all time.

- build_system_metrics: the headline numbers, as one flat dict.
- build_system_monitoring: everything GET /metrics returns (those numbers
  plus per-tool stats, workers, running jobs and the queue).

Per-task metrics are in metrics_service.build_research_metrics."""

from datetime import datetime

from sqlalchemy import case, cast, func, null, select, true, type_coerce
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    AgentRun,
    JobAttempt,
    JobEvent,
    JobEventType,
    JobStatus,
    ResearchJob,
    utc_now,
)
from app.jobs.service import get_queue_health, get_worker_activity


def _percent(part: int, whole: int) -> float | None:
    """part / whole as a percentage, or None when there's nothing to
    measure (rather than a misleading 0%)."""

    return round(part / whole * 100, 2) if whole else None


def _round(value) -> float | None:
    return round(float(value), 2) if value is not None else None


def _seconds(start, end):
    return func.extract("epoch", end - start)


def _in_window(column, since: datetime | None):
    """column >= since, or no condition at all (true) for all time."""

    return column >= since if since is not None else true()


async def job_stats(db: AsyncSession, since: datetime | None) -> dict:
    """Jobs created since `since` (all jobs if None)."""

    in_window = _in_window(ResearchJob.created_at, since)

    by_status = {status.value: 0 for status in JobStatus}

    for status, count in (
        await db.execute(
            select(ResearchJob.status, func.count())
            .where(in_window)
            .group_by(ResearchJob.status)
        )
    ).all():
        by_status[JobStatus(status).value] = count

    started, retried, total_attempts = (
        await db.execute(
            select(
                func.count().filter(ResearchJob.attempts >= 1),
                func.count().filter(ResearchJob.attempts > 1),
                func.coalesce(func.sum(ResearchJob.attempts), 0),
            ).where(in_window)
        )
    ).one()

    # A finished job's duration: all its attempts added up, or its own
    # start to completion if it has no attempt records (older jobs).
    attempt_seconds = (
        select(
            JobAttempt.job_id,
            func.sum(_seconds(JobAttempt.started_at, JobAttempt.ended_at))
            .label("seconds"),
        )
        .where(JobAttempt.ended_at.is_not(None))
        .group_by(JobAttempt.job_id)
        .subquery()
    )

    average_duration_seconds = await db.scalar(
        select(
            func.avg(
                func.coalesce(
                    attempt_seconds.c.seconds,
                    _seconds(ResearchJob.started_at, ResearchJob.completed_at),
                )
            )
        )
        .select_from(ResearchJob)
        .outerjoin(attempt_seconds, attempt_seconds.c.job_id == ResearchJob.id)
        .where(
            in_window,
            ResearchJob.status.in_([JobStatus.COMPLETED, JobStatus.FAILED]),
        )
    )

    completed = by_status[JobStatus.COMPLETED.value]
    failed = by_status[JobStatus.FAILED.value]

    return {
        "total": sum(by_status.values()),
        "by_status": by_status,
        "total_attempts": total_attempts,
        "retry_count": max(total_attempts - started, 0),
        # Started jobs that needed more than one attempt.
        "retry_rate": _percent(retried, started),
        "average_duration_ms": (
            _round(average_duration_seconds * 1000)
            if average_duration_seconds is not None
            else None
        ),
        # Of the jobs that finished.
        "success_rate": _percent(completed, completed + failed),
    }


async def agent_stats(db: AsyncSession, since: datetime | None) -> dict:
    """Agent runs started since `since` (all runs if None)."""

    in_window = _in_window(AgentRun.started_at, since)

    by_status = {
        status: count
        for status, count in (
            await db.execute(
                select(AgentRun.status, func.count())
                .where(in_window)
                .group_by(AgentRun.status)
            )
        ).all()
    }

    row = (
        await db.execute(
            select(
                func.count(),
                func.coalesce(func.sum(AgentRun.input_tokens), 0),
                func.coalesce(func.sum(AgentRun.output_tokens), 0),
                func.coalesce(func.sum(AgentRun.total_tokens), 0),
                func.coalesce(func.sum(AgentRun.estimated_cost_usd), 0.0),
                func.count().filter(AgentRun.estimated_cost_usd.is_(None)),
                func.avg(AgentRun.iteration_count),
                func.avg(AgentRun.tool_call_count),
                func.avg(_seconds(AgentRun.started_at, AgentRun.completed_at)),
            ).where(in_window)
        )
    ).one()

    (
        runs,
        input_tokens,
        output_tokens,
        total_tokens,
        cost,
        unpriced_runs,
        average_iterations,
        average_tool_calls,
        average_duration_seconds,
    ) = row

    return {
        "total_runs": runs,
        "by_status": by_status,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        # USD, over the runs with known pricing; unpriced_runs says how many
        # are left out (so the true total is higher if it's above 0).
        "estimated_cost_usd": round(float(cost), 6),
        "unpriced_runs": unpriced_runs,
        "average_iterations": _round(average_iterations),
        "average_tool_calls": _round(average_tool_calls),
        # Finished runs; a resumed run includes the wait between attempts.
        "average_duration_ms": (
            _round(average_duration_seconds * 1000)
            if average_duration_seconds is not None
            else None
        ),
    }


def finished_tool_calls(
    since: datetime | None,
    until: datetime | None = None,
):
    """One row per finished tool call (a tool_completed or tool_failed job
    event): tool, success, duration_ms, read from metadata_json in SQL.

    A call counts as failed if its metadata says success: false, whatever
    the event type; without that key, tool_failed means failed. Events whose
    metadata isn't valid JSON still count (as calls with no tool or
    duration) instead of breaking the query.
    """

    metadata = type_coerce(
        case(
            (
                func.pg_input_is_valid(JobEvent.metadata_json, "jsonb"),
                cast(JobEvent.metadata_json, JSONB),
            ),
            else_=null(),
        ),
        JSONB,
    )

    return (
        select(
            metadata["tool"].astext.label("tool"),
            func.coalesce(
                metadata["success"].as_boolean(),
                JobEvent.event_type == JobEventType.TOOL_COMPLETED.value,
            ).label("success"),
            metadata["duration_ms"].as_float().label("duration_ms"),
        )
        .where(
            JobEvent.event_type.in_(
                [
                    JobEventType.TOOL_COMPLETED.value,
                    JobEventType.TOOL_FAILED.value,
                ]
            ),
            _in_window(JobEvent.created_at, since),
            JobEvent.created_at <= until if until is not None else true(),
        )
        .subquery()
    )


def _tool_aggregates(calls):
    return (
        func.count(),
        func.count().filter(calls.c.success.is_(True)),
        func.count().filter(calls.c.success.is_not(True)),
        func.avg(calls.c.duration_ms),
        func.percentile_cont(0.95).within_group(calls.c.duration_ms.asc()),
    )


def _tool_row(calls, succeeded, failed, average_ms, p95_ms) -> dict:
    return {
        "calls": calls,
        "succeeded": succeeded,
        "failed": failed,
        "success_rate": _percent(succeeded, calls),
        "average_latency_ms": _round(average_ms),
        "p95_latency_ms": _round(p95_ms),
    }


async def tool_stats(db: AsyncSession, since: datetime | None) -> dict:
    """Finished tool calls (from job events) since `since` (all if None),
    overall and per tool."""

    calls = finished_tool_calls(since)

    overall = (
        await db.execute(select(*_tool_aggregates(calls)))
    ).one()

    per_tool = (
        await db.execute(
            select(calls.c.tool, *_tool_aggregates(calls))
            .where(calls.c.tool.is_not(None))
            .group_by(calls.c.tool)
            .order_by(calls.c.tool)
        )
    ).all()

    return {
        **_tool_row(*overall),
        "by_tool": {
            tool: _tool_row(*aggregates)
            for tool, *aggregates in per_tool
        },
    }


async def running_jobs(db: AsyncSession) -> list[dict]:
    """Jobs RUNNING now, longest-running first."""

    now = utc_now()

    jobs = (
        await db.execute(
            select(ResearchJob)
            .where(ResearchJob.status == JobStatus.RUNNING)
            .order_by(ResearchJob.started_at.asc(), ResearchJob.id.asc())
        )
    ).scalars().all()

    return [
        {
            "job_id": job.id,
            "task_id": job.task_id,
            "worker_id": job.worker_id,
            "attempt": job.attempts,
            "started_at": job.started_at,
            "running_for_seconds": (
                round((now - job.started_at).total_seconds(), 1)
                if job.started_at is not None
                else None
            ),
            "lease_expires_at": job.lease_expires_at,
            # Its worker probably died; recovered on the next poll.
            "lease_expired": (
                job.lease_expires_at is None
                or job.lease_expires_at < now
            ),
        }
        for job in jobs
    ]


async def build_system_metrics(
    db: AsyncSession,
    since: datetime | None = None,
) -> dict:
    """
    Build aggregate metrics across the entire research system.

    All time by default; with `since`, jobs created, agent runs started and
    tool calls made since then.
    """

    jobs = await job_stats(db, since)
    agents = await agent_stats(db, since)
    tools = await tool_stats(db, since)

    by_status = jobs["by_status"]

    total_jobs = jobs["total"]
    completed_jobs = by_status[JobStatus.COMPLETED.value]

    # Completed jobs as a percentage of all jobs (as in
    # metrics_service.build_research_metrics).
    if total_jobs:
        success_rate = round(
            completed_jobs
            / total_jobs
            * 100,
            2,
        )
    else:
        success_rate = 0.0

    return {
        "total_jobs": total_jobs,

        "pending_jobs": by_status[JobStatus.PENDING.value],
        "running_jobs": by_status[JobStatus.RUNNING.value],
        "completed_jobs": completed_jobs,
        "failed_jobs": by_status[JobStatus.FAILED.value],

        "total_attempts": jobs["total_attempts"],
        "retry_count": jobs["retry_count"],

        # Finished jobs, over all their attempts.
        "average_job_duration_ms": jobs["average_duration_ms"],

        # Finished tool calls, successful or failed.
        "total_tool_calls": tools["calls"],
        "successful_tool_calls": tools["succeeded"],
        "failed_tool_calls": tools["failed"],

        "average_tool_latency_ms": tools["average_latency_ms"],

        "total_agent_runs": agents["total_runs"],

        "total_input_tokens": agents["input_tokens"],
        "total_output_tokens": agents["output_tokens"],
        "total_tokens": agents["total_tokens"],

        # USD, over runs with known pricing (see agent_stats).
        "estimated_cost": agents["estimated_cost_usd"],

        "success_rate": success_rate,
    }


async def build_system_monitoring(
    db: AsyncSession,
    since: datetime,
) -> dict:
    """Everything GET /metrics returns. The summary and the job, agent,
    tool and worker stats cover `since` until now; the queue and running
    jobs are the current state."""

    queue = await get_queue_health(db)

    return {
        "since": since,
        "generated_at": utc_now(),
        "summary": await build_system_metrics(db, since),
        "jobs": await job_stats(db, since),
        "agents": await agent_stats(db, since),
        "tools": await tool_stats(db, since),
        "workers": await get_worker_activity(db=db, since=since),
        "running_jobs": await running_jobs(db),
        "queue": {
            "pending_jobs": queue["jobs"][JobStatus.PENDING.value],
            "running_jobs": queue["jobs"][JobStatus.RUNNING.value],
            "oldest_pending_seconds": queue["oldest_pending_seconds"],
            "expired_leases": queue["expired_leases"],
        },
    }
