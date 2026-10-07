from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    AgentRun,
    JobEventType,
    JobStatus,
)
from app.jobs.event_service import (
    get_job_events,
    parse_event_metadata,
)
from app.jobs.service import get_jobs_for_task, list_job_attempts


# A search that finished, either way (tool_started doesn't count: the
# search may still be running).
FINISHED_TOOL_EVENTS = (
    JobEventType.TOOL_COMPLETED,
    JobEventType.TOOL_FAILED,
)


async def build_research_metrics(
    db: AsyncSession,
    task_id: int,
) -> dict:
    """
    Build aggregate execution metrics for a research task.

    Metrics are calculated from:
    - ResearchJob (jobs, statuses, attempts)
    - JobAttempt (execution time, over every attempt; the job's own
      times for jobs without attempt records)
    - JobEvent (tool calls and their latency)
    - AgentRun (tokens and cost)
    """

    jobs = await get_jobs_for_task(
        db=db,
        task_id=task_id,
    )

    total_jobs = len(jobs)

    completed_jobs = sum(
        1
        for job in jobs
        if job.status == JobStatus.COMPLETED
    )

    failed_jobs = sum(
        1
        for job in jobs
        if job.status == JobStatus.FAILED
    )

    total_attempts = sum(
        job.attempts
        for job in jobs
    )

    retry_count = sum(
        max(job.attempts - 1, 0)
        for job in jobs
    )

    # Every finished attempt, not just each job's last one: started_at is
    # reset when a job is retried. A job without attempt records (run
    # before they existed) counts its own start to completion.
    total_execution_time_ms = 0.0

    for job in jobs:

        attempts = await list_job_attempts(db=db, job_id=job.id)

        if attempts:
            spans = [
                (attempt.started_at, attempt.ended_at)
                for attempt in attempts
            ]
        else:
            spans = [(job.started_at, job.completed_at)]

        for started_at, ended_at in spans:
            if started_at and ended_at:
                duration = (
                    ended_at - started_at
                ).total_seconds() * 1000

                total_execution_time_ms += duration

    if total_execution_time_ms == 0:
        total_execution_time_ms = None

    total_tool_calls = 0
    successful_tool_calls = 0
    failed_tool_calls = 0

    tool_latencies: list[float] = []

    for job in jobs:
        events = await get_job_events(
            db=db,
            job_id=job.id,
        )

        for event in events:
            if event.event_type not in FINISHED_TOOL_EVENTS:
                continue

            total_tool_calls += 1

            metadata = parse_event_metadata(event)

            # tool_failed events also carry success: false; the event type
            # decides if it's missing.
            success = metadata.get(
                "success",
                event.event_type == JobEventType.TOOL_COMPLETED,
            )

            if success:
                successful_tool_calls += 1
            else:
                failed_tool_calls += 1

            duration_ms = metadata.get("duration_ms")

            if duration_ms is not None:
                tool_latencies.append(
                    float(duration_ms)
                )

    if tool_latencies:
        average_tool_latency_ms = (
            sum(tool_latencies)
            / len(tool_latencies)
        )
    else:
        average_tool_latency_ms = None

    result = await db.execute(
        select(AgentRun)
        .where(AgentRun.task_id == task_id)
    )

    agent_runs = list(
        result.scalars().all()
    )

    total_input_tokens = sum(
        run.input_tokens
        for run in agent_runs
    )

    total_output_tokens = sum(
        run.output_tokens
        for run in agent_runs
    )

    total_tokens = sum(
        run.total_tokens
        for run in agent_runs
    )

    # None if any run's cost is unknown: a total without it would be too
    # low.
    if any(run.estimated_cost_usd is None for run in agent_runs):
        estimated_cost = None
    else:
        estimated_cost = sum(
            run.estimated_cost_usd
            for run in agent_runs
        )

    if total_jobs:
        success_rate = (
            completed_jobs
            / total_jobs
            * 100
        )
    else:
        success_rate = 0.0

    return {
        "task_id": task_id,
        "total_jobs": total_jobs,
        "completed_jobs": completed_jobs,
        "failed_jobs": failed_jobs,
        "total_attempts": total_attempts,
        "retry_count": retry_count,
        "total_tool_calls": total_tool_calls,
        "successful_tool_calls": successful_tool_calls,
        "failed_tool_calls": failed_tool_calls,
        "average_tool_latency_ms": average_tool_latency_ms,
        "total_execution_time_ms": total_execution_time_ms,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "estimated_cost": estimated_cost,
        "success_rate": success_rate,
    }
