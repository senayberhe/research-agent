"""The unified execution timeline of a research task: its jobs' events, its
research steps and its agent runs, merged into one list ordered by time
(the items of ExecutionTimelineResponse)."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    AgentRun,
    AgentRunStatus,
    ResearchStep,
)
from app.jobs.event_service import (
    get_job_events,
    parse_event_metadata,
)
from app.jobs.execution_service import summarize_checkpoint
from app.jobs.service import get_jobs_for_task


async def build_execution_timeline(
    db: AsyncSession,
    task_id: int,
) -> list[dict]:
    """Oldest first. Sources: "job" (job_events: the queue, workers and
    each search's start and end), "step" (research_steps) and "agent"
    (agent_runs: when each run started and ended, with its totals)."""

    timeline: list[dict] = []

    jobs = await get_jobs_for_task(
        db=db,
        task_id=task_id,
    )

    for job in jobs:

        events = await get_job_events(
            db=db,
            job_id=job.id,
        )

        for event in events:

            timeline.append(
                {
                    "timestamp": event.created_at,
                    "event_type": event.event_type,
                    "source": "job",
                    "message": event.message,
                    "metadata": parse_event_metadata(
                        event
                    ),
                }
            )

    result = await db.execute(
        select(ResearchStep)
        .where(
            ResearchStep.task_id == task_id
        )
        .order_by(
            ResearchStep.id.asc()
        )
    )

    steps = result.scalars().all()

    for step in steps:

        timeline.append(
            {
                "timestamp": step.created_at,
                "event_type": "research_step",
                "source": "step",
                "message": (
                    f"{step.tool} research step"
                ),
                "metadata": {
                    "step_id": step.id,
                    "tool": step.tool,
                    "query": step.query,
                    "iteration": step.iteration,
                    "status": step.status,
                    "duration_ms": step.duration_ms,
                },
            }
        )

    result = await db.execute(
        select(AgentRun)
        .where(
            AgentRun.task_id == task_id
        )
        .order_by(
            AgentRun.started_at.asc(),
            AgentRun.id.asc(),
        )
    )

    runs = result.scalars().all()

    for run in runs:

        # A summary of run.state (the checkpoint), not the checkpoint
        # itself: that holds the whole conversation.
        checkpoint = summarize_checkpoint(run)

        timeline.append(
            {
                "timestamp": run.started_at,
                "event_type": "agent_started",
                "source": "agent",
                "message": (
                    f"Agent run {run.id} started."
                ),
                "metadata": {
                    "run_id": run.id,
                    "state": (
                        {
                            "iteration": checkpoint.iteration,
                            "conversation_items": (
                                checkpoint.conversation_items
                            ),
                            "valid": checkpoint.valid,
                        }
                        if checkpoint is not None
                        else None
                    ),
                    "status": run.status,
                },
            }
        )

        if run.completed_at:

            # Failed runs have a completed_at too.
            failed = AgentRunStatus(run.status) == AgentRunStatus.FAILED

            timeline.append(
                {
                    "timestamp": run.completed_at,
                    "event_type": (
                        "agent_failed"
                        if failed
                        else "agent_completed"
                    ),
                    "source": "agent",
                    "message": (
                        f"Agent run {run.id} failed: {run.error}"
                        if failed
                        else f"Agent run {run.id} completed."
                    ),
                    "metadata": {
                        "run_id": run.id,
                        "status": run.status,
                        "iterations": (
                            run.iteration_count
                        ),
                        "tool_calls": (
                            run.tool_call_count
                        ),
                        "total_tokens": (
                            run.total_tokens
                        ),
                        "estimated_cost": (
                            run.estimated_cost_usd
                        ),
                    },
                }
            )

    # Stable: entries with the same timestamp keep the order above (job
    # events, then steps, then runs).
    timeline.sort(
        key=lambda event: event["timestamp"]
    )

    return timeline
