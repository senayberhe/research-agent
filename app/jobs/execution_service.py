"""How a job's AI execution behaved, in one place: the job's current state,
its history of events, and its agent run (iterations, tool calls, tokens,
cost, checkpoint)."""

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.research_agent import AgentCheckpoint
from app.db.models import AgentRun, JobEvent, ResearchJob, ResearchStep
from app.jobs.event_service import get_job_events
from app.jobs.service import get_research_job
from app.services.agent_run_service import get_latest_agent_run
from app.services.step_service import list_task_steps


@dataclass
class CheckpointSummary:
    """The run's saved checkpoint, without the conversation itself (it can
    be long)."""

    saved_at: object
    iteration: int | None
    executed_tool_calls: int | None
    conversation_items: int | None
    # False if the stored checkpoint is corrupt; error says why.
    valid: bool
    error: str | None


@dataclass
class JobExecution:
    job: ResearchJob
    events: list[JobEvent]
    # The task's latest run: a resumed task continues the same run, so this
    # is the run the job executed (None if the job never started one).
    agent_run: AgentRun | None
    checkpoint: CheckpointSummary | None
    steps: list[ResearchStep]


def summarize_checkpoint(run: AgentRun) -> CheckpointSummary | None:

    if run.state is None:
        return None

    try:
        checkpoint = AgentCheckpoint.from_dict(run.state)
    except ValueError as exc:
        return CheckpointSummary(
            saved_at=run.state_updated_at,
            iteration=None,
            executed_tool_calls=None,
            conversation_items=None,
            valid=False,
            error=str(exc),
        )

    return CheckpointSummary(
        saved_at=run.state_updated_at,
        iteration=checkpoint.iteration,
        executed_tool_calls=checkpoint.executed_tool_calls,
        conversation_items=len(checkpoint.input_items),
        valid=True,
        error=None,
    )


async def get_job_execution(
    db: AsyncSession,
    job_id: int,
) -> JobExecution | None:
    """None if there is no such job."""

    job = await get_research_job(db=db, job_id=job_id)

    if job is None:
        return None

    run = await get_latest_agent_run(db=db, task_id=job.task_id)

    return JobExecution(
        job=job,
        events=await get_job_events(db=db, job_id=job_id),
        agent_run=run,
        checkpoint=summarize_checkpoint(run) if run is not None else None,
        steps=await list_task_steps(db=db, task_id=job.task_id),
    )
