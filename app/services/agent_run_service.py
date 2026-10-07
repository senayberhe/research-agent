from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.research_agent import AgentCheckpoint, TokenUsage
from app.db.models import (
    AgentRun,
    AgentRunStatus,
    ResearchTask,
    TaskStatus,
    utc_now,
)
from app.services.state_machine import (
    can_transition_task,
    transition_agent_run,
    transition_task,
)


async def create_agent_run(
    db: AsyncSession,
    task_id: int,
) -> AgentRun:

    run = AgentRun(
        task_id=task_id,
        status=AgentRunStatus.CREATED,
    )

    db.add(run)

    await db.flush()

    return run


# Statuses of a run that hasn't finished. At startup, a run in one of these
# was cut off by a crash or restart (the app runs a single worker, so no
# other process can still be working on it).
UNFINISHED_STATUSES = (
    AgentRunStatus.CREATED,
    AgentRunStatus.RUNNING,
    AgentRunStatus.WAITING_FOR_TOOL,
    AgentRunStatus.PROCESSING_RESULT,
)


async def get_latest_agent_run(
    db: AsyncSession,
    task_id: int,
) -> AgentRun | None:

    return await db.scalar(
        select(AgentRun)
        .where(AgentRun.task_id == task_id)
        .order_by(AgentRun.id.desc())
        .limit(1)
    )


async def find_unfinished_agent_runs(
    db: AsyncSession,
) -> list[AgentRun]:

    result = await db.execute(
        select(AgentRun)
        .where(AgentRun.status.in_(UNFINISHED_STATUSES))
        .order_by(AgentRun.id)
    )

    return list(result.scalars().all())


def totals_from_checkpoint(run: AgentRun) -> tuple[int, int, TokenUsage]:
    """Iterations, tool calls and usage for a run that stopped without an
    AgentResult (a crash or restart): its latest checkpoint, which is where
    it got to; without one, what the run already had recorded."""

    if run.state is not None:

        checkpoint = AgentCheckpoint.from_dict(run.state)

        return (
            checkpoint.iteration,
            checkpoint.tool_call_count,
            checkpoint.usage,
        )

    return (
        run.iteration_count,
        run.tool_call_count,
        TokenUsage(
            input_tokens=run.input_tokens,
            output_tokens=run.output_tokens,
            total_tokens=run.total_tokens,
            estimated_cost=run.estimated_cost_usd,
        ),
    )


async def update_agent_run_status(
    db: AsyncSession,
    run: AgentRun,
    status: AgentRunStatus,
) -> AgentRun:

    # Already there (e.g. a reopened run that the agent then reports as
    # RUNNING): nothing to change.
    if AgentRunStatus(run.status) != status:
        transition_agent_run(run, status)

    await db.flush()

    return run


async def reopen_agent_run(
    db: AsyncSession,
    run: AgentRun,
) -> AgentRun:
    """Puts a FAILED run back to RUNNING so it can be resumed in place.

    Its checkpoint, counts and tokens stay; the error and completion time of
    the failed attempt are cleared.
    """

    transition_agent_run(run, AgentRunStatus.RUNNING)

    run.error = None
    run.completed_at = None

    await db.flush()

    return run


async def clear_agent_checkpoint(
    db: AsyncSession,
    run: AgentRun,
) -> AgentRun:

    # A completed run has nothing left to resume.
    run.state = None
    run.state_updated_at = None

    await db.flush()

    return run


async def save_agent_checkpoint(
    db: AsyncSession,
    run: AgentRun,
    state: dict[str, Any],
) -> AgentRun:

    # A new dict each time, so SQLAlchemy sees the change (it doesn't track
    # edits inside a JSONB value).
    run.state = state
    run.state_updated_at = utc_now()

    await db.flush()

    return run


async def load_agent_checkpoint(
    db: AsyncSession,
    run: AgentRun,
) -> tuple[AgentCheckpoint | None, int]:
    """The run's latest checkpoint and the last iteration it finished, or
    (None, 0) if it has none (never finished an iteration, or completed)."""

    # Read state from the database, not from what this session last saw.
    await db.refresh(run, attribute_names=["state"])

    if run.state is None:
        return None, 0

    checkpoint = AgentCheckpoint.from_dict(run.state)

    return checkpoint, checkpoint.iteration


async def complete_agent_run(
    db: AsyncSession,
    run: AgentRun,
    iteration_count: int,
    tool_call_count: int,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    estimated_cost_usd: float | None,
) -> AgentRun:

    transition_agent_run(run, AgentRunStatus.COMPLETED)

    run.iteration_count = iteration_count
    run.tool_call_count = tool_call_count

    run.input_tokens = input_tokens
    run.output_tokens = output_tokens
    run.total_tokens = total_tokens

    # None when the model's pricing is unknown.
    run.estimated_cost_usd = estimated_cost_usd

    run.completed_at = utc_now()

    await db.flush()

    return run


async def fail_agent_run(
    db: AsyncSession,
    run: AgentRun,
    error: str,
    iteration_count: int,
    tool_call_count: int,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    estimated_cost_usd: float | None,
) -> AgentRun:

    transition_agent_run(run, AgentRunStatus.FAILED)

    run.error = error

    run.iteration_count = iteration_count
    run.tool_call_count = tool_call_count

    run.input_tokens = input_tokens
    run.output_tokens = output_tokens
    run.total_tokens = total_tokens

    # None when the model's pricing is unknown.
    run.estimated_cost_usd = estimated_cost_usd

    run.completed_at = utc_now()

    await db.flush()

    return run


async def fail_unfinished_runs_for_task(
    db: AsyncSession,
    task_id: int,
    error: str,
) -> list[AgentRun]:
    """Marks the task's unfinished runs FAILED (totals from their
    checkpoint, which is kept) and the task FAILED, so the task can be
    resumed. For a task whose worker is known to be gone. Flushes."""

    result = await db.execute(
        select(AgentRun)
        .where(
            AgentRun.task_id == task_id,
            AgentRun.status.in_(UNFINISHED_STATUSES),
        )
        .order_by(AgentRun.id)
    )

    runs = list(result.scalars().all())

    for run in runs:

        iteration_count, tool_call_count, usage = totals_from_checkpoint(run)

        await fail_agent_run(
            db=db,
            run=run,
            error=error,
            iteration_count=iteration_count,
            tool_call_count=tool_call_count,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            estimated_cost_usd=usage.estimated_cost,
        )

    task = await db.get(ResearchTask, task_id)

    if (
        runs
        and task is not None
        and can_transition_task(task, TaskStatus.FAILED)
    ):
        transition_task(task, TaskStatus.FAILED)

    await db.flush()

    return runs


async def list_agent_runs(
    db: AsyncSession,
    task_id: int,
) -> list[AgentRun]:

    result = await db.execute(
        select(AgentRun)
        .where(AgentRun.task_id == task_id)
        .order_by(AgentRun.id)
    )

    return list(result.scalars().all())
