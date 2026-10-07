import json
import logging


from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.research_agent import (
    AgentCheckpoint,
    AgentPhase,
    ResearchAgent,
    ToolExecution,
    format_cost,
)
from app.core.config import settings
from app.db.models import (
    AgentRun,
    AgentRunStatus,
    ResearchStep,
    ResearchTask,
    StepStatus,
    TaskStatus,
)
from app.llm.provider import OpenAIProvider
from app.services.agent_run_service import (
    clear_agent_checkpoint,
    complete_agent_run,
    create_agent_run,
    fail_agent_run,
    load_agent_checkpoint,
    reopen_agent_run,
    save_agent_checkpoint,
    totals_from_checkpoint,
    update_agent_run_status,
)
from app.services.step_service import create_agent_step
from app.services.result_service import save_tool_result
from app.services.state_machine import (
    can_transition_agent_run,
    can_transition_task,
    is_resumable,
    transition_task,
)
from app.jobs.events import record_tool_event
from app.jobs.models import JobEventType
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


async def persist_agent_step(
    db: AsyncSession,
    task_id: int,
    tool: str,
    query: str,
    iteration: int,
) -> ResearchStep:

    step = await create_agent_step(
        db=db,
        task_id=task_id,
        tool=tool,
        query=query,
        iteration=iteration,
    )

    await db.commit()

    return step


async def persist_agent_result(
    db: AsyncSession,
    step: ResearchStep,
    execution: ToolExecution,
) -> None:

    result = execution.result

    step.duration_ms = execution.duration_ms

    await save_tool_result(
        db=db,
        step=step,
        result=result,
    )

    step.status = (
        StepStatus.COMPLETED
        if result.success
        else StepStatus.FAILED
    )

    await db.commit()


async def persist_agent_checkpoint(
    db: AsyncSession,
    run: AgentRun,
    checkpoint: str,
) -> None:

    # The agent's _serialize_checkpoint() gives a JSON string; the JSONB
    # column stores the parsed structure, not the string.
    await save_agent_checkpoint(
        db=db,
        run=run,
        state=json.loads(checkpoint),
    )

    # Committed straight away: the point is to survive a crash.
    await db.commit()


async def persist_agent_status(
    db: AsyncSession,
    run: AgentRun,
    phase: AgentPhase,
) -> None:

    await update_agent_run_status(
        db=db,
        run=run,
        status=AgentRunStatus(phase.value),
    )

    # Committed straight away, so the run's current phase is visible (and
    # known after a crash).
    await db.commit()


async def execute_research_workflow(
    db: AsyncSession,
    task: ResearchTask,
    job_id: int | None = None,
) -> ResearchTask:
    """Runs the agent for a new task. With a job_id (the worker passes its
    job's), each search is also recorded in the job's events."""

    logger.info(
        "Starting research workflow task_id=%s",
        task.id,
    )

    return await _run_research(
        db=db,
        task=task,
        job_id=job_id,
    )


async def check_can_resume(
    db: AsyncSession,
    task: ResearchTask,
    agent_run: AgentRun | None,
) -> AgentCheckpoint:
    """Returns the checkpoint to resume from, or raises ValueError saying
    why the task can't be resumed. Changes nothing."""

    # Reload: the checks below must see the current database status (not a
    # stale copy), and reading an expired attribute would otherwise fail
    # with MissingGreenlet on an async session.
    await db.refresh(task)

    # The task first: for a task that isn't failed, that's the reason,
    # whatever its run looks like.
    if not is_resumable(task):
        raise ValueError(
            f"Research task {task.id} is {TaskStatus(task.status).value}; "
            "only failed tasks can be resumed."
        )

    if agent_run is None:
        raise ValueError(
            f"Research task {task.id} has no agent run to resume."
        )

    await db.refresh(agent_run)

    if agent_run.task_id != task.id:
        raise ValueError(
            f"Agent run {agent_run.id} belongs to task {agent_run.task_id}, "
            f"not task {task.id}."
        )

    if AgentRunStatus(agent_run.status) != AgentRunStatus.FAILED:
        raise ValueError(
            f"Agent run {agent_run.id} is "
            f"{AgentRunStatus(agent_run.status).value}; only failed runs "
            "can be resumed."
        )

    checkpoint, _ = (
        await load_agent_checkpoint(
            db=db,
            run=agent_run,
        )
    )

    if checkpoint is None:
        raise ValueError(
            "Agent run has no checkpoint to resume."
        )

    return checkpoint


async def resume_research_workflow(
    db: AsyncSession,
    task: ResearchTask,
    agent_run: AgentRun,
    job_id: int | None = None,
) -> ResearchTask:
    """Resumes a failed run in place, from its checkpoint. job_id: as for
    execute_research_workflow."""

    checkpoint = await check_can_resume(
        db=db,
        task=task,
        agent_run=agent_run,
    )

    last_iteration = checkpoint.iteration

    logger.info(
        "Resuming agent run_id=%s from iteration=%s",
        agent_run.id,
        last_iteration,
    )

    return await _run_research(
        db=db,
        task=task,
        checkpoint=checkpoint,
        agent_run=agent_run,
        job_id=job_id,
    )


async def _run_research(
    db: AsyncSession,
    task: ResearchTask,
    checkpoint: AgentCheckpoint | None = None,
    agent_run: AgentRun | None = None,
    job_id: int | None = None,
) -> ResearchTask:
    """Runs the agent for a new task, or resumes a failed run in place
    from its checkpoint (pass both checkpoint and agent_run)."""

    # Read up front: after a rollback, task attributes are expired and
    # can't be loaded lazily with an async session.
    task_id = task.id

    # Set once the agent finishes, so a failure after that point can still
    # save the run's counts and token usage.
    agent_result = None

    try:

        # --------------------------------------------------
        # 1. Planning / agent startup
        # --------------------------------------------------

        # A resume skips planning: the task goes FAILED -> RESEARCHING.
        if agent_run is None:

            transition_task(task, TaskStatus.PLANNING)

            await db.commit()

        logger.info(
            "Starting research agent task_id=%s",
            task.id,
        )

        llm = OpenAIProvider()

        registry = ToolRegistry()

        # The job's events for each search: tool_started before it runs,
        # then tool_completed or tool_failed. Only for a run with a job.
        async def on_tool_call(
            tool_name: str,
            query: str,
        ):
            await record_tool_event(
                db=db,
                job_id=job_id,
                event_type=JobEventType.TOOL_STARTED,
                tool=tool_name,
                query=query,
            )

            await db.commit()

        async def on_tool_result(
            tool_name: str,
            query: str,
            success: bool,
            duration_ms: float,
            error: str | None = None,
        ):
            await record_tool_event(
                db=db,
                job_id=job_id,
                event_type=(
                    JobEventType.TOOL_COMPLETED
                    if success
                    else JobEventType.TOOL_FAILED
                ),
                tool=tool_name,
                query=query,
                success=success,
                duration_ms=duration_ms,
                error=error,
            )

            await db.commit()

        # What the agent calls: save the step (as before), then record the
        # job event.
        async def start_tool_step(tool: str, query: str, iteration: int):

            step = await persist_agent_step(
                db=db,
                task_id=task.id,
                tool=tool,
                query=query,
                iteration=iteration,
            )

            if job_id is not None:
                await on_tool_call(tool, query)

            return step

        async def finish_tool_step(step: ResearchStep, execution: ToolExecution):

            await persist_agent_result(
                db=db,
                step=step,
                execution=execution,
            )

            if job_id is not None:
                await on_tool_result(
                    tool_name=step.tool,
                    query=step.query,
                    success=execution.result.success,
                    duration_ms=execution.duration_ms,
                    error=execution.result.error,
                )

        # The lambdas look agent_run up when they run (after it's set
        # below), not when they are defined.
        agent = ResearchAgent(
            llm=llm,
            registry=registry,

            on_tool_call=start_tool_step,

            on_tool_result=finish_tool_step,

            on_checkpoint=lambda checkpoint: (
                persist_agent_checkpoint(
                    db=db,
                    run=agent_run,
                    checkpoint=checkpoint,
                )
            ),

            on_status_change=lambda phase: (
                persist_agent_status(
                    db=db,
                    run=agent_run,
                    phase=phase,
                )
            ),

            # Limits from the AGENT_* environment variables.
            max_iterations=settings.agent_max_iterations,
            max_tool_calls=settings.agent_max_tool_calls,
            tool_timeout_seconds=settings.agent_tool_timeout_seconds,
            max_tool_retries=settings.agent_max_tool_retries,
            max_query_length=settings.agent_max_query_length,
            max_result_length=settings.agent_max_result_length,
            max_total_tokens=settings.agent_max_total_tokens,
        )

        # --------------------------------------------------
        # 2. Research
        # --------------------------------------------------

        transition_task(task, TaskStatus.RESEARCHING)

        if agent_run is None:

            # CREATED; the agent moves it to RUNNING when it starts.
            agent_run = await create_agent_run(
                db=db,
                task_id=task.id,
            )

        else:

            # Same run, FAILED -> RUNNING; its checkpoint stays.
            await reopen_agent_run(
                db=db,
                run=agent_run,
            )

        await db.commit()

        agent_result = await agent.run(
            question=task.question,
            checkpoint=checkpoint,
        )

        # The result covers the whole job (a resumed agent carries on from
        # the checkpoint's counts and usage), which is what the run records.
        # Saved whether the run succeeded or not, so failed and over-budget
        # runs still record what they used.
        if agent_result.success:

            await complete_agent_run(
                db=db,
                run=agent_run,
                iteration_count=agent_result.iteration_count,
                tool_call_count=agent_result.tool_call_count,
                input_tokens=agent_result.usage.input_tokens,
                output_tokens=agent_result.usage.output_tokens,
                total_tokens=agent_result.usage.total_tokens,
                estimated_cost_usd=agent_result.usage.estimated_cost,
            )

            # Nothing left to resume.
            await clear_agent_checkpoint(
                db=db,
                run=agent_run,
            )

        else:

            # The checkpoint stays, so the run can be resumed.
            await fail_agent_run(
                db=db,
                run=agent_run,
                error=agent_result.error or "Agent run failed.",
                iteration_count=agent_result.iteration_count,
                tool_call_count=agent_result.tool_call_count,
                input_tokens=agent_result.usage.input_tokens,
                output_tokens=agent_result.usage.output_tokens,
                total_tokens=agent_result.usage.total_tokens,
                estimated_cost_usd=agent_result.usage.estimated_cost,
            )

        logger.info(
            "Agent run finished task_id=%s status=%s tokens=%d cost=%s",
            task.id,
            agent_run.status,
            agent_run.total_tokens,
            format_cost(agent_result.usage.estimated_cost),
        )

        # --------------------------------------------------
        # 3. Final answer
        # --------------------------------------------------

        if not agent_result.answer:

            transition_task(task, TaskStatus.FAILED)

            await db.commit()

            logger.error(
                "Agent returned empty answer task_id=%s",
                task.id,
            )

            return task

        transition_task(task, TaskStatus.SYNTHESIZING)

        await db.commit()

        task.summary = agent_result.answer

        transition_task(task, TaskStatus.COMPLETED)

        await db.commit()

        await db.refresh(task)

        logger.info(
            "Research workflow completed task_id=%s",
            task.id,
        )

        return task

    except Exception as exc:

        logger.exception(
            "Research workflow failed task_id=%s",
            task_id,
        )

        await db.rollback()

        # The rollback expired task and agent_run; reload them so their
        # current status can be read (no lazy loading with async sessions).
        await db.refresh(task)

        if (
            agent_run is not None
            and inspect(agent_run).persistent
        ):
            await db.refresh(agent_run)

        # Only a run that reached the database and hasn't finished is marked
        # failed: one that already completed stays completed.
        if (
            agent_run is not None
            and inspect(agent_run).persistent
            and can_transition_agent_run(agent_run, AgentRunStatus.FAILED)
        ):
            if agent_result is not None:
                iteration_count = agent_result.iteration_count
                tool_call_count = agent_result.tool_call_count
                usage = agent_result.usage
            else:
                iteration_count, tool_call_count, usage = (
                    totals_from_checkpoint(agent_run)
                )

            await fail_agent_run(
                db=db,
                run=agent_run,
                error=str(exc),
                iteration_count=iteration_count,
                tool_call_count=tool_call_count,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
                estimated_cost_usd=usage.estimated_cost,
            )

        # E.g. a task that already reached COMPLETED stays completed.
        if can_transition_task(task, TaskStatus.FAILED):
            transition_task(task, TaskStatus.FAILED)

        await db.commit()

        return task
