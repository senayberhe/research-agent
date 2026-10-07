"""A research task's progress, for following it live: the task (and its
summary once done), its latest job, its agent run with live totals, each
tool call (query, status, duration, error, a preview of what it
returned), and everything that has happened so far, in one snapshot.

While the agent runs, its iteration, tool-call, token and cost totals are
only in its checkpoint (saved after every iteration); the run's own columns
are written when it ends. So an unfinished run reports its checkpoint's
totals.
"""

from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AgentRun, ResearchStep, TaskStatus, utc_now
from app.jobs.service import get_latest_job_for_task
from app.services.agent_run_service import (
    UNFINISHED_STATUSES,
    get_latest_agent_run,
    totals_from_checkpoint,
)
from app.services.research_service import get_research_task
from app.services.step_service import list_task_steps
from app.services.timeline_service import build_execution_timeline
from app.services.workflow_service import check_can_resume


# The task won't change any more (unless resumed).
FINISHED_TASK_STATUSES = (TaskStatus.COMPLETED, TaskStatus.FAILED)

# How much of a tool's result to include: enough to see what came back,
# without sending whole documents every 2 seconds.
RESULT_PREVIEW_CHARS = 1200


def _tool_call(step: ResearchStep) -> dict:
    """One tool call: a research step and its result (once it has one)."""

    result = step.results[0] if step.results else None

    preview = None
    truncated = False

    if result is not None and result.content:
        preview = result.content[:RESULT_PREVIEW_CHARS]
        truncated = len(result.content) > RESULT_PREVIEW_CHARS

    return {
        "id": step.id,
        "tool": step.tool,
        "query": step.query,
        "iteration": step.iteration,
        # pending, running, completed or failed.
        "status": step.status,
        "duration_ms": step.duration_ms,
        "started_at": step.created_at,
        # None while the call is running (no result yet).
        "success": result.success if result is not None else None,
        "error": result.error if result is not None else None,
        "result_preview": preview,
        "result_truncated": truncated,
        "result_length": len(result.content) if result is not None and result.content else 0,
    }


def _run_progress(run: AgentRun, now: datetime) -> dict:

    live = run.status in UNFINISHED_STATUSES

    if live:
        try:
            iterations, tool_calls, usage = totals_from_checkpoint(run)
        except ValueError:
            # A corrupt checkpoint: fall back to the recorded columns.
            live = False

    if not live:
        iterations = run.iteration_count
        tool_calls = run.tool_call_count
        input_tokens = run.input_tokens
        output_tokens = run.output_tokens
        total_tokens = run.total_tokens
        cost = run.estimated_cost_usd
    else:
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        total_tokens = usage.total_tokens
        cost = usage.estimated_cost

    ended = run.completed_at if not live else None

    return {
        "id": run.id,
        # created, running, waiting_for_tool, processing_result,
        # completed or failed.
        "status": run.status,
        # True while the run is going: the totals are from its checkpoint.
        "live": live,
        "iterations": iterations,
        "tool_calls": tool_calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": cost,
        "error": run.error,
        "started_at": run.started_at,
        "completed_at": run.completed_at,
        "elapsed_seconds": round(
            ((ended or now) - run.started_at).total_seconds(),
            1,
        ),
    }


async def build_task_progress(
    db: AsyncSession,
    task_id: int,
    now: datetime | None = None,
) -> dict | None:
    """None if there is no such task."""

    now = now or utc_now()

    task = await get_research_task(db=db, task_id=task_id)

    if task is None:
        return None

    job = await get_latest_job_for_task(db=db, task_id=task_id)
    run = await get_latest_agent_run(db=db, task_id=task_id)

    status = TaskStatus(task.status)
    finished = status in FINISHED_TASK_STATUSES

    can_resume = False

    if status == TaskStatus.FAILED:
        try:
            await check_can_resume(db=db, task=task, agent_run=run)
            can_resume = True
        except ValueError:
            pass

    return {
        "task": task,
        # Completed or failed: nothing will change unless it's resumed.
        "finished": finished,
        "can_resume": can_resume,
        "job": job,
        "run": _run_progress(run, now) if run is not None else None,
        "tool_calls": [
            _tool_call(step)
            for step in await list_task_steps(db=db, task_id=task_id)
        ],
        "events": await build_execution_timeline(db=db, task_id=task_id),
    }
