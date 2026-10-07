import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ResearchTask, TaskStatus
from app.jobs.models import JobStatus
from app.jobs.service import (
    create_research_job,
    find_running_jobs,
    get_latest_job_for_task,
    mark_job_failed,
    requeue_job,
)
from app.services.agent_run_service import (
    fail_agent_run,
    find_unfinished_agent_runs,
    totals_from_checkpoint,
)
from app.services.state_machine import can_transition_task, transition_task

logger = logging.getLogger(__name__)


INTERRUPTED_ERROR = (
    "Interrupted: the process stopped before the run finished."
)


async def recover_interrupted_runs(db: AsyncSession) -> list[int]:
    """Marks runs cut off by a crash or restart as FAILED, and their tasks
    too. Returns the ids of the tasks that can be resumed (their run has a
    checkpoint).

    Call it only when nothing can still be working on an unfinished run:
    at worker startup, with a single worker.
    """

    resumable_task_ids = []

    for run in await find_unfinished_agent_runs(db):

        # The run never returned a result; its checkpoint says how far it
        # got (and keeps its totals from being recorded as 0).
        iteration_count, tool_call_count, usage = totals_from_checkpoint(run)

        await fail_agent_run(
            db=db,
            run=run,
            error=INTERRUPTED_ERROR,
            iteration_count=iteration_count,
            tool_call_count=tool_call_count,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            estimated_cost_usd=usage.estimated_cost,
        )

        task = await db.get(ResearchTask, run.task_id)

        if task is not None and can_transition_task(task, TaskStatus.FAILED):
            transition_task(task, TaskStatus.FAILED)

        logger.warning(
            "Recovered interrupted agent run run_id=%s task_id=%s "
            "iteration=%s resumable=%s",
            run.id,
            run.task_id,
            iteration_count,
            run.state is not None,
        )

        if run.state is not None:
            resumable_task_ids.append(run.task_id)

    await db.commit()

    return resumable_task_ids


async def recover_interrupted_jobs(
    db: AsyncSession,
    requeue: bool,
    max_attempts: int,
) -> list[int]:
    """Recovery for a worker starting up: anything a dead worker left
    behind is failed and, if requeue is true, put back in the queue to
    continue. Returns the ids of the tasks queued to continue.

    - Unfinished runs are marked FAILED (see recover_interrupted_runs).
    - Jobs left RUNNING are marked FAILED.
    - A task that can continue (still PENDING, or its run has a checkpoint
      to resume from) gets its job requeued, unless the job has used up its
      attempts. A task interrupted before jobs existed gets a new job.

    Assumes a single worker: with several, a job still running on another
    worker would look interrupted too.
    """

    resumable_task_ids = set(await recover_interrupted_runs(db))

    interrupted_jobs = await find_running_jobs(db)

    for job in interrupted_jobs:

        await mark_job_failed(db=db, job=job, error=INTERRUPTED_ERROR)

        logger.warning(
            "Recovered interrupted job job_id=%s task_id=%s worker_id=%s",
            job.id,
            job.task_id,
            job.worker_id,
        )

    await db.commit()

    if not requeue:

        if interrupted_jobs or resumable_task_ids:
            logger.info(
                "Not requeueing interrupted work (AGENT_RESUME_ON_STARTUP "
                "is off); tasks can be resumed with POST "
                "/research/{task_id}/resume"
            )

        return []

    queued_task_ids = []

    candidate_task_ids = resumable_task_ids | {
        job.task_id for job in interrupted_jobs
    }

    for task_id in sorted(candidate_task_ids):

        task = await db.get(ResearchTask, task_id)

        can_continue = task is not None and (
            TaskStatus(task.status) == TaskStatus.PENDING
            or task_id in resumable_task_ids
        )

        if not can_continue:
            continue

        job = await get_latest_job_for_task(db=db, task_id=task_id)

        if job is None:
            await create_research_job(db=db, task_id=task_id)

        elif JobStatus(job.status) == JobStatus.FAILED:

            if job.attempts >= max_attempts:
                logger.warning(
                    "Not requeueing job_id=%s: it has used all %d attempts",
                    job.id,
                    max_attempts,
                )
                continue

            await requeue_job(db=db, job=job)

        # PENDING: already queued. COMPLETED/RUNNING: leave it alone.
        elif JobStatus(job.status) != JobStatus.PENDING:
            continue

        queued_task_ids.append(task_id)

    if queued_task_ids:
        logger.info(
            "Queued %d interrupted tasks to continue: %s",
            len(queued_task_ids),
            queued_task_ids,
        )

    return queued_task_ids
