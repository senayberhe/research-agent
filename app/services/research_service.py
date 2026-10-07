import logging

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ResearchTask, TaskStatus
from app.jobs.service import create_research_job


logger = logging.getLogger(__name__)


async def create_research_task(
    db: AsyncSession,
    question: str,
) -> ResearchTask:

    task = ResearchTask(
        question=question,
        status=TaskStatus.PENDING,
    )

    db.add(task)

    # Gives the task its id without committing, so the task and its job are
    # saved together (create_research_job's commit covers both): there's
    # never a task without a job for a worker to pick up.
    await db.flush()

    await create_research_job(
        db=db,
        task_id=task.id,
    )

    await db.commit()

    await db.refresh(task)

    logger.info(
        "Research task created task_id=%s",
        task.id,
    )

    return task


async def get_research_task(
    db: AsyncSession,
    task_id: int,
) -> ResearchTask | None:

    result = await db.execute(
        select(ResearchTask).where(
            ResearchTask.id == task_id
        )
    )

    task = result.scalar_one_or_none()

    if task is None:

        logger.warning(
            "Research task not found task_id=%s",
            task_id,
        )

    return task


async def list_research_tasks(
    db: AsyncSession,
    limit: int = 20,
    offset: int = 0,
    status: TaskStatus | None = None,
) -> tuple[list[ResearchTask], int]:
    """A page of tasks, newest first, and how many there are in all
    (optionally only those with this status)."""

    query = select(ResearchTask)
    count = select(func.count()).select_from(ResearchTask)

    if status is not None:
        query = query.where(ResearchTask.status == status)
        count = count.where(ResearchTask.status == status)

    result = await db.execute(
        query
        .order_by(ResearchTask.created_at.desc(), ResearchTask.id.desc())
        .limit(limit)
        .offset(offset)
    )

    return list(result.scalars().all()), await db.scalar(count)

