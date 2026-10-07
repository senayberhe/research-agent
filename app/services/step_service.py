from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import ResearchStep, StepStatus

async def update_step_status(
    db: AsyncSession,
    step: ResearchStep,
    status: StepStatus,
) -> ResearchStep:
    step.status = status
    await db.flush()
    return step


async def create_agent_step(
    db: AsyncSession,
    task_id: int,
    tool: str,
    query: str,
    iteration: int,
) -> ResearchStep:

    step = ResearchStep(
        task_id=task_id,
        tool=tool,
        query=query,
        iteration=iteration,
        status=StepStatus.RUNNING,
    )

    db.add(step)

    await db.flush()

    return step


async def list_task_steps(
    db: AsyncSession,
    task_id: int,
) -> list[ResearchStep]:
    """The task's tool calls in order, with their results loaded."""

    result = await db.execute(
        select(ResearchStep)
        .where(ResearchStep.task_id == task_id)
        .order_by(ResearchStep.id)
        .options(selectinload(ResearchStep.results))
    )

    return list(result.scalars().all())
