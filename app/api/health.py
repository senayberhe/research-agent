from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user
from app.db.database import get_db
from app.jobs.service import get_queue_health


router = APIRouter(
    tags=["Health"],
)

# A pending job waiting longer than this suggests no worker is running.
STUCK_PENDING_SECONDS = 600


# GET /health (system health) is in app/main.py.


# /health is public (container health checks); queue details are not.
@router.get("/health/jobs", dependencies=[Depends(get_current_user)])
async def jobs_health(
    db: AsyncSession = Depends(get_db),
):
    """The job queue, for monitoring. status is "degraded" when a worker
    died holding a job (expired lease) or pending jobs have waited more
    than STUCK_PENDING_SECONDS (no worker picking them up)."""

    queue = await get_queue_health(db)

    problems = []

    if queue["expired_leases"]:
        problems.append(
            f"{queue['expired_leases']} running jobs have expired leases"
        )

    oldest = queue["oldest_pending_seconds"]

    if oldest is not None and oldest > STUCK_PENDING_SECONDS:
        problems.append(
            f"oldest pending job has waited {oldest:.0f}s; "
            "is a worker running?"
        )

    return {
        "status": "degraded" if problems else "healthy",
        "problems": problems,
        **queue,
    }
