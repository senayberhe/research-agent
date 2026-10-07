from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
)
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user, require
from app.core.permissions import Permission
from app.db.database import get_db
from app.db.models import TaskStatus, User, utc_now
from app.jobs.event_service import (
    get_job_events,
    parse_event_metadata,
)
from app.jobs.models import JobStatus
from app.jobs.service import (
    create_research_job,
    get_jobs_for_task,
    get_latest_job_for_task,
    requeue_job,
)
from app.schemas.jobs import (
    AgentRunResponse,
    StepResponse,
)
from app.schemas.metrics import JobTimeseriesResponse, SystemMetricsSummary
from app.schemas.research import (
    ErrorBudgetReportResponse,
    ExecutionTimelineResponse,
    JobEventResponse,
    ResearchExecutionResponse,
    ResearchJobResponse,
    ResearchListResponse,
    ResearchMetricsResponse,
    ResearchRequest,
    ResearchResponse,
    SLOReportResponse,
    TaskProgressResponse,
)
from app.services.agent_run_service import (
    get_latest_agent_run,
    list_agent_runs,
)
from app.services.error_budget_service import build_error_budget_report
from app.services.metrics_service import build_research_metrics
from app.services.progress_service import build_task_progress
from app.services.research_service import (
    create_research_task,
    get_research_task,
    list_research_tasks,
)
from app.services.slo_service import build_slo_report
from app.services.step_service import list_task_steps
from app.services.timeseries_service import RANGES, build_job_timeseries
from app.services.system_metrics_service import (
    build_system_metrics,
)
from app.services.timeline_service import (
    build_execution_timeline,
)
from app.services.workflow_service import check_can_resume


# Every /research endpoint needs a signed-in user.
router = APIRouter(
    prefix="/research",
    tags=["Research"],
    dependencies=[Depends(get_current_user)],
)


@router.post(
    "",
    response_model=ResearchResponse,
    status_code=202,
)
async def create_task(
    request: ResearchRequest,
    db: AsyncSession = Depends(get_db),
    # Researchers and admins (403 for viewers).
    user: User = Depends(require(Permission.RESEARCH_CREATE)),
):

    task = await create_research_task(
        db=db,
        question=request.question,
        # The signed-in user, as this request loaded it from the database
        # (a stand-in user that isn't stored, as in tests, isn't linked).
        created_by=user if inspect(user).persistent else None,
    )

    return task


@router.get(
    "",
    response_model=ResearchListResponse,
)
async def list_tasks(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status: TaskStatus | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Research tasks, newest first, a page at a time; optionally only
    those with this status."""

    tasks, total = await list_research_tasks(
        db=db,
        limit=limit,
        offset=offset,
        status=status,
    )

    return {
        "items": tasks,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


# Declared before "/{task_id}": otherwise GET /research/metrics would match
# that route (task_id="metrics") and fail with 422.
@router.get(
    "/metrics",
    response_model=SystemMetricsSummary,
    # Analytics: operators and admins.
    dependencies=[Depends(require(Permission.ANALYTICS_VIEW))],
)
async def get_system_metrics(
    db: AsyncSession = Depends(get_db),
):
    """Headline metrics across the whole system, all time (see
    build_system_metrics). GET /research/metrics/system has these for a
    time window, plus per-tool stats, workers, running jobs and the queue.
    For Prometheus, scrape GET /metrics."""

    return await build_system_metrics(
        db=db,
    )


# Also before "/{task_id}".
@router.get(
    "/metrics/timeseries",
    response_model=JobTimeseriesResponse,
    dependencies=[Depends(require(Permission.ANALYTICS_VIEW))],
)
async def get_job_timeseries(
    range: str = Query(default="24h", pattern="^(" + "|".join(RANGES) + ")$"),
    db: AsyncSession = Depends(get_db),
):
    """Jobs finished (completed and failed) and their p95 latency, per
    bucket: hourly for 24h, 6-hourly for 7d, daily for 30d."""

    return await build_job_timeseries(db, range, now=utc_now())


# Also before "/{task_id}" (GET /research/slo isn't a task).
@router.get(
    "/slo",
    response_model=SLOReportResponse,
    dependencies=[Depends(require(Permission.ANALYTICS_VIEW))],
)
async def get_slo_report(
    db: AsyncSession = Depends(get_db),
):
    """Each SLO (app/core/slo.py) against its SLI over the last 30 days:
    healthy or breached."""

    return await build_slo_report(db=db)


@router.get(
    "/slo/error-budget",
    response_model=ErrorBudgetReportResponse,
    dependencies=[Depends(require(Permission.ANALYTICS_VIEW))],
)
async def get_error_budget_report(
    db: AsyncSession = Depends(get_db),
):
    """Each SLO's error budget over the last 30 days: how much is used, how
    much is left, and how fast it's going (burn rate over the last hour)."""

    return await build_error_budget_report(db=db)


@router.get(
    "/{task_id}",
    response_model=ResearchResponse,
)
async def get_task(
    task_id: int,
    db: AsyncSession = Depends(get_db),
):
    task = await get_research_task(
        db=db,
        task_id=task_id,
    )

    if task is None:
        raise HTTPException(
            status_code=404,
            detail="Research task not found",
        )

    return task


@router.post(
    "/{task_id}/resume",
    response_model=ResearchResponse,
    status_code=202,
)
async def resume_research(
    task_id: int,
    db: AsyncSession = Depends(get_db),
    # Researchers and admins (403 for viewers).
    user: User = Depends(require(Permission.RESEARCH_RESUME)),
):
    """Queues a failed task to continue: a worker claims its job and resumes
    the task's run in place, from its last checkpoint."""

    task = await get_research_task(
        db=db,
        task_id=task_id,
    )

    if task is None:
        raise HTTPException(
            status_code=404,
            detail="Research task not found",
        )

    run = await get_latest_agent_run(
        db=db,
        task_id=task_id,
    )

    # Checked here so the caller learns straight away why a task can't be
    # resumed, instead of a worker failing the job later.
    try:
        await check_can_resume(
            db=db,
            task=task,
            agent_run=run,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc

    job = await get_latest_job_for_task(
        db=db,
        task_id=task_id,
    )

    if job is not None and JobStatus(job.status) == JobStatus.RUNNING:
        raise HTTPException(
            status_code=409,
            detail=f"Research task {task_id} is already being run "
            f"(job {job.id}).",
        )

    if job is not None and JobStatus(job.status) == JobStatus.FAILED:
        await requeue_job(db=db, job=job)

    # No job (a task from before jobs existed) or a finished one: queue a
    # new job. An already PENDING job is left as it is.
    elif job is None or JobStatus(job.status) == JobStatus.COMPLETED:
        await create_research_job(db=db, task_id=task_id)

    await db.refresh(task)

    return task


async def _require_task(db: AsyncSession, task_id: int) -> None:

    task = await get_research_task(
        db=db,
        task_id=task_id,
    )

    if task is None:
        raise HTTPException(
            status_code=404,
            detail="Research task not found",
        )


@router.get(
    "/{task_id}/jobs",
    response_model=list[ResearchJobResponse],
)
async def get_task_jobs(
    task_id: int,
    db: AsyncSession = Depends(get_db),
):
    """The task's jobs, oldest first. Each job's events are at
    GET /research/{task_id}/jobs/{job_id}/events."""

    await _require_task(db, task_id)

    jobs = await get_jobs_for_task(
        db=db,
        task_id=task_id,
    )

    return [
        {
            "id": job.id,
            "task_id": job.task_id,
            "status": job.status,
            "attempts": job.attempts,
            "worker_id": job.worker_id,
            "lease_expires_at": job.lease_expires_at,
            "error": job.error,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
        }
        for job in jobs
    ]


@router.get(
    "/{task_id}/execution",
    response_model=ResearchExecutionResponse,
)
async def get_research_execution(
    task_id: int,
    db: AsyncSession = Depends(get_db),
):
    """The task and all its jobs, oldest first."""

    task = await get_research_task(
        db=db,
        task_id=task_id,
    )

    if task is None:
        raise HTTPException(
            status_code=404,
            detail="Research task not found.",
        )

    jobs = await get_jobs_for_task(
        db=db,
        task_id=task_id,
    )

    return {
        "task": {
            "id": task.id,
            "question": task.question,
            "summary": task.summary,
            "status": task.status,
            "created_at": task.created_at,
        },
        "jobs": [
            {
                "id": job.id,
                "task_id": job.task_id,
                "status": job.status,
                "attempts": job.attempts,
                "worker_id": job.worker_id,
                "lease_expires_at": job.lease_expires_at,
                "error": job.error,
                "created_at": job.created_at,
                "started_at": job.started_at,
                "completed_at": job.completed_at,
            }
            for job in jobs
        ],
    }


@router.get(
    "/{task_id}/progress",
    response_model=TaskProgressResponse,
)
async def get_task_progress(
    task_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Everything needed to follow a task live, in one snapshot: the task
    (and its summary once done), its latest job, its agent run's live
    totals, and its timeline. Poll it until "finished" is true."""

    progress = await build_task_progress(db=db, task_id=task_id)

    if progress is None:
        raise HTTPException(
            status_code=404,
            detail="Research task not found.",
        )

    return progress


@router.get(
    "/{task_id}/timeline",
    response_model=ExecutionTimelineResponse,
)
async def get_research_timeline(
    task_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Everything that happened to the task, oldest first: job events,
    research steps and agent runs (see build_execution_timeline)."""

    task = await get_research_task(
        db=db,
        task_id=task_id,
    )

    if task is None:
        raise HTTPException(
            status_code=404,
            detail="Research task not found.",
        )

    events = await build_execution_timeline(
        db=db,
        task_id=task_id,
    )

    return {
        "task_id": task_id,
        "events": events,
    }


@router.get(
    "/{task_id}/metrics",
    response_model=ResearchMetricsResponse,
)
async def get_research_metrics(
    task_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Aggregate execution metrics for the task: jobs, retries, tool calls
    and latency, execution time, tokens, cost and success rate (see
    build_research_metrics)."""

    task = await get_research_task(
        db=db,
        task_id=task_id,
    )

    if task is None:
        raise HTTPException(
            status_code=404,
            detail="Research task not found.",
        )

    return await build_research_metrics(
        db=db,
        task_id=task_id,
    )


@router.get(
    "/{task_id}/jobs/{job_id}/events",
    response_model=list[JobEventResponse],
)
async def get_job_event_history(
    task_id: int,
    job_id: int,
    db: AsyncSession = Depends(get_db),
):
    """The job's events, oldest first. 404 if the job doesn't belong to
    this task."""

    jobs = await get_jobs_for_task(
        db=db,
        task_id=task_id,
    )

    job = next(
        (
            item
            for item in jobs
            if item.id == job_id
        ),
        None,
    )

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Job not found.",
        )

    events = await get_job_events(
        db=db,
        job_id=job_id,
    )

    return [
        {
            "id": event.id,
            "job_id": event.job_id,
            "event_type": event.event_type,
            "message": event.message,
            "metadata": parse_event_metadata(event),
            "created_at": event.created_at,
        }
        for event in events
    ]


@router.get(
    "/{task_id}/runs",
    response_model=list[AgentRunResponse],
)
async def get_task_runs(
    task_id: int,
    db: AsyncSession = Depends(get_db),
):
    """The task's agent runs: duration, tokens, cost, tool calls, error."""

    await _require_task(db, task_id)

    return await list_agent_runs(
        db=db,
        task_id=task_id,
    )


@router.get(
    "/{task_id}/steps",
    response_model=list[StepResponse],
)
async def get_task_steps(
    task_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Every tool call the agent made for the task, in order."""

    await _require_task(db, task_id)

    return await list_task_steps(
        db=db,
        task_id=task_id,
    )
