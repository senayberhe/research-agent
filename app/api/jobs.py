from datetime import timedelta

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
)
from fastapi.responses import PlainTextResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user, require
from app.core.permissions import Permission
from app.db.database import get_db
from app.db.models import utc_now
from app.jobs.models import JobStatus
from app.jobs.event_service import (
    describe_job_event,
    format_job_timeline,
    get_job_events,
)
from app.jobs.execution_service import get_job_execution
from app.jobs.service import (
    get_research_job,
    get_worker_activity,
    list_job_attempts,
    list_jobs,
)
from app.schemas.jobs import (
    AgentExecutionResponse,
    CheckpointResponse,
    CostResponse,
    JobAttemptResponse,
    JobEventDetailResponse,
    JobExecutionResponse,
    JobListResponse,
    JobResponse,
    JobStateResponse,
    StepResponse,
    TokensResponse,
    ToolsResponse,
    WorkerResponse,
)
from app.schemas.research import ResearchJobResponse


# Every /jobs and /workers endpoint needs a signed-in user.
router = APIRouter(
    tags=["Jobs"],
    dependencies=[Depends(get_current_user)],
)


def event_response(event) -> JobEventDetailResponse:

    response = JobEventDetailResponse.model_validate(event)
    response.summary = describe_job_event(event)

    return response


@router.get(
    "/jobs",
    response_model=JobListResponse,
)
async def list_all_jobs(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status: JobStatus | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Jobs, newest first, a page at a time; optionally only those with
    this status. Each job's attempts and events are at GET /jobs/{job_id}."""

    rows, total = await list_jobs(
        db=db,
        limit=limit,
        offset=offset,
        status=status,
    )

    return {
        "items": [
            {
                **ResearchJobResponse.model_validate(job).model_dump(),
                "question": question,
            }
            for job, question in rows
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get(
    "/jobs/{job_id}",
    response_model=JobResponse,
)
async def get_job(
    job_id: int,
    db: AsyncSession = Depends(get_db),
):
    """The job, every attempt to run it, and its timeline of events."""

    job = await get_research_job(
        db=db,
        job_id=job_id,
    )

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Job not found",
        )

    attempts = await list_job_attempts(
        db=db,
        job_id=job_id,
    )

    events = await get_job_events(
        db=db,
        job_id=job_id,
    )

    # Built from the columns, then the history: validating JobResponse
    # straight from the job would read job.events, a lazy load, which async
    # sessions don't allow.
    return JobResponse(
        **ResearchJobResponse.model_validate(job).model_dump(),
        attempt_history=[
            JobAttemptResponse.model_validate(attempt)
            for attempt in attempts
        ],
        events=[event_response(event) for event in events],
    )


@router.get(
    "/jobs/{job_id}/execution",
    response_model=JobExecutionResponse,
)
async def get_execution(
    job_id: int,
    db: AsyncSession = Depends(get_db),
):
    """How the job's AI execution behaved: the job's current state, its
    history of events, and its agent run (iterations, tool calls, tokens,
    cost, checkpoint)."""

    execution = await get_job_execution(
        db=db,
        job_id=job_id,
    )

    if execution is None:
        raise HTTPException(
            status_code=404,
            detail="Job not found",
        )

    job = execution.job
    run = execution.agent_run

    agent_run = None

    if run is not None:
        checkpoint = execution.checkpoint

        agent_run = AgentExecutionResponse(
            id=run.id,
            status=run.status,
            started_at=run.started_at,
            completed_at=run.completed_at,
            error=run.error,
            iterations=run.iteration_count,
            tokens=TokensResponse(
                input=run.input_tokens,
                output=run.output_tokens,
                total=run.total_tokens,
            ),
            cost=CostResponse(
                estimated_usd=run.estimated_cost_usd,
            ),
            tools=ToolsResponse(
                requested=run.tool_call_count,
                executed=(
                    checkpoint.executed_tool_calls
                    if checkpoint is not None
                    else None
                ),
                calls=[
                    StepResponse.model_validate(step)
                    for step in execution.steps
                ],
            ),
            checkpoint=(
                CheckpointResponse.model_validate(checkpoint)
                if checkpoint is not None
                else None
            ),
        )

    return JobExecutionResponse(
        job_id=job.id,
        task_id=job.task_id,
        current_state=JobStateResponse.model_validate(job),
        history=[event_response(event) for event in execution.events],
        agent_run=agent_run,
    )


@router.get(
    "/jobs/{job_id}/timeline",
    response_class=PlainTextResponse,
)
async def get_timeline(
    job_id: int,
    db: AsyncSession = Depends(get_db),
):
    """The job's events as plain text, one per line, e.g.

        tool_started     → Tavily
        tool_completed   → Tavily, 842 ms
    """

    job = await get_research_job(
        db=db,
        job_id=job_id,
    )

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Job not found",
        )

    return format_job_timeline(
        await get_job_events(
            db=db,
            job_id=job_id,
        )
    )


@router.get(
    "/workers",
    # Analytics: operators and admins.
    dependencies=[Depends(require(Permission.ANALYTICS_VIEW))],
    response_model=list[WorkerResponse],
)
async def list_workers(
    hours: float = Query(default=24, gt=0),
    db: AsyncSession = Depends(get_db),
):
    """Workers that have run a job in the last `hours`, and what each is
    doing now."""

    return await get_worker_activity(
        db=db,
        since=utc_now() - timedelta(hours=hours),
    )
