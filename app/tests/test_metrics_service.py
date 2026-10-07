"""Aggregate execution metrics for a research task (build_research_metrics)."""

from datetime import timedelta

import pytest
from sqlalchemy import select

from app.db.models import (
    AgentRun,
    AgentRunStatus,
    JobAttempt,
    JobStatus,
    ResearchJob,
    utc_now,
)
from app.jobs.events import record_tool_event
from app.jobs.service import (
    claim_next_job,
    get_jobs_for_task,
    mark_job_completed,
    mark_job_failed,
    requeue_job,
)
from app.schemas.research import ResearchMetricsResponse
from app.services.metrics_service import build_research_metrics
from app.services.research_service import create_research_task
from app.tests.conftest import TestSessionLocal
from app.tests.fake_agent_llm import FakeAgentLLM
from app.tests.test_timeline_service import (  # noqa: F401 (autouse fixture)
    fake_tools_and_health_file,
    run_task_through_worker,
)
from app.tests.test_workflow_service import create_task


async def metrics_for(task_id: int) -> dict:
    async with TestSessionLocal() as db:
        return await build_research_metrics(db=db, task_id=task_id)


@pytest.mark.asyncio
async def test_metrics_of_a_completed_task(clean_tables, monkeypatch):

    task_id = await run_task_through_worker(monkeypatch, FakeAgentLLM)

    metrics = await metrics_for(task_id)

    assert metrics["task_id"] == task_id

    assert metrics["total_jobs"] == 1
    assert metrics["completed_jobs"] == 1
    assert metrics["failed_jobs"] == 0
    assert metrics["total_attempts"] == 1
    assert metrics["retry_count"] == 0
    assert metrics["success_rate"] == 100.0

    # FakeAgentLLM searches once with the (fake) Tavily tool.
    assert metrics["total_tool_calls"] == 1
    assert metrics["successful_tool_calls"] == 1
    assert metrics["failed_tool_calls"] == 0
    assert metrics["average_tool_latency_ms"] is not None

    assert metrics["total_execution_time_ms"] > 0

    # Two LLM calls of 100 input + 50 output tokens.
    assert metrics["total_input_tokens"] == 200
    assert metrics["total_output_tokens"] == 100
    assert metrics["total_tokens"] == 300

    async with TestSessionLocal() as db:
        run = await db.scalar(select(AgentRun))

    assert metrics["estimated_cost"] == pytest.approx(run.estimated_cost_usd)

    # Fits the response schema.
    assert ResearchMetricsResponse(**metrics).total_tokens == 300


@pytest.mark.asyncio
async def test_failed_tool_calls_are_counted(clean_tables, monkeypatch):

    from app.services import workflow_service
    from app.tools.base import ToolResult

    class RateLimitedTavily:

        name = "tavily"

        async def search(self, query: str) -> ToolResult:
            return ToolResult(
                tool="tavily",
                query=query,
                content="",
                success=False,
                error="rate limited",
            )

    class RateLimitedRegistry:

        def __init__(self):
            self.tools = {"tavily": RateLimitedTavily()}

        def get(self, name: str):
            return self.tools[name]

    monkeypatch.setattr(workflow_service, "ToolRegistry", RateLimitedRegistry)

    task_id = await run_task_through_worker(monkeypatch, FakeAgentLLM)

    metrics = await metrics_for(task_id)

    assert metrics["total_tool_calls"] == 1
    assert metrics["successful_tool_calls"] == 0
    assert metrics["failed_tool_calls"] == 1
    assert metrics["average_tool_latency_ms"] is not None


@pytest.mark.asyncio
async def test_retries_and_execution_time_over_all_attempts(clean_tables):

    async with TestSessionLocal() as db:

        task = await create_research_task(db=db, question="What is RAG?")
        [job] = await get_jobs_for_task(db=db, task_id=task.id)

        # Attempt 1 fails...
        await claim_next_job(db=db, worker_id="worker-1")
        await mark_job_failed(
            db=db,
            job=job,
            error="Tavily timed out",
            lease_token=job.lease_token,
        )
        await db.commit()

        # ...attempt 2 completes.
        await requeue_job(db=db, job=job)
        await claim_next_job(db=db, worker_id="worker-2")
        await mark_job_completed(db=db, job=job, lease_token=job.lease_token)
        await db.commit()

        # Give the attempts known durations: 2s and 3s.
        attempts = (
            await db.execute(select(JobAttempt).order_by(JobAttempt.id))
        ).scalars().all()

        start = utc_now() - timedelta(minutes=10)

        for attempt, seconds in zip(attempts, (2, 3)):
            attempt.started_at = start
            attempt.ended_at = start + timedelta(seconds=seconds)

        await db.commit()

    metrics = await metrics_for(task.id)

    assert metrics["total_attempts"] == 2
    assert metrics["retry_count"] == 1
    assert metrics["completed_jobs"] == 1
    assert metrics["failed_jobs"] == 0

    # Both attempts, not just the last one.
    assert metrics["total_execution_time_ms"] == pytest.approx(5000)


@pytest.mark.asyncio
async def test_estimated_cost_is_none_when_a_run_is_unpriced(clean_tables):

    async with TestSessionLocal() as db:

        task = await create_research_task(db=db, question="What is RAG?")

        db.add_all(
            [
                AgentRun(
                    task_id=task.id,
                    status=AgentRunStatus.FAILED,
                    estimated_cost_usd=0.01,
                ),
                AgentRun(
                    task_id=task.id,
                    status=AgentRunStatus.COMPLETED,
                    estimated_cost_usd=None,
                ),
            ]
        )
        await db.commit()

    metrics = await metrics_for(task.id)

    assert metrics["estimated_cost"] is None
    assert ResearchMetricsResponse(**metrics).estimated_cost is None


@pytest.mark.asyncio
async def test_metrics_of_a_queued_task(clean_tables):

    async with TestSessionLocal() as db:
        task = await create_research_task(db=db, question="What is RAG?")

    metrics = await metrics_for(task.id)

    assert metrics["total_jobs"] == 1
    assert metrics["completed_jobs"] == 0
    assert metrics["total_attempts"] == 0
    assert metrics["total_tool_calls"] == 0
    assert metrics["average_tool_latency_ms"] is None
    assert metrics["total_execution_time_ms"] is None
    assert metrics["total_tokens"] == 0
    # No runs: nothing unpriced, so 0 rather than None.
    assert metrics["estimated_cost"] == 0
    assert metrics["success_rate"] == 0.0


@pytest.mark.asyncio
async def test_metrics_endpoint_after_a_run(clean_tables, monkeypatch):

    from app.tests.test_research_api import client

    task_id = await run_task_through_worker(monkeypatch, FakeAgentLLM)

    response = client.get(f"/research/{task_id}/metrics")

    assert response.status_code == 200

    data = response.json()

    assert data["completed_jobs"] == 1
    assert data["successful_tool_calls"] == 1
    assert data["total_tokens"] == 300
    assert data["success_rate"] == 100.0
    assert data["average_tool_latency_ms"] is not None


# -------------------------------------------------------------------
# Built from rows directly (in the rolled-back db_session)
# -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metrics_with_no_jobs(db_session):
    metrics = await build_research_metrics(
        db=db_session,
        task_id=999,
    )

    assert metrics["task_id"] == 999
    assert metrics["total_jobs"] == 0
    assert metrics["completed_jobs"] == 0
    assert metrics["failed_jobs"] == 0
    assert metrics["total_attempts"] == 0
    assert metrics["retry_count"] == 0
    assert metrics["total_tool_calls"] == 0
    assert metrics["successful_tool_calls"] == 0
    assert metrics["failed_tool_calls"] == 0
    assert metrics["average_tool_latency_ms"] is None
    assert metrics["total_execution_time_ms"] is None
    assert metrics["total_input_tokens"] == 0
    assert metrics["total_output_tokens"] == 0
    assert metrics["total_tokens"] == 0
    assert metrics["estimated_cost"] == 0
    assert metrics["success_rate"] == 0


@pytest.mark.asyncio
async def test_metrics_completed_job(db_session):
    now = utc_now()

    # Jobs and runs need a real task (foreign key).
    task = await create_task(db_session)

    job = ResearchJob(
        task_id=task.id,
        status=JobStatus.COMPLETED,
        attempts=1,
        started_at=now,
        completed_at=now + timedelta(seconds=2),
    )

    db_session.add(job)
    await db_session.flush()

    await record_tool_event(
        db=db_session,
        job_id=job.id,
        event_type="tool_completed",
        tool="tavily",
        query="AI research",
        success=True,
        duration_ms=100,
    )

    await record_tool_event(
        db=db_session,
        job_id=job.id,
        event_type="tool_completed",
        tool="arxiv",
        query="AI papers",
        success=True,
        duration_ms=300,
    )

    await db_session.commit()

    metrics = await build_research_metrics(
        db=db_session,
        task_id=task.id,
    )

    assert metrics["total_jobs"] == 1
    assert metrics["completed_jobs"] == 1
    assert metrics["failed_jobs"] == 0

    assert metrics["total_attempts"] == 1
    assert metrics["retry_count"] == 0

    assert metrics["total_tool_calls"] == 2
    assert metrics["successful_tool_calls"] == 2
    assert metrics["failed_tool_calls"] == 0

    assert metrics["average_tool_latency_ms"] == 200

    # No attempt records: the job's own start to completion.
    assert metrics["total_execution_time_ms"] == 2000

    assert metrics["success_rate"] == 100


@pytest.mark.asyncio
async def test_metrics_failed_job_and_retries(db_session):
    now = utc_now()

    task = await create_task(db_session)

    job = ResearchJob(
        task_id=task.id,
        status=JobStatus.FAILED,
        attempts=3,
        started_at=now,
        completed_at=now + timedelta(seconds=5),
    )

    db_session.add(job)
    await db_session.flush()

    await record_tool_event(
        db=db_session,
        job_id=job.id,
        event_type="tool_completed",
        tool="tavily",
        query="failed search",
        success=False,
        duration_ms=500,
        error="Tool failed",
    )

    await db_session.commit()

    metrics = await build_research_metrics(
        db=db_session,
        task_id=task.id,
    )

    assert metrics["total_jobs"] == 1
    assert metrics["completed_jobs"] == 0
    assert metrics["failed_jobs"] == 1

    assert metrics["total_attempts"] == 3
    assert metrics["retry_count"] == 2

    assert metrics["total_tool_calls"] == 1
    assert metrics["successful_tool_calls"] == 0
    assert metrics["failed_tool_calls"] == 1

    assert metrics["average_tool_latency_ms"] == 500

    assert metrics["total_execution_time_ms"] == 5000

    assert metrics["success_rate"] == 0


@pytest.mark.asyncio
async def test_metrics_agent_run_usage(db_session):
    task = await create_task(db_session)

    # state is the run's checkpoint (JSON), not a status; left unset.
    run_one = AgentRun(
        task_id=task.id,
        status=AgentRunStatus.COMPLETED,
        input_tokens=1000,
        output_tokens=500,
        total_tokens=1500,
        estimated_cost_usd=0.25,
    )

    run_two = AgentRun(
        task_id=task.id,
        status=AgentRunStatus.COMPLETED,
        input_tokens=2000,
        output_tokens=1000,
        total_tokens=3000,
        estimated_cost_usd=0.50,
    )

    db_session.add_all([
        run_one,
        run_two,
    ])

    await db_session.commit()

    metrics = await build_research_metrics(
        db=db_session,
        task_id=task.id,
    )

    assert metrics["total_input_tokens"] == 3000
    assert metrics["total_output_tokens"] == 1500
    assert metrics["total_tokens"] == 4500
    assert metrics["estimated_cost"] == 0.75


@pytest.mark.asyncio
async def test_metrics_multiple_jobs(db_session):
    now = utc_now()

    task = await create_task(db_session)

    completed_job = ResearchJob(
        task_id=task.id,
        status=JobStatus.COMPLETED,
        attempts=1,
        started_at=now,
        completed_at=now + timedelta(seconds=2),
    )

    failed_job = ResearchJob(
        task_id=task.id,
        status=JobStatus.FAILED,
        attempts=2,
        started_at=now,
        completed_at=now + timedelta(seconds=3),
    )

    db_session.add_all([
        completed_job,
        failed_job,
    ])

    await db_session.flush()

    await record_tool_event(
        db=db_session,
        job_id=completed_job.id,
        event_type="tool_completed",
        tool="tavily",
        query="query one",
        success=True,
        duration_ms=100,
    )

    await record_tool_event(
        db=db_session,
        job_id=failed_job.id,
        event_type="tool_completed",
        tool="arxiv",
        query="query two",
        success=False,
        duration_ms=300,
    )

    await db_session.commit()

    metrics = await build_research_metrics(
        db=db_session,
        task_id=task.id,
    )

    assert metrics["total_jobs"] == 2
    assert metrics["completed_jobs"] == 1
    assert metrics["failed_jobs"] == 1

    assert metrics["total_attempts"] == 3
    assert metrics["retry_count"] == 1

    assert metrics["total_tool_calls"] == 2
    assert metrics["successful_tool_calls"] == 1
    assert metrics["failed_tool_calls"] == 1

    assert metrics["average_tool_latency_ms"] == 200

    assert metrics["total_execution_time_ms"] == 5000

    assert metrics["success_rate"] == 50

