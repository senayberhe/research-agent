"""build_system_metrics: headline metrics across the whole system, from
rows built directly in the (rolled-back) test session."""

from datetime import timedelta

import pytest

from app.db.models import (
    AgentRun,
    AgentRunStatus,
    JobEvent,
    JobStatus,
    ResearchJob,
    utc_now,
)
from app.jobs.events import record_tool_event
from app.services.system_metrics_service import (
    build_system_metrics,
)
from app.tests.test_workflow_service import create_task


@pytest.mark.asyncio
async def test_system_metrics_empty(db_session):
    metrics = await build_system_metrics(
        db=db_session,
    )

    assert metrics["total_jobs"] == 0

    assert metrics["pending_jobs"] == 0
    assert metrics["running_jobs"] == 0
    assert metrics["completed_jobs"] == 0
    assert metrics["failed_jobs"] == 0

    assert metrics["total_attempts"] == 0
    assert metrics["retry_count"] == 0

    assert metrics["average_job_duration_ms"] is None

    assert metrics["total_tool_calls"] == 0
    assert metrics["successful_tool_calls"] == 0
    assert metrics["failed_tool_calls"] == 0

    assert metrics["average_tool_latency_ms"] is None

    assert metrics["total_agent_runs"] == 0

    assert metrics["total_input_tokens"] == 0
    assert metrics["total_output_tokens"] == 0
    assert metrics["total_tokens"] == 0

    assert metrics["estimated_cost"] == 0

    assert metrics["success_rate"] == 0


@pytest.mark.asyncio
async def test_system_metrics_aggregate_data(
    db_session,
):
    now = utc_now()

    # Jobs and runs need a real task (foreign key); one task is enough.
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
        attempts=3,
        started_at=now,
        completed_at=now + timedelta(seconds=4),
    )

    pending_job = ResearchJob(
        task_id=task.id,
        status=JobStatus.PENDING,
        attempts=0,
    )

    running_job = ResearchJob(
        task_id=task.id,
        status=JobStatus.RUNNING,
        attempts=1,
        started_at=now,
    )

    db_session.add_all(
        [
            completed_job,
            failed_job,
            pending_job,
            running_job,
        ]
    )

    await db_session.flush()

    await record_tool_event(
        db=db_session,
        job_id=completed_job.id,
        event_type="tool_completed",
        tool="tavily",
        query="AI research",
        success=True,
        duration_ms=100,
    )

    await record_tool_event(
        db=db_session,
        job_id=completed_job.id,
        event_type="tool_completed",
        tool="arxiv",
        query="AI papers",
        success=True,
        duration_ms=300,
    )

    await record_tool_event(
        db=db_session,
        job_id=failed_job.id,
        event_type="tool_completed",
        tool="wikipedia",
        query="AI",
        success=False,
        duration_ms=500,
        error="Tool failed",
    )

    # state is the run's checkpoint (JSON), not a status; left unset.
    agent_run = AgentRun(
        task_id=task.id,
        status=AgentRunStatus.COMPLETED,
        input_tokens=1000,
        output_tokens=500,
        total_tokens=1500,
        estimated_cost_usd=0.25,
    )

    db_session.add(agent_run)

    await db_session.commit()

    metrics = await build_system_metrics(
        db=db_session,
    )

    assert metrics["total_jobs"] == 4

    assert metrics["pending_jobs"] == 1
    assert metrics["running_jobs"] == 1
    assert metrics["completed_jobs"] == 1
    assert metrics["failed_jobs"] == 1

    assert metrics["total_attempts"] == 5
    assert metrics["retry_count"] == 2

    assert (
        metrics["average_job_duration_ms"]
        == 3000
    )

    assert metrics["total_tool_calls"] == 3
    assert metrics["successful_tool_calls"] == 2
    assert metrics["failed_tool_calls"] == 1

    assert (
        metrics["average_tool_latency_ms"]
        == 300
    )

    assert metrics["total_agent_runs"] == 1

    assert metrics["total_input_tokens"] == 1000
    assert metrics["total_output_tokens"] == 500
    assert metrics["total_tokens"] == 1500

    assert metrics["estimated_cost"] == 0.25

    assert metrics["success_rate"] == 25


@pytest.mark.asyncio
async def test_system_metrics_survive_corrupt_event_metadata(db_session):

    task = await create_task(db_session)

    job = ResearchJob(
        task_id=task.id,
        status=JobStatus.COMPLETED,
        attempts=1,
    )

    db_session.add(job)
    await db_session.flush()

    await record_tool_event(
        db=db_session,
        job_id=job.id,
        event_type="tool_completed",
        tool="tavily",
        query="AI",
        success=True,
        duration_ms=100,
    )

    # Not valid JSON: counted as a call, without breaking the query.
    db_session.add(
        JobEvent(
            job_id=job.id,
            event_type="tool_failed",
            metadata_json="{not json",
        )
    )

    await db_session.commit()

    metrics = await build_system_metrics(
        db=db_session,
    )

    assert metrics["total_tool_calls"] == 2
    assert metrics["successful_tool_calls"] == 1
    assert metrics["failed_tool_calls"] == 1
    assert metrics["average_tool_latency_ms"] == 100
