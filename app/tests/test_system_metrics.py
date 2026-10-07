"""System-wide operational metrics (GET /metrics): jobs, agent runs,
tools, workers and the queue."""

from datetime import timedelta

import pytest
from sqlalchemy import select

from app.db.models import (
    AgentRun,
    AgentRunStatus,
    JobAttempt,
    ResearchJob,
    utc_now,
)
from app.jobs.events import record_tool_event
from app.jobs.service import (
    claim_next_job,
    create_research_job,
    mark_job_completed,
    mark_job_failed,
    mark_job_running,
    requeue_job,
)
from app.schemas.metrics import SystemMetricsResponse
from app.services.system_metrics_service import (
    build_system_metrics,
    build_system_monitoring,
)
from app.tests.conftest import TestSessionLocal
from app.tests.test_research_api import client
from app.tests.test_workflow_service import create_task


async def set_attempt_durations(db, job_id: int, seconds: list[float]) -> None:
    """Gives the job's attempts known durations, in order."""

    attempts = (
        await db.execute(
            select(JobAttempt)
            .where(JobAttempt.job_id == job_id)
            .order_by(JobAttempt.id)
        )
    ).scalars().all()

    start = utc_now() - timedelta(minutes=30)

    for attempt, duration in zip(attempts, seconds, strict=True):
        attempt.started_at = start
        attempt.ended_at = start + timedelta(seconds=duration)


async def seed() -> None:
    """In the last 24 hours:

    jobs:  A completed (1 attempt, 2s), B failed after a retry (1s + 3s),
           C pending, D running on worker-busy
    runs:  1500 tokens / $0.25 / 2 iterations / 3 tool calls / 10s, and
           300 tokens / unpriced / 4 iterations / 1 tool call / unfinished
    tools: tavily ok 100ms, tavily failed 300ms, arxiv ok 200ms, and a
           tavily call still running (job events)

    Plus a completed job, a run and a tool call from two days ago, outside
    the window.
    """

    async with TestSessionLocal() as db:

        task = await create_task(db)
        now = utc_now()
        two_days_ago = now - timedelta(days=2)

        # A: completed.
        job_a = await create_research_job(db=db, task_id=task.id)
        await claim_next_job(db=db, worker_id="worker-1")
        await mark_job_completed(db=db, job=job_a, lease_token=job_a.lease_token)
        await db.commit()
        await set_attempt_durations(db, job_a.id, [2])

        # B: fails, is retried, fails again.
        job_b = await create_research_job(db=db, task_id=task.id)

        for worker_id in ("worker-1", "worker-2"):
            await claim_next_job(db=db, worker_id=worker_id)
            await mark_job_failed(
                db=db,
                job=job_b,
                error="Tavily timed out",
                lease_token=job_b.lease_token,
            )
            await db.commit()

            if worker_id == "worker-1":
                await requeue_job(db=db, job=job_b)

        await set_attempt_durations(db, job_b.id, [1, 3])

        # C: pending. D: running.
        await create_research_job(db=db, task_id=task.id)
        job_d = await create_research_job(db=db, task_id=task.id)

        # C was created first, so it's next in the queue; take D directly.
        await mark_job_running(db=db, job=job_d, worker_id="worker-busy")
        await db.commit()

        # Old: outside the window.
        db.add(
            ResearchJob(
                task_id=task.id,
                status="completed",
                attempts=1,
                created_at=two_days_ago,
                started_at=two_days_ago,
                completed_at=two_days_ago + timedelta(seconds=99),
            )
        )

        db.add_all(
            [
                AgentRun(
                    task_id=task.id,
                    status=AgentRunStatus.COMPLETED,
                    iteration_count=2,
                    tool_call_count=3,
                    input_tokens=1000,
                    output_tokens=500,
                    total_tokens=1500,
                    estimated_cost_usd=0.25,
                    started_at=now - timedelta(seconds=10),
                    completed_at=now,
                ),
                AgentRun(
                    task_id=task.id,
                    status=AgentRunStatus.RUNNING,
                    iteration_count=4,
                    tool_call_count=1,
                    input_tokens=200,
                    output_tokens=100,
                    total_tokens=300,
                    estimated_cost_usd=None,
                    started_at=now,
                ),
                AgentRun(
                    task_id=task.id,
                    status=AgentRunStatus.COMPLETED,
                    total_tokens=99999,
                    estimated_cost_usd=99.0,
                    started_at=two_days_ago,
                    completed_at=two_days_ago,
                ),
            ]
        )

        # Tool calls, as the workflow records them (on job A).
        async def tool_call(event_type, tool, success=None, duration_ms=None):
            return await record_tool_event(
                db=db,
                job_id=job_a.id,
                event_type=event_type,
                tool=tool,
                query="RAG",
                success=success,
                duration_ms=duration_ms,
            )

        await tool_call("tool_completed", "tavily", True, 100.0)
        await tool_call("tool_failed", "tavily", False, 300.0)
        await tool_call("tool_completed", "arxiv", True, 200.0)
        # Still running: not counted.
        await tool_call("tool_started", "tavily")

        old = await tool_call("tool_completed", "tavily", True, 9999.0)
        old.created_at = two_days_ago

        await db.commit()


async def metrics(hours: float = 24) -> dict:
    async with TestSessionLocal() as db:
        return await build_system_monitoring(
            db=db,
            since=utc_now() - timedelta(hours=hours),
        )


@pytest.mark.asyncio
async def test_job_stats(clean_tables):

    await seed()

    jobs = (await metrics())["jobs"]

    assert jobs["total"] == 4
    assert jobs["by_status"] == {
        "pending": 1,
        "running": 1,
        "completed": 1,
        "failed": 1,
    }

    # A: 1, B: 2, D: 1 attempts; 3 jobs started, 1 needed a retry.
    assert jobs["total_attempts"] == 4
    assert jobs["retry_count"] == 1
    assert jobs["retry_rate"] == pytest.approx(33.33)

    # Finished jobs: A took 2s, B 1s + 3s.
    assert jobs["average_duration_ms"] == pytest.approx(3000)

    # 1 of the 2 finished jobs completed.
    assert jobs["success_rate"] == 50.0


@pytest.mark.asyncio
async def test_agent_stats(clean_tables):

    await seed()

    agents = (await metrics())["agents"]

    assert agents["total_runs"] == 2
    assert agents["by_status"] == {"completed": 1, "running": 1}

    assert agents["input_tokens"] == 1200
    assert agents["output_tokens"] == 600
    assert agents["total_tokens"] == 1800

    # The unpriced run isn't in the total, and is counted.
    assert agents["estimated_cost_usd"] == pytest.approx(0.25)
    assert agents["unpriced_runs"] == 1

    assert agents["average_iterations"] == 3
    assert agents["average_tool_calls"] == 2

    # Only the finished run has a duration.
    assert agents["average_duration_ms"] == pytest.approx(10000)


@pytest.mark.asyncio
async def test_tool_stats(clean_tables):

    await seed()

    tools = (await metrics())["tools"]

    # Finished calls only (the running one isn't counted).
    assert tools["calls"] == 3
    assert tools["succeeded"] == 2
    assert tools["failed"] == 1
    assert tools["success_rate"] == pytest.approx(66.67)
    assert tools["average_latency_ms"] == pytest.approx(200)
    # 95th percentile of 100, 200, 300 (interpolated).
    assert tools["p95_latency_ms"] == pytest.approx(290)

    assert set(tools["by_tool"]) == {"arxiv", "tavily"}

    tavily = tools["by_tool"]["tavily"]

    assert tavily["calls"] == 2
    assert tavily["succeeded"] == 1
    assert tavily["failed"] == 1
    assert tavily["success_rate"] == 50.0
    assert tavily["average_latency_ms"] == pytest.approx(200)

    arxiv = tools["by_tool"]["arxiv"]

    assert arxiv["calls"] == 1
    assert arxiv["success_rate"] == 100.0
    assert arxiv["p95_latency_ms"] == pytest.approx(200)


@pytest.mark.asyncio
async def test_workers_running_jobs_and_queue(clean_tables):

    await seed()

    result = await metrics()

    [running] = result["running_jobs"]

    assert running["worker_id"] == "worker-busy"
    assert running["attempt"] == 1
    assert running["running_for_seconds"] >= 0
    assert running["lease_expired"] is False

    workers = {worker["worker_id"]: worker for worker in result["workers"]}

    assert workers["worker-busy"]["state"] == "busy"
    assert workers["worker-1"]["state"] == "idle"
    assert workers["worker-2"]["state"] == "idle"

    assert result["queue"] == {
        "pending_jobs": 1,
        "running_jobs": 1,
        "oldest_pending_seconds": result["queue"]["oldest_pending_seconds"],
        "expired_leases": 0,
    }
    assert result["queue"]["oldest_pending_seconds"] >= 0

    # Fits the response schema.
    SystemMetricsResponse(**result)


@pytest.mark.asyncio
async def test_longer_window_includes_older_data(clean_tables):

    await seed()

    result = await metrics(hours=72)

    assert result["jobs"]["total"] == 5
    assert result["agents"]["total_tokens"] == 1800 + 99999
    assert result["tools"]["calls"] == 4


@pytest.mark.asyncio
async def test_metrics_with_nothing_recorded(clean_tables):

    result = await metrics()

    assert result["jobs"]["total"] == 0
    assert result["jobs"]["by_status"] == {
        "pending": 0,
        "running": 0,
        "completed": 0,
        "failed": 0,
    }
    # Nothing to measure: None, not a misleading 0%.
    assert result["jobs"]["retry_rate"] is None
    assert result["jobs"]["success_rate"] is None
    assert result["jobs"]["average_duration_ms"] is None

    assert result["agents"]["total_runs"] == 0
    assert result["agents"]["estimated_cost_usd"] == 0
    assert result["agents"]["average_iterations"] is None

    assert result["tools"]["calls"] == 0
    assert result["tools"]["success_rate"] is None
    assert result["tools"]["p95_latency_ms"] is None
    assert result["tools"]["by_tool"] == {}

    assert result["workers"] == []
    assert result["running_jobs"] == []

    SystemMetricsResponse(**result)


# -------------------------
# build_system_metrics (the flat summary)
# -------------------------


SUMMARY_KEYS = {
    "total_jobs",
    "pending_jobs",
    "running_jobs",
    "completed_jobs",
    "failed_jobs",
    "total_attempts",
    "retry_count",
    "average_job_duration_ms",
    "total_tool_calls",
    "successful_tool_calls",
    "failed_tool_calls",
    "average_tool_latency_ms",
    "total_agent_runs",
    "total_input_tokens",
    "total_output_tokens",
    "total_tokens",
    "estimated_cost",
    "success_rate",
}


@pytest.mark.asyncio
async def test_system_metrics_all_time(clean_tables):

    await seed()

    async with TestSessionLocal() as db:
        summary = await build_system_metrics(db=db)

    assert set(summary) == SUMMARY_KEYS

    # Includes the job, run and step from two days ago.
    assert summary["total_jobs"] == 5
    assert summary["pending_jobs"] == 1
    assert summary["running_jobs"] == 1
    assert summary["completed_jobs"] == 2
    assert summary["failed_jobs"] == 1

    # A: 1, B: 2, D: 1, old: 1 attempts.
    assert summary["total_attempts"] == 5
    assert summary["retry_count"] == 1

    # A 2s, B 1s + 3s, old job 99s (no attempt records: its own times).
    assert summary["average_job_duration_ms"] == pytest.approx(
        (2000 + 4000 + 99000) / 3
    )

    # Failed calls counted too.
    assert summary["total_tool_calls"] == 4
    assert summary["successful_tool_calls"] == 3
    assert summary["failed_tool_calls"] == 1
    assert summary["average_tool_latency_ms"] == pytest.approx(
        (100 + 300 + 200 + 9999) / 4
    )

    assert summary["total_agent_runs"] == 3
    assert summary["total_input_tokens"] == 1200
    assert summary["total_output_tokens"] == 600
    assert summary["total_tokens"] == 1800 + 99999
    # Priced runs only (the unpriced one is left out).
    assert summary["estimated_cost"] == pytest.approx(99.25)

    # 2 of 5 jobs completed.
    assert summary["success_rate"] == 40.0


@pytest.mark.asyncio
async def test_system_metrics_in_a_window(clean_tables):

    await seed()

    async with TestSessionLocal() as db:
        summary = await build_system_metrics(
            db=db,
            since=utc_now() - timedelta(hours=24),
        )

    assert summary["total_jobs"] == 4
    assert summary["completed_jobs"] == 1
    assert summary["total_tool_calls"] == 3
    assert summary["total_agent_runs"] == 2
    assert summary["estimated_cost"] == pytest.approx(0.25)
    assert summary["success_rate"] == 25.0


@pytest.mark.asyncio
async def test_system_metrics_with_nothing_recorded(clean_tables):

    async with TestSessionLocal() as db:
        summary = await build_system_metrics(db=db)

    assert summary["total_jobs"] == 0
    assert summary["average_job_duration_ms"] is None
    assert summary["average_tool_latency_ms"] is None
    assert summary["total_tokens"] == 0
    assert summary["estimated_cost"] == 0
    assert summary["success_rate"] == 0.0


# -------------------------
# GET /research/metrics/system
# -------------------------


@pytest.mark.asyncio
async def test_metrics_endpoint(clean_tables):

    await seed()

    response = client.get("/research/metrics/system")

    assert response.status_code == 200

    body = response.json()

    assert body["jobs"]["total"] == 4
    assert body["jobs"]["success_rate"] == 50.0
    assert body["summary"]["total_jobs"] == 4
    assert body["summary"]["success_rate"] == 25.0
    assert body["agents"]["total_tokens"] == 1800
    assert body["tools"]["by_tool"]["tavily"]["calls"] == 2
    assert body["running_jobs"][0]["worker_id"] == "worker-busy"
    assert body["queue"]["pending_jobs"] == 1

    # A wider window.
    assert client.get("/research/metrics/system?hours=72").json()["jobs"]["total"] == 5


@pytest.mark.asyncio
async def test_metrics_endpoint_rejects_bad_window(clean_tables):

    assert client.get("/research/metrics/system?hours=0").status_code == 422
    assert client.get("/research/metrics/system?hours=-1").status_code == 422


# -------------------------
# GET /research/metrics
# -------------------------


@pytest.mark.asyncio
async def test_research_metrics_endpoint(clean_tables):

    await seed()

    response = client.get("/research/metrics")

    # Not mistaken for GET /research/{task_id}.
    assert response.status_code == 200

    body = response.json()

    assert set(body) == SUMMARY_KEYS

    # All time: includes the job from two days ago.
    assert body["total_jobs"] == 5
    assert body["failed_tool_calls"] == 1
    assert body["success_rate"] == 40.0


@pytest.mark.asyncio
async def test_research_task_routes_still_work(clean_tables):

    # The new route doesn't shadow these.
    assert client.get("/research/999999").status_code == 404
    assert client.get("/research/999999/metrics").status_code == 404
