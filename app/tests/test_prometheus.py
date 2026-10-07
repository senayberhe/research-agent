"""Prometheus metrics at GET /metrics, computed from the database: job,
tool and agent activity (counters, histograms) and the current state."""

from datetime import timedelta

import pytest
from prometheus_client.parser import text_string_to_metric_families
from sqlalchemy import select

from app.db.models import AgentRun, JobAttempt, JobStatus, ResearchJob, utc_now
from app.jobs.events import record_tool_event
from app.jobs.service import (
    create_research_job,
    mark_job_completed,
    mark_job_running,
    recover_stale_jobs,
    release_job,
)
from app.tests.conftest import TestSessionLocal
from app.tests.fake_agent_llm import FakeAgentLLM
from app.tests.test_research_api import client
from app.tests.test_timeline_service import (  # noqa: F401 (autouse fixture)
    fake_tools_and_health_file,
    run_task_through_worker,
)
from app.tests.test_workflow_service import BrokenLLM, create_task


class Metrics:
    """GET /metrics, parsed."""

    def __init__(self):
        response = client.get("/metrics")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")

        self.text = response.text
        self.samples = {
            (sample.name, frozenset(sample.labels.items())): sample.value
            for family in text_string_to_metric_families(response.text)
            for sample in family.samples
        }

    def __call__(self, name: str, **labels) -> float:
        """A sample's value; 0 if it isn't there (nothing recorded)."""

        return self.samples.get((name, frozenset(labels.items())), 0.0)


# -------------------------
# Current state
# -------------------------


@pytest.mark.asyncio
async def test_state_metrics(clean_tables):

    async with TestSessionLocal() as db:

        task = await create_task(db)
        now = utc_now()

        def job(status, lease_seconds=None, worker_id=None, waited=0):
            return ResearchJob(
                task_id=task.id,
                status=status,
                attempts=0 if status == JobStatus.PENDING else 1,
                worker_id=worker_id,
                created_at=now - timedelta(seconds=waited),
                lease_expires_at=(
                    now + timedelta(seconds=lease_seconds)
                    if lease_seconds is not None
                    else None
                ),
            )

        db.add_all(
            [
                job(JobStatus.PENDING, waited=120),
                job(JobStatus.RUNNING, 300, "worker-1"),
                job(JobStatus.RUNNING, -5, "worker-dead"),
                job(JobStatus.COMPLETED),
            ]
        )
        await db.commit()

    metrics = Metrics()

    assert metrics("research_jobs", status="pending") == 1
    assert metrics("research_jobs", status="running") == 2
    assert metrics("research_jobs", status="completed") == 1
    assert metrics("research_jobs", status="failed") == 0

    # Every job ever created, whatever its status.
    assert metrics("research_jobs_created_total") == 4

    assert metrics("research_queue_oldest_pending_seconds") >= 120
    assert metrics("research_jobs_expired_leases") == 1
    assert metrics("research_active_workers") == 1


@pytest.mark.asyncio
async def test_metrics_when_empty(clean_tables):

    metrics = Metrics()

    assert metrics("research_queue_oldest_pending_seconds") == 0
    assert metrics("research_active_workers") == 0
    assert metrics("research_agent_iterations_total") == 0
    assert metrics("research_llm_cost_usd_total") == 0

    # Every metric is declared, even before anything is recorded.
    for name, kind in (
        ("research_job_attempts_total", "counter"),
        ("research_tool_calls_total", "counter"),
        ("research_tool_latency_seconds", "histogram"),
        ("research_job_attempt_duration_seconds", "histogram"),
        ("research_llm_tokens_total", "counter"),
    ):
        assert f"# TYPE {name} {kind}" in metrics.text


# -------------------------
# Activity, after real worker runs
# -------------------------


@pytest.mark.asyncio
async def test_metrics_after_a_completed_job(clean_tables, monkeypatch):

    await run_task_through_worker(monkeypatch, FakeAgentLLM)

    metrics = Metrics()

    assert metrics("research_job_attempts_total", outcome="completed") == 1
    assert metrics(
        "research_job_attempt_duration_seconds_count", outcome="completed"
    ) == 1

    # FakeAgentLLM searches once with the (fake) Tavily tool.
    assert metrics(
        "research_tool_calls_total", tool="tavily", outcome="success"
    ) == 1
    assert metrics("research_tool_latency_seconds_count", tool="tavily") == 1

    assert metrics("research_agent_runs", status="completed") == 1
    assert metrics("research_agent_iterations_total") == 2

    # Two LLM calls of 100 input + 50 output tokens.
    assert metrics("research_llm_tokens_total", type="input") == 200
    assert metrics("research_llm_tokens_total", type="output") == 100

    async with TestSessionLocal() as db:
        run = await db.scalar(select(AgentRun))

    if run.estimated_cost_usd is None:
        assert metrics("research_agent_runs_unpriced") == 1
    else:
        assert metrics("research_llm_cost_usd_total") == pytest.approx(
            run.estimated_cost_usd
        )


@pytest.mark.asyncio
async def test_metrics_after_a_failed_job(clean_tables, monkeypatch):

    await run_task_through_worker(monkeypatch, BrokenLLM)

    metrics = Metrics()

    assert metrics("research_job_attempts_total", outcome="failed") == 1
    assert metrics("research_agent_runs", status="failed") == 1


# -------------------------
# Built from rows directly
# -------------------------


async def new_job(db):
    task = await create_task(db)
    return await create_research_job(db=db, task_id=task.id)


@pytest.mark.asyncio
async def test_attempt_outcomes_and_duration_histogram(clean_tables):

    async with TestSessionLocal() as db:

        # Each job claimed directly (claim_next_job would pick the released
        # one again: it's back in the queue).

        # Completed.
        job = await new_job(db)
        await mark_job_running(db=db, job=job, worker_id="worker-1")
        await mark_job_completed(db=db, job=job, lease_token=job.lease_token)
        await db.commit()

        # Released on shutdown.
        job = await new_job(db)
        await mark_job_running(db=db, job=job, worker_id="worker-1")
        await release_job(
            db=db,
            job=job,
            lease_token=job.lease_token,
            reason="Worker shut down",
        )
        await db.commit()

        # Worker died: lease expired, recovered.
        job = await new_job(db)
        await mark_job_running(db=db, job=job, worker_id="worker-dead")
        job.lease_expires_at = utc_now() - timedelta(seconds=1)
        await db.commit()
        await recover_stale_jobs(db)
        await db.commit()

        # Known durations: completed 2s, released 40s, lease_expired 700s.
        attempts = (
            await db.execute(select(JobAttempt).order_by(JobAttempt.id))
        ).scalars().all()

        start = utc_now() - timedelta(hours=1)

        for attempt, seconds in zip(attempts, (2, 40, 700), strict=True):
            attempt.started_at = start
            attempt.ended_at = start + timedelta(seconds=seconds)

        await db.commit()

    metrics = Metrics()

    for outcome in ("completed", "released", "lease_expired"):
        assert metrics("research_job_attempts_total", outcome=outcome) == 1

    # Buckets are cumulative: 2s falls in le=5 and above, 40s from le=60.
    bucket = "research_job_attempt_duration_seconds_bucket"

    assert metrics(bucket, outcome="completed", le="1.0") == 0
    assert metrics(bucket, outcome="completed", le="5.0") == 1
    assert metrics(bucket, outcome="released", le="30.0") == 0
    assert metrics(bucket, outcome="released", le="60.0") == 1
    assert metrics(bucket, outcome="lease_expired", le="600.0") == 0
    assert metrics(bucket, outcome="lease_expired", le="1200.0") == 1
    assert metrics(bucket, outcome="lease_expired", le="+Inf") == 1

    assert metrics(
        "research_job_attempt_duration_seconds_sum", outcome="released"
    ) == pytest.approx(40)


@pytest.mark.asyncio
async def test_tool_calls_and_latency_histogram(clean_tables):

    async with TestSessionLocal() as db:

        job = await new_job(db)

        for event_type, tool, success, duration_ms in (
            ("tool_completed", "tavily", True, 200),
            ("tool_completed", "tavily", True, 3000),
            ("tool_failed", "tavily", False, 15000),
            ("tool_completed", "arxiv", True, 800),
            # Not finished: not counted.
            ("tool_started", "wikipedia", None, None),
        ):
            await record_tool_event(
                db=db,
                job_id=job.id,
                event_type=event_type,
                tool=tool,
                query="RAG",
                success=success,
                duration_ms=duration_ms,
            )

        await db.commit()

    metrics = Metrics()

    calls = "research_tool_calls_total"

    assert metrics(calls, tool="tavily", outcome="success") == 2
    assert metrics(calls, tool="tavily", outcome="failure") == 1
    assert metrics(calls, tool="arxiv", outcome="success") == 1
    assert metrics(calls, tool="wikipedia", outcome="success") == 0

    bucket = "research_tool_latency_seconds_bucket"

    assert metrics(bucket, tool="tavily", le="0.25") == 1
    assert metrics(bucket, tool="tavily", le="5.0") == 2
    assert metrics(bucket, tool="tavily", le="20.0") == 3
    assert metrics("research_tool_latency_seconds_count", tool="tavily") == 3
    assert metrics(
        "research_tool_latency_seconds_sum", tool="tavily"
    ) == pytest.approx(18.2)
