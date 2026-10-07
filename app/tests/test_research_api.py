import httpx
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.db.database import get_db
from app.db.models import (
    AgentRun,
    JobStatus,
    ResearchJob,
    ResearchTask,
    TaskStatus,
)
from app.jobs.events import record_job_event
from app.jobs.service import get_jobs_for_task
from app.main import app
from app.services.research_service import create_research_task
from app.tests.conftest import TestSessionLocal


async def override_get_db():
    async with TestSessionLocal() as session:
        yield session


app.dependency_overrides[get_db] = override_get_db


client = TestClient(app)


def test_health():

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] in ("healthy", "degraded")
    assert body["database"]["status"] == "healthy"
    assert set(body) == {"status", "database", "workers", "jobs"}


def test_create_research_task():

    response = client.post(
        "/research",
        json={
            "question": "How does RAG work?"
        },
    )

    assert response.status_code == 202

    data = response.json()

    assert data["id"] > 0
    assert data["question"] == (
        "How does RAG work?"
    )
    assert data["status"] == "pending"
    assert "created_at" in data


def test_get_research_task():

    create_response = client.post(
        "/research",
        json={
            "question": (
                "What is machine learning?"
            )
        },
    )

    assert create_response.status_code == 202

    task_id = create_response.json()["id"]

    response = client.get(
        f"/research/{task_id}"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["id"] == task_id
    assert data["question"] == (
        "What is machine learning?"
    )


def test_get_missing_research_task():

    response = client.get(
        "/research/999999"
    )

    assert response.status_code == 404

    assert response.json() == {
        "detail": "Research task not found"
    }


def test_create_research_task_without_question():

    response = client.post(
        "/research",
        json={}
    )

    assert response.status_code == 422


def test_create_research_task_with_invalid_body():

    response = client.post(
        "/research",
        json={
            "question": 123
        },
    )

    assert response.status_code == 422

# -------------------------
# POST /research/{task_id}/resume
# -------------------------


async def add_task_and_run(
    task_status,
    run_status=None,
    state=None,
    job_status=None,
) -> int:

    async with TestSessionLocal() as db:

        task = ResearchTask(question="What is RAG?", status=task_status)
        db.add(task)
        await db.flush()

        if run_status is not None:
            db.add(AgentRun(task_id=task.id, status=run_status, state=state))

        if job_status is not None:
            db.add(ResearchJob(task_id=task.id, status=job_status, attempts=1))

        await db.commit()

        return task.id


def resumable_state() -> dict:

    from app.agents.research_agent import AgentCheckpoint

    return AgentCheckpoint(
        iteration=2,
        input_items=[{"role": "user", "content": "What is RAG?"}],
        tool_call_count=2,
        input_tokens=200,
        output_tokens=100,
        total_tokens=300,
        estimated_cost=None,
        executed_tool_calls=2,
    ).to_dict()


async def jobs_for(task_id: int) -> list[ResearchJob]:

    async with TestSessionLocal() as db:
        result = await db.execute(
            select(ResearchJob)
            .where(ResearchJob.task_id == task_id)
            .order_by(ResearchJob.id)
        )
        return list(result.scalars().all())


@pytest.mark.asyncio
async def test_resume_requeues_failed_job(clean_tables):

    task_id = await add_task_and_run(
        "failed",
        "failed",
        resumable_state(),
        job_status="failed",
    )

    response = client.post(f"/research/{task_id}/resume")

    assert response.status_code == 202
    assert response.json()["id"] == task_id

    # The same job, back in the queue for a worker (attempts kept).
    [job] = await jobs_for(task_id)

    assert job.status == JobStatus.PENDING
    assert job.attempts == 1


@pytest.mark.asyncio
async def test_resume_task_without_job_queues_new_job(clean_tables):

    # A task from before jobs existed.
    task_id = await add_task_and_run("failed", "failed", resumable_state())

    response = client.post(f"/research/{task_id}/resume")

    assert response.status_code == 202

    [job] = await jobs_for(task_id)

    assert job.status == JobStatus.PENDING


@pytest.mark.asyncio
async def test_resume_already_queued_is_left_alone(clean_tables):

    task_id = await add_task_and_run(
        "failed",
        "failed",
        resumable_state(),
        job_status="pending",
    )

    response = client.post(f"/research/{task_id}/resume")

    # Fine to ask twice: still exactly one queued job.
    assert response.status_code == 202

    [job] = await jobs_for(task_id)

    assert job.status == JobStatus.PENDING


@pytest.mark.asyncio
async def test_resume_refused_while_job_running(clean_tables):

    task_id = await add_task_and_run(
        "failed",
        "failed",
        resumable_state(),
        job_status="running",
    )

    response = client.post(f"/research/{task_id}/resume")

    assert response.status_code == 409
    assert "already being run" in response.json()["detail"]


def test_resume_missing_task():

    response = client.post("/research/999999/resume")

    assert response.status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("task_status", "run_status", "with_state", "reason"),
    [
        ("pending", None, False, "only failed tasks can be resumed"),
        ("completed", "completed", False, "only failed tasks can be resumed"),
        ("failed", None, False, "has no agent run to resume"),
        ("failed", "completed", False, "only failed runs can be resumed"),
        ("failed", "failed", False, "no checkpoint to resume"),
    ],
)
async def test_resume_refused_with_reason(
    clean_tables,
    task_status,
    run_status,
    with_state,
    reason,
):

    task_id = await add_task_and_run(
        task_status,
        run_status,
        resumable_state() if with_state else None,
        job_status="failed",
    )

    response = client.post(f"/research/{task_id}/resume")

    assert response.status_code == 409
    assert reason in response.json()["detail"]

    # Nothing queued: the job is still failed.
    [job] = await jobs_for(task_id)

    assert job.status == JobStatus.FAILED


# -------------------------
# Tasks are queued as durable jobs
# -------------------------


@pytest.mark.asyncio
async def test_create_research_task_creates_pending_job(clean_tables):

    # Through the real API: the request commits, and the job is read back
    # with a separate session.
    response = client.post(
        "/research",
        json={"question": "How does RAG work?"},
    )

    assert response.status_code == 202

    task_id = response.json()["id"]

    async with TestSessionLocal() as db:

        task = await db.get(ResearchTask, task_id)

        result = await db.execute(
            select(ResearchJob).where(
                ResearchJob.task_id == task.id
            )
        )

        job = result.scalar_one()

        assert job.task_id == task.id
        assert job.status == JobStatus.PENDING
        assert job.attempts == 0

        # The task waits for a worker: still pending, nothing run yet.
        assert task.status == TaskStatus.PENDING
        assert job.worker_id is None

        runs = await db.execute(
            select(AgentRun).where(AgentRun.task_id == task.id)
        )

        assert runs.scalars().all() == []


# -------------------------------------------------------------------
# Jobs, events and execution (async client)
# -------------------------------------------------------------------


# Registered as "client" for the tests below; the function has another
# name so it doesn't replace the module-level TestClient the tests above
# use.
@pytest_asyncio.fixture(name="client")
async def async_client():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as async_client:
        yield async_client


# A real session: the API reads through its own sessions, which only see
# committed data. clean_tables empties the tables afterwards.
@pytest_asyncio.fixture
async def db(clean_tables):
    async with TestSessionLocal() as session:
        yield session


@pytest.mark.asyncio
async def test_get_task_jobs(
    client,
    db,
):
    task = await create_research_task(
        db=db,
        question="What is RAG?",
    )

    await db.commit()

    response = await client.get(
        f"/research/{task.id}/jobs"
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 1
    assert data[0]["task_id"] == task.id
    assert data[0]["status"] == "pending"


@pytest.mark.asyncio
async def test_get_job_events(
    client,
    db,
):
    task = await create_research_task(
        db=db,
        question="What is RAG?",
    )

    job = await get_jobs_for_task(
        db=db,
        task_id=task.id,
    )

    # "claimed": event types must be JobEventType values.
    await record_job_event(
        db=db,
        job_id=job[0].id,
        event_type="claimed",
        message="Agent started.",
    )

    await db.commit()

    response = await client.get(
        f"/research/{task.id}/jobs/{job[0].id}/events"
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data) >= 2
    assert data[0]["event_type"] == "created"
    assert data[1]["event_type"] == "claimed"
    assert data[1]["message"] == "Agent started."


@pytest.mark.asyncio
async def test_job_cannot_be_accessed_through_wrong_task(
    client,
    db,
):
    task_a = await create_research_task(
        db=db,
        question="Question A",
    )

    task_b = await create_research_task(
        db=db,
        question="Question B",
    )

    jobs_b = await get_jobs_for_task(
        db=db,
        task_id=task_b.id,
    )

    await db.commit()

    response = await client.get(
        f"/research/{task_a.id}/jobs/"
        f"{jobs_b[0].id}/events"
    )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_execution_for_missing_task(
    client,
):
    response = await client.get(
        "/research/999999/execution"
    )

    assert response.status_code == 404



@pytest.mark.asyncio
async def test_get_research_timeline(
    client,
    db,
):

    task = await create_research_task(
        db=db,
        question="What is RAG?",
    )

    await db.commit()

    response = await client.get(
        f"/research/{task.id}/timeline"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["task_id"] == task.id
    assert "events" in data
    assert len(data["events"]) >= 1


@pytest.mark.asyncio
async def test_get_research_metrics(
    client,
    db,
):

    task = await create_research_task(
        db=db,
        question="What is RAG?",
    )

    await db.commit()

    response = await client.get(
        f"/research/{task.id}/metrics"
    )

    assert response.status_code == 200

    data = response.json()

    # Queued, not run yet.
    assert data["task_id"] == task.id
    assert data["total_jobs"] == 1
    assert data["completed_jobs"] == 0
    assert data["total_tool_calls"] == 0
    assert data["total_tokens"] == 0
    assert data["success_rate"] == 0.0


@pytest.mark.asyncio
async def test_metrics_for_missing_task(
    client,
):
    response = await client.get(
        "/research/999999/metrics"
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Research task not found."


def test_get_system_metrics():

    response = client.get("/research/metrics")

    assert response.status_code == 200

    data = response.json()

    assert "total_jobs" in data
    assert "completed_jobs" in data
    assert "failed_jobs" in data
    assert "total_tool_calls" in data
    assert "total_tokens" in data
    assert "estimated_cost" in data
    assert "success_rate" in data


def test_metrics_routes():
    """JSON metrics for applications under /research/metrics; Prometheus
    text for monitoring systems at /metrics."""

    summary = client.get("/research/metrics")
    system = client.get("/research/metrics/system")
    prometheus = client.get("/metrics")

    for response in (summary, system, prometheus):
        assert response.status_code == 200

    assert summary.headers["content-type"] == "application/json"
    assert "total_jobs" in summary.json()

    assert system.headers["content-type"] == "application/json"
    assert {"summary", "jobs", "agents", "tools", "queue"} <= set(system.json())

    assert prometheus.headers["content-type"].startswith("text/plain")
    assert "# TYPE research_jobs gauge" in prometheus.text
