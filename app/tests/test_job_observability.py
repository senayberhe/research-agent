"""Job attempt history (job_attempts), job events (job_events) and the read
endpoints for jobs, workers, agent runs and tool calls."""

import json
from datetime import timedelta

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.db.database import get_db
from app.db.models import (
    AgentRun,
    AgentRunStatus,
    AttemptOutcome,
    JobEventType,
    JobStatus,
    ResearchResult,
    ResearchStep,
    StepStatus,
    utc_now,
)
from app.jobs.service import (
    MAX_ATTEMPTS_ERROR,
    STALE_LEASE_ERROR,
    claim_next_job,
    create_research_job,
    get_worker_activity,
    list_job_attempts,
    mark_job_completed,
    mark_job_failed,
    recover_stale_jobs,
    release_job,
    requeue_job,
)
from app.jobs.event_service import get_job_events
from app.main import app
from app.tests.conftest import TestSessionLocal
from app.tests.test_workflow_service import create_task


async def override_get_db():
    async with TestSessionLocal() as session:
        yield session


app.dependency_overrides[get_db] = override_get_db


client = TestClient(app)


@pytest_asyncio.fixture
async def db(db_session):
    return db_session


@pytest_asyncio.fixture
async def task_id(db) -> int:
    task = await create_task(db)
    return task.id


async def expire_lease(db, job) -> None:
    job.lease_expires_at = utc_now() - timedelta(seconds=1)
    await db.commit()


# -------------------------
# Attempt history
# -------------------------


@pytest.mark.asyncio
async def test_claim_starts_an_attempt(db, task_id):

    job = await create_research_job(db=db, task_id=task_id)

    await claim_next_job(db=db, worker_id="worker-1")

    [attempt] = await list_job_attempts(db=db, job_id=job.id)

    assert attempt.attempt_number == 1
    assert attempt.worker_id == "worker-1"
    assert attempt.outcome == AttemptOutcome.RUNNING
    assert attempt.ended_at is None


@pytest.mark.asyncio
async def test_retry_keeps_the_earlier_failure(db, task_id):

    job = await create_research_job(db=db, task_id=task_id)

    await claim_next_job(db=db, worker_id="worker-1")
    await mark_job_failed(
        db=db,
        job=job,
        error="Tavily timed out",
        lease_token=job.lease_token,
    )
    await db.commit()

    await requeue_job(db=db, job=job)

    await claim_next_job(db=db, worker_id="worker-2")
    await mark_job_completed(db=db, job=job, lease_token=job.lease_token)
    await db.commit()

    first, second = await list_job_attempts(db=db, job_id=job.id)

    # The job only shows the latest attempt...
    assert job.status == JobStatus.COMPLETED
    assert job.error is None
    assert job.attempts == 2

    # ...the history keeps both.
    assert first.attempt_number == 1
    assert first.worker_id == "worker-1"
    assert first.outcome == AttemptOutcome.FAILED
    assert first.error == "Tavily timed out"
    assert first.ended_at is not None

    assert second.attempt_number == 2
    assert second.worker_id == "worker-2"
    assert second.outcome == AttemptOutcome.COMPLETED
    assert second.error is None
    assert second.ended_at is not None


@pytest.mark.asyncio
async def test_stale_recovery_ends_attempt_as_lease_expired(db, task_id):

    job = await create_research_job(db=db, task_id=task_id)

    await claim_next_job(db=db, worker_id="worker-1")
    await expire_lease(db, job)

    await recover_stale_jobs(db)
    await db.commit()

    [attempt] = await list_job_attempts(db=db, job_id=job.id)

    assert attempt.outcome == AttemptOutcome.LEASE_EXPIRED
    assert attempt.error == STALE_LEASE_ERROR
    assert attempt.ended_at is not None


@pytest.mark.asyncio
async def test_release_ends_attempt_as_released(db, task_id):

    job = await create_research_job(db=db, task_id=task_id)

    await claim_next_job(db=db, worker_id="worker-1")

    await release_job(
        db=db,
        job=job,
        lease_token=job.lease_token,
        reason="Worker shut down",
    )
    await db.commit()

    [attempt] = await list_job_attempts(db=db, job_id=job.id)

    assert attempt.outcome == AttemptOutcome.RELEASED
    assert attempt.error == "Worker shut down"


@pytest.mark.asyncio
async def test_wrong_lease_leaves_attempt_running(db, task_id):

    job = await create_research_job(db=db, task_id=task_id)

    await claim_next_job(db=db, worker_id="worker-1")

    assert not await mark_job_completed(db=db, job=job, lease_token="stale")

    [attempt] = await list_job_attempts(db=db, job_id=job.id)

    assert attempt.outcome == AttemptOutcome.RUNNING
    assert attempt.ended_at is None


# -------------------------
# Events
# -------------------------


def timeline(events) -> list[tuple]:
    return [
        (JobEventType(event.event_type), event.attempt, event.worker_id)
        for event in events
    ]


@pytest.mark.asyncio
async def test_events_record_a_retried_job(db, task_id):

    job = await create_research_job(db=db, task_id=task_id)

    await claim_next_job(db=db, worker_id="worker-1")
    await mark_job_failed(
        db=db,
        job=job,
        error="Tavily timed out",
        lease_token=job.lease_token,
    )
    await db.commit()

    await requeue_job(db=db, job=job)

    await claim_next_job(db=db, worker_id="worker-2")
    await mark_job_completed(db=db, job=job, lease_token=job.lease_token)
    await db.commit()

    events = await get_job_events(db=db, job_id=job.id)

    assert timeline(events) == [
        (JobEventType.CREATED, 0, None),
        (JobEventType.CLAIMED, 1, "worker-1"),
        (JobEventType.FAILED, 1, "worker-1"),
        (JobEventType.REQUEUED, 1, None),
        (JobEventType.CLAIMED, 2, "worker-2"),
        (JobEventType.COMPLETED, 2, "worker-2"),
    ]

    assert events[2].message == "Tavily timed out"


@pytest.mark.asyncio
async def test_events_record_lease_expiry_and_max_attempts(
    db,
    task_id,
    monkeypatch,
):

    from app.core.config import settings

    monkeypatch.setattr(settings, "worker_max_attempts", 1)

    job = await create_research_job(db=db, task_id=task_id)

    await claim_next_job(db=db, worker_id="worker-1")
    await expire_lease(db, job)

    await recover_stale_jobs(db)
    await db.commit()

    events = await get_job_events(db=db, job_id=job.id)

    assert timeline(events)[-2:] == [
        (JobEventType.LEASE_EXPIRED, 1, "worker-1"),
        (JobEventType.FAILED, 1, None),
    ]

    assert events[-2].message == STALE_LEASE_ERROR
    assert events[-1].message == MAX_ATTEMPTS_ERROR


@pytest.mark.asyncio
async def test_events_record_release(db, task_id):

    job = await create_research_job(db=db, task_id=task_id)

    await claim_next_job(db=db, worker_id="worker-1")
    await release_job(
        db=db,
        job=job,
        lease_token=job.lease_token,
        reason="Worker shut down",
    )
    await db.commit()

    [*_, released] = await get_job_events(db=db, job_id=job.id)

    assert released.event_type == JobEventType.RELEASED
    assert released.worker_id == "worker-1"
    assert released.message == "Worker shut down"


@pytest.mark.asyncio
async def test_record_job_event(db, task_id):

    from app.jobs.events import record_job_event

    job = await create_research_job(db=db, task_id=task_id)

    # A plain string type works; metadata is stored as JSON.
    event = await record_job_event(
        db=db,
        job_id=job.id,
        event_type="failed",
        message="Out of credits",
        metadata={"attempt": 1, "provider": "openai"},
    )

    await db.refresh(event)

    assert event.id is not None
    assert event.event_type == JobEventType.FAILED
    assert event.message == "Out of credits"
    assert json.loads(event.metadata_json) == {
        "attempt": 1,
        "provider": "openai",
    }
    assert event.attempt == 1
    assert event.details["provider"] == "openai"


@pytest.mark.asyncio
async def test_record_job_event_without_metadata(db, task_id):

    from app.jobs.events import record_job_event

    job = await create_research_job(db=db, task_id=task_id)

    event = await record_job_event(db=db, job_id=job.id, event_type="requeued")

    assert event.metadata_json is None
    assert event.details == {}
    assert event.attempt is None


@pytest.mark.asyncio
async def test_record_tool_event(db, task_id):

    from app.jobs.events import record_tool_event

    job = await create_research_job(db=db, task_id=task_id)

    started = await record_tool_event(
        db=db,
        job_id=job.id,
        event_type="tool_started",
        tool="tavily",
        query="retrieval augmented generation",
    )

    completed = await record_tool_event(
        db=db,
        job_id=job.id,
        event_type="tool_completed",
        tool="tavily",
        query="retrieval augmented generation",
        success=True,
        duration_ms=842,
    )

    # Before the search finishes: only the tool and query.
    assert started.details == {
        "tool": "tavily",
        "query": "retrieval augmented generation",
    }

    assert completed.message == "tavily tool event."
    assert completed.details == {
        "tool": "tavily",
        "query": "retrieval augmented generation",
        "success": True,
        "duration_ms": 842,
    }
    assert completed.tool == "tavily"
    assert completed.duration_ms == 842


@pytest.mark.asyncio
async def test_record_job_event_rejects_unknown_type(db, task_id):

    from app.jobs.events import record_job_event

    job = await create_research_job(db=db, task_id=task_id)

    with pytest.raises(ValueError):
        await record_job_event(db=db, job_id=job.id, event_type="complted")


def test_parse_event_metadata():

    from app.db.models import JobEvent
    from app.jobs.event_service import parse_event_metadata

    assert parse_event_metadata(
        JobEvent(metadata_json='{"tool": "tavily", "duration_ms": 842}')
    ) == {"tool": "tavily", "duration_ms": 842}

    # Nothing stored, or not valid JSON: empty, not an error.
    assert parse_event_metadata(JobEvent(metadata_json=None)) == {}
    assert parse_event_metadata(JobEvent(metadata_json="")) == {}
    assert parse_event_metadata(JobEvent(metadata_json="{not json")) == {}

    assert JobEvent(metadata_json="{not json").details == {}


@pytest.mark.asyncio
async def test_bad_metadata_does_not_break_job_endpoints(clean_tables):

    from app.db.models import JobEvent

    _, job_id = await seed_failed_then_completed_job()

    async with TestSessionLocal() as db:
        db.add(
            JobEvent(
                job_id=job_id,
                event_type="tool_completed",
                metadata_json="{not json",
            )
        )
        await db.commit()

    assert client.get(f"/jobs/{job_id}").status_code == 200
    assert client.get(f"/jobs/{job_id}/execution").status_code == 200

    timeline = client.get(f"/jobs/{job_id}/timeline")

    assert timeline.status_code == 200
    assert timeline.text.endswith("completed\ntool_completed\n")


def test_format_job_timeline():

    from app.db.models import JobEvent
    from app.jobs.event_service import format_job_timeline

    def event(event_type, **details):
        return JobEvent(
            event_type=event_type,
            message=details.pop("message", None),
            metadata_json=json.dumps(details) if details else None,
        )

    events = [
        event("tool_started", tool="tavily"),
        event("tool_completed", tool="tavily", duration_ms=842.4),
        event("tool_started", tool="arxiv"),
        event("tool_completed", tool="arxiv", duration_ms=1204.0),
        event("tool_started", tool="wikipedia"),
        event(
            "tool_failed",
            tool="wikipedia",
            duration_ms=5000.0,
            error="timed out",
        ),
        event("completed", worker_id="worker-1"),
    ]

    assert format_job_timeline(events) == (
        "tool_started     → Tavily\n"
        "tool_completed   → Tavily, 842 ms\n"
        "\n"
        "tool_started     → arXiv\n"
        "tool_completed   → arXiv, 1204 ms\n"
        "\n"
        "tool_started     → Wikipedia\n"
        "tool_failed      → Wikipedia, 5000 ms: timed out\n"
        "\n"
        "completed\n"
    )


@pytest.mark.asyncio
async def test_get_job_timeline(clean_tables):

    _, job_id = await seed_failed_then_completed_job()

    response = client.get(f"/jobs/{job_id}/timeline")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert response.text == (
        "created\n"
        "claimed          → worker-1\n"
        "failed           → Tavily timed out\n"
        "requeued\n"
        "claimed          → worker-2\n"
        "completed\n"
    )

    assert client.get("/jobs/999/timeline").status_code == 404


@pytest.mark.asyncio
async def test_job_events_relationship(db, task_id):

    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    from app.db.models import ResearchJob

    job = await create_research_job(db=db, task_id=task_id)
    await claim_next_job(db=db, worker_id="worker-1")

    loaded = await db.scalar(
        select(ResearchJob)
        .where(ResearchJob.id == job.id)
        .options(selectinload(ResearchJob.events))
        .execution_options(populate_existing=True)
    )

    assert [JobEventType(event.event_type) for event in loaded.events] == [
        JobEventType.CREATED,
        JobEventType.CLAIMED,
    ]
    assert loaded.events[0].job is loaded


# -------------------------
# Worker activity
# -------------------------


@pytest.mark.asyncio
async def test_worker_activity(db):

    since = utc_now() - timedelta(hours=1)

    # worker-busy: running a job under a live lease.
    busy_job = await create_research_job(
        db=db,
        task_id=(await create_task(db)).id,
    )
    await claim_next_job(db=db, worker_id="worker-busy")

    # worker-dead: running a job whose lease has expired.
    dead_job = await create_research_job(
        db=db,
        task_id=(await create_task(db)).id,
    )
    await claim_next_job(db=db, worker_id="worker-dead")
    await expire_lease(db, dead_job)

    # worker-idle: finished its job.
    done_job = await create_research_job(
        db=db,
        task_id=(await create_task(db)).id,
    )
    await claim_next_job(db=db, worker_id="worker-idle")
    await mark_job_completed(
        db=db,
        job=done_job,
        lease_token=done_job.lease_token,
    )
    await db.commit()

    workers = {
        worker["worker_id"]: worker
        for worker in await get_worker_activity(db=db, since=since)
    }

    assert workers["worker-busy"]["state"] == "busy"
    assert workers["worker-busy"]["current_job_id"] == busy_job.id

    assert workers["worker-dead"]["state"] == "unresponsive"
    assert workers["worker-dead"]["current_job_id"] == dead_job.id

    assert workers["worker-idle"]["state"] == "idle"
    assert workers["worker-idle"]["current_job_id"] is None
    assert workers["worker-idle"]["attempts"] == {"completed": 1}

    # Nothing in a window that ends before the attempts started.
    assert await get_worker_activity(db=db, since=utc_now()) == []


# -------------------------
# API
# -------------------------


async def seed_failed_then_completed_job() -> tuple[int, int]:
    """A task whose job failed once and then completed, with an agent run
    and two tool calls (one failed). Returns (task_id, job_id)."""

    async with TestSessionLocal() as db:

        task = await create_task(db)
        job = await create_research_job(db=db, task_id=task.id)

        await claim_next_job(db=db, worker_id="worker-1")
        await mark_job_failed(
            db=db,
            job=job,
            error="Tavily timed out",
            lease_token=job.lease_token,
        )
        await db.commit()

        await requeue_job(db=db, job=job)
        await claim_next_job(db=db, worker_id="worker-2")
        await mark_job_completed(db=db, job=job, lease_token=job.lease_token)

        started = utc_now() - timedelta(seconds=90)

        db.add(
            AgentRun(
                task_id=task.id,
                status=AgentRunStatus.COMPLETED,
                iteration_count=3,
                tool_call_count=2,
                input_tokens=1200,
                output_tokens=300,
                total_tokens=1500,
                estimated_cost_usd=0.0123,
                started_at=started,
                completed_at=started + timedelta(seconds=90),
            )
        )

        ok = ResearchStep(
            task_id=task.id,
            tool="wikipedia",
            query="RAG",
            iteration=1,
            status=StepStatus.COMPLETED,
            duration_ms=120.0,
        )
        failed = ResearchStep(
            task_id=task.id,
            tool="tavily",
            query="RAG factuality",
            iteration=2,
            status=StepStatus.FAILED,
            duration_ms=5000.0,
        )
        db.add_all([ok, failed])
        await db.flush()

        db.add_all(
            [
                ResearchResult(
                    step_id=ok.id,
                    tool="wikipedia",
                    query="RAG",
                    content="Retrieval-augmented generation is...",
                ),
                ResearchResult(
                    step_id=failed.id,
                    tool="tavily",
                    query="RAG factuality",
                    content="",
                    success=False,
                    error="timed out",
                ),
            ]
        )

        await db.commit()

        return task.id, job.id


@pytest.mark.asyncio
async def test_get_job_shows_attempt_history(clean_tables):

    _, job_id = await seed_failed_then_completed_job()

    response = client.get(f"/jobs/{job_id}")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "completed"
    assert body["attempts"] == 2
    assert body["error"] is None

    first, second = body["attempt_history"]

    assert first["worker_id"] == "worker-1"
    assert first["outcome"] == "failed"
    assert first["error"] == "Tavily timed out"
    assert first["duration_seconds"] is not None

    assert second["worker_id"] == "worker-2"
    assert second["outcome"] == "completed"

    assert [event["event_type"] for event in body["events"]] == [
        "created",
        "claimed",
        "failed",
        "requeued",
        "claimed",
        "completed",
    ]
    assert body["events"][2]["message"] == "Tavily timed out"
    assert body["events"][1]["summary"] == "worker-1"


@pytest.mark.asyncio
async def test_get_job_execution(clean_tables):

    task_id, job_id = await seed_failed_then_completed_job()

    response = client.get(f"/jobs/{job_id}/execution")

    assert response.status_code == 200

    body = response.json()

    assert body["job_id"] == job_id
    assert body["task_id"] == task_id

    # Current state.
    assert body["current_state"]["status"] == "completed"
    assert body["current_state"]["attempts"] == 2
    assert body["current_state"]["worker_id"] == "worker-2"

    # History.
    assert [event["event_type"] for event in body["history"]] == [
        "created",
        "claimed",
        "failed",
        "requeued",
        "claimed",
        "completed",
    ]

    # The agent run: iterations, tokens, tools, cost.
    run = body["agent_run"]

    assert run["status"] == "completed"
    assert run["iterations"] == 3
    assert run["duration_seconds"] == pytest.approx(90)
    assert run["tokens"] == {"input": 1200, "output": 300, "total": 1500}
    assert run["cost"]["estimated_usd"] == pytest.approx(0.0123)
    assert run["tools"]["requested"] == 2
    assert [call["tool"] for call in run["tools"]["calls"]] == [
        "wikipedia",
        "tavily",
    ]

    # Completed runs clear their checkpoint.
    assert run["checkpoint"] is None


@pytest.mark.asyncio
async def test_job_execution_shows_checkpoint(clean_tables):

    from app.tests.test_research_api import resumable_state

    _, job_id = await seed_failed_then_completed_job()

    async with TestSessionLocal() as db:
        run = await db.scalar(
            select(AgentRun)
        )
        run.status = AgentRunStatus.FAILED
        run.state = resumable_state()
        run.state_updated_at = utc_now()
        await db.commit()

    run = client.get(f"/jobs/{job_id}/execution").json()["agent_run"]

    assert run["status"] == "failed"
    assert run["tools"]["executed"] == 2
    assert run["checkpoint"]["valid"] is True
    assert run["checkpoint"]["iteration"] == 2
    assert run["checkpoint"]["conversation_items"] == 1
    assert run["checkpoint"]["saved_at"] is not None

    # The conversation itself isn't sent.
    assert "input_items" not in run["checkpoint"]


@pytest.mark.asyncio
async def test_job_execution_reports_corrupt_checkpoint(clean_tables):

    _, job_id = await seed_failed_then_completed_job()

    async with TestSessionLocal() as db:
        run = await db.scalar(
            select(AgentRun)
        )
        run.state = {"version": 99}
        await db.commit()

    checkpoint = client.get(
        f"/jobs/{job_id}/execution"
    ).json()["agent_run"]["checkpoint"]

    assert checkpoint["valid"] is False
    assert "version" in checkpoint["error"]


@pytest.mark.asyncio
async def test_job_execution_before_any_run(clean_tables):

    async with TestSessionLocal() as db:
        task = await create_task(db)
        job = await create_research_job(db=db, task_id=task.id)

    body = client.get(f"/jobs/{job.id}/execution").json()

    assert body["current_state"]["status"] == "pending"
    assert [event["event_type"] for event in body["history"]] == ["created"]
    assert body["agent_run"] is None


@pytest.mark.asyncio
async def test_job_execution_missing_job_is_404(clean_tables):

    assert client.get("/jobs/999/execution").status_code == 404


@pytest.mark.asyncio
async def test_get_missing_job_is_404(clean_tables):

    assert client.get("/jobs/999").status_code == 404


@pytest.mark.asyncio
async def test_get_task_jobs(clean_tables):

    task_id, job_id = await seed_failed_then_completed_job()

    response = client.get(f"/research/{task_id}/jobs")

    assert response.status_code == 200
    [job] = response.json()

    assert job["id"] == job_id

    # The history is at /jobs/{id}, not in the list.
    assert "events" not in job


@pytest.mark.asyncio
async def test_get_research_execution(clean_tables):

    task_id, job_id = await seed_failed_then_completed_job()

    response = client.get(f"/research/{task_id}/execution")

    assert response.status_code == 200

    body = response.json()

    assert body["task"]["id"] == task_id
    assert body["task"]["question"].startswith("How does retrieval")
    assert body["task"]["status"] == "pending"

    [job] = body["jobs"]

    assert job["id"] == job_id
    assert job["task_id"] == task_id
    assert job["status"] == "completed"
    assert job["attempts"] == 2
    assert job["worker_id"] == "worker-2"


@pytest.mark.asyncio
async def test_research_execution_missing_task_is_404(clean_tables):

    response = client.get("/research/999/execution")

    assert response.status_code == 404
    assert response.json()["detail"] == "Research task not found."


@pytest.mark.asyncio
async def test_get_job_event_history(clean_tables):

    task_id, job_id = await seed_failed_then_completed_job()

    response = client.get(f"/research/{task_id}/jobs/{job_id}/events")

    assert response.status_code == 200

    events = response.json()

    assert [event["event_type"] for event in events] == [
        "created",
        "claimed",
        "failed",
        "requeued",
        "claimed",
        "completed",
    ]

    claimed = events[1]

    assert claimed["job_id"] == job_id
    assert claimed["metadata"] == {"worker_id": "worker-1", "attempt": 1}
    assert events[2]["message"] == "Tavily timed out"
    assert set(claimed) == {
        "id",
        "job_id",
        "event_type",
        "message",
        "metadata",
        "created_at",
    }


@pytest.mark.asyncio
async def test_job_event_history_404s(clean_tables):

    task_id, job_id = await seed_failed_then_completed_job()
    other_task_id, _ = await seed_failed_then_completed_job()

    # No such job.
    assert client.get(
        f"/research/{task_id}/jobs/999/events"
    ).status_code == 404

    # A real job, but another task's.
    assert client.get(
        f"/research/{other_task_id}/jobs/{job_id}/events"
    ).status_code == 404


@pytest.mark.asyncio
async def test_get_task_runs_shows_tokens_cost_and_duration(clean_tables):

    task_id, _ = await seed_failed_then_completed_job()

    response = client.get(f"/research/{task_id}/runs")

    assert response.status_code == 200

    [run] = response.json()

    assert run["status"] == "completed"
    assert run["tool_call_count"] == 2
    assert run["input_tokens"] == 1200
    assert run["output_tokens"] == 300
    assert run["total_tokens"] == 1500
    assert run["estimated_cost_usd"] == pytest.approx(0.0123)
    assert run["duration_seconds"] == pytest.approx(90)

    # The checkpoint (the whole conversation) isn't sent.
    assert "state" not in run


@pytest.mark.asyncio
async def test_get_task_steps_shows_tool_calls(clean_tables):

    task_id, _ = await seed_failed_then_completed_job()

    response = client.get(f"/research/{task_id}/steps")

    assert response.status_code == 200

    ok, failed = response.json()

    assert ok["tool"] == "wikipedia"
    assert ok["status"] == "completed"
    assert ok["duration_ms"] == 120.0
    assert ok["results"][0]["success"] is True

    assert failed["tool"] == "tavily"
    assert failed["status"] == "failed"
    assert failed["results"][0]["error"] == "timed out"

    # Result content can be long; it isn't sent.
    assert "content" not in ok["results"][0]


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["jobs", "runs", "steps"])
async def test_task_endpoints_404_for_missing_task(clean_tables, path):

    assert client.get(f"/research/999/{path}").status_code == 404


@pytest.mark.asyncio
async def test_list_workers(clean_tables):

    await seed_failed_then_completed_job()

    response = client.get("/workers")

    assert response.status_code == 200

    workers = {worker["worker_id"]: worker for worker in response.json()}

    assert workers["worker-1"]["state"] == "idle"
    assert workers["worker-1"]["attempts"] == {"failed": 1}

    assert workers["worker-2"]["state"] == "idle"
    assert workers["worker-2"]["attempts"] == {"completed": 1}


@pytest.mark.asyncio
async def test_list_workers_rejects_bad_window(clean_tables):

    assert client.get("/workers?hours=0").status_code == 422
