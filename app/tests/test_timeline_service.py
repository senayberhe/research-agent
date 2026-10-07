"""The unified execution timeline: job events, research steps and agent
runs of a task, merged in time order."""

import pytest
import pytest_asyncio

from app.jobs.events import record_job_event
from app.jobs.service import create_research_job
from app.jobs.worker import run_once
from app.schemas.research import ExecutionTimelineResponse
from app.services import workflow_service
from app.services.timeline_service import build_execution_timeline
from app.tests.conftest import TestSessionLocal
from app.tests.fake_agent_llm import FakeAgentLLM
from app.tests.fake_agent_tools import FakeRegistry
from app.tests.test_workflow_service import BrokenLLM, create_task


@pytest.fixture(autouse=True)
def fake_tools_and_health_file(monkeypatch, tmp_path):
    from app.core.config import settings

    monkeypatch.setattr(workflow_service, "ToolRegistry", FakeRegistry)
    monkeypatch.setattr(
        settings,
        "worker_health_file",
        str(tmp_path / "worker-health.json"),
    )


# For tests that build the timeline in the test's own session: rolled
# back afterwards.
@pytest_asyncio.fixture
async def db(db_session):
    return db_session


# Jobs have a foreign key to research_tasks, so they need a real task (the
# test database has no task 1).
@pytest_asyncio.fixture
async def task_id(db) -> int:
    task = await create_task(db)
    return task.id


async def run_task_through_worker(monkeypatch, llm_class) -> int:
    """Creates a task and has a worker run its job; returns the task id."""

    from app.services.research_service import create_research_task

    monkeypatch.setattr(workflow_service, "OpenAIProvider", llm_class)

    async with TestSessionLocal() as db:
        task = await create_research_task(db=db, question="What is RAG?")

    assert await run_once(session_factory=TestSessionLocal, worker_id="worker-1")

    return task.id


async def timeline_for(task_id: int) -> list[dict]:
    async with TestSessionLocal() as db:
        return await build_execution_timeline(db=db, task_id=task_id)


@pytest.mark.asyncio
async def test_timeline_of_a_completed_task(clean_tables, monkeypatch):

    task_id = await run_task_through_worker(monkeypatch, FakeAgentLLM)

    timeline = await timeline_for(task_id)

    assert [(item["source"], item["event_type"]) for item in timeline] == [
        ("job", "created"),
        ("job", "claimed"),
        ("agent", "agent_started"),
        ("step", "research_step"),
        ("job", "tool_started"),
        ("job", "tool_completed"),
        ("agent", "agent_completed"),
        ("job", "completed"),
    ]

    # In time order.
    timestamps = [item["timestamp"] for item in timeline]
    assert timestamps == sorted(timestamps)

    by_type = {item["event_type"]: item for item in timeline}

    step = by_type["research_step"]
    assert step["message"] == "tavily research step"
    assert step["metadata"]["tool"] == "tavily"
    assert step["metadata"]["query"] == "retrieval augmented generation"
    assert step["metadata"]["status"] == "completed"
    assert step["metadata"]["duration_ms"] is not None

    completed = by_type["agent_completed"]
    run_id = completed["metadata"]["run_id"]
    assert completed["message"] == f"Agent run {run_id} completed."
    assert completed["metadata"]["iterations"] == 2
    assert completed["metadata"]["tool_calls"] == 1
    assert completed["metadata"]["total_tokens"] == 300
    assert "estimated_cost" in completed["metadata"]

    started = by_type["agent_started"]
    assert started["message"] == f"Agent run {run_id} started."
    # A completed run's checkpoint is cleared.
    assert started["metadata"]["state"] is None


@pytest.mark.asyncio
async def test_timeline_of_a_failed_run(clean_tables, monkeypatch):

    task_id = await run_task_through_worker(monkeypatch, BrokenLLM)

    timeline = await timeline_for(task_id)

    [failed] = [
        item
        for item in timeline
        if item["source"] == "agent" and item["event_type"] != "agent_started"
    ]

    assert failed["event_type"] == "agent_failed"
    assert failed["message"].endswith("failed: LLM service unavailable")
    assert failed["metadata"]["status"] == "failed"


@pytest.mark.asyncio
async def test_timeline_validates_as_response(clean_tables, monkeypatch):

    task_id = await run_task_through_worker(monkeypatch, FakeAgentLLM)

    response = ExecutionTimelineResponse(
        task_id=task_id,
        events=await timeline_for(task_id),
    )

    assert response.task_id == task_id
    assert {event.source for event in response.events} == {
        "job",
        "step",
        "agent",
    }

    # Serialises (datetimes, enums) without errors.
    assert response.model_dump_json()


@pytest.mark.asyncio
async def test_timeline_of_unknown_task_is_empty(clean_tables):

    assert await timeline_for(999) == []


# -------------------------
# GET /research/{task_id}/timeline
# -------------------------


@pytest.mark.asyncio
async def test_timeline_endpoint(clean_tables, monkeypatch):

    from app.tests.test_research_api import client

    task_id = await run_task_through_worker(monkeypatch, FakeAgentLLM)

    response = client.get(f"/research/{task_id}/timeline")

    assert response.status_code == 200

    body = response.json()

    assert body["task_id"] == task_id
    assert [event["event_type"] for event in body["events"]] == [
        "created",
        "claimed",
        "agent_started",
        "research_step",
        "tool_started",
        "tool_completed",
        "agent_completed",
        "completed",
    ]

    tool_completed = body["events"][5]

    assert tool_completed["source"] == "job"
    assert tool_completed["metadata"]["tool"] == "tavily"
    assert tool_completed["metadata"]["success"] is True
    assert tool_completed["timestamp"]


@pytest.mark.asyncio
async def test_timeline_endpoint_before_any_run(clean_tables):

    from app.services.research_service import create_research_task
    from app.tests.test_research_api import client

    async with TestSessionLocal() as db:
        task = await create_research_task(db=db, question="What is RAG?")

    body = client.get(f"/research/{task.id}/timeline").json()

    # Queued, nothing run yet.
    assert [event["event_type"] for event in body["events"]] == ["created"]


@pytest.mark.asyncio
async def test_timeline_endpoint_missing_task(clean_tables):

    from app.tests.test_research_api import client

    response = client.get("/research/999999/timeline")

    assert response.status_code == 404
    assert response.json()["detail"] == "Research task not found."


# -------------------------
# Job events only
# -------------------------


@pytest.mark.asyncio
async def test_build_execution_timeline(db, task_id):

    job = await create_research_job(
        db=db,
        task_id=task_id,
    )

    # Event types must be JobEventType values ("started" isn't one).
    await record_job_event(
        db=db,
        job_id=job.id,
        event_type="claimed",
        message="Started.",
    )

    await record_job_event(
        db=db,
        job_id=job.id,
        event_type="completed",
        message="Completed.",
    )

    timeline = await build_execution_timeline(
        db=db,
        task_id=task_id,
    )

    assert len(timeline) >= 3

    event_types = [
        event["event_type"]
        for event in timeline
    ]

    assert "created" in event_types
    assert "claimed" in event_types
    assert "completed" in event_types


@pytest.mark.asyncio
async def test_timeline_is_chronological(db, task_id):

    job = await create_research_job(
        db=db,
        task_id=task_id,
    )

    await record_job_event(
        db=db,
        job_id=job.id,
        event_type="claimed",
    )

    await record_job_event(
        db=db,
        job_id=job.id,
        event_type="completed",
    )

    timeline = await build_execution_timeline(
        db=db,
        task_id=task_id,
    )

    timestamps = [
        event["timestamp"]
        for event in timeline
    ]

    assert timestamps == sorted(timestamps)

    # Recorded order kept.
    assert [event["event_type"] for event in timeline] == [
        "created",
        "claimed",
        "completed",
    ]

