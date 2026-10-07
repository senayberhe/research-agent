"""Tool events: record_tool_event stores the search's details as JSON in
metadata_json, and parse_event_metadata reads them back."""

import json

import pytest
import pytest_asyncio

from app.jobs.event_service import parse_event_metadata
from app.jobs.events import record_tool_event
from app.jobs.service import create_research_job
from app.tests.test_workflow_service import create_task


@pytest_asyncio.fixture
async def db(db_session):
    return db_session


# Jobs have a foreign key to research_tasks, so they need a real task (the
# test database has no task 1).
@pytest_asyncio.fixture
async def task_id(db) -> int:
    task = await create_task(db)
    return task.id


@pytest.mark.asyncio
async def test_tool_event_stores_metadata(db, task_id):

    job = await create_research_job(
        db=db,
        task_id=task_id,
    )

    event = await record_tool_event(
        db=db,
        job_id=job.id,
        event_type="tool_completed",
        tool="tavily",
        query="RAG research",
        success=True,
        duration_ms=845.5,
    )

    assert event.event_type == "tool_completed"
    assert event.metadata_json is not None

    metadata = json.loads(
        event.metadata_json
    )

    assert metadata["tool"] == "tavily"
    assert metadata["query"] == "RAG research"
    assert metadata["success"] is True
    assert metadata["duration_ms"] == 845.5


@pytest.mark.asyncio
async def test_tool_failure_event(db, task_id):

    job = await create_research_job(
        db=db,
        task_id=task_id,
    )

    # tool_failed: what the workflow records for a search that failed.
    event = await record_tool_event(
        db=db,
        job_id=job.id,
        event_type="tool_failed",
        tool="arxiv",
        query="broken query",
        success=False,
        duration_ms=100.0,
        error="Connection timeout",
    )

    metadata = json.loads(
        event.metadata_json
    )

    assert event.event_type == "tool_failed"
    assert metadata["tool"] == "arxiv"
    assert metadata["success"] is False
    assert metadata["error"] == "Connection timeout"


@pytest.mark.asyncio
async def test_parse_event_metadata(db, task_id):

    job = await create_research_job(
        db=db,
        task_id=task_id,
    )

    event = await record_tool_event(
        db=db,
        job_id=job.id,
        event_type="tool_completed",
        tool="wikipedia",
        query="machine learning",
        success=True,
        duration_ms=500.0,
    )

    metadata = parse_event_metadata(event)

    assert metadata["tool"] == "wikipedia"
    assert metadata["query"] == "machine learning"


# -------------------------
# Response schemas
# -------------------------


@pytest.mark.asyncio
async def test_job_event_response_includes_parsed_metadata(db, task_id):

    from app.schemas.research import JobEventResponse

    job = await create_research_job(
        db=db,
        task_id=task_id,
    )

    event = await record_tool_event(
        db=db,
        job_id=job.id,
        event_type="tool_completed",
        tool="tavily",
        query="RAG research",
        success=True,
        duration_ms=845.5,
    )

    await db.refresh(event)

    response = JobEventResponse.model_validate(event)

    assert response.id == event.id
    assert response.job_id == job.id
    assert response.event_type == "tool_completed"
    assert response.message == "tavily tool event."
    assert response.metadata == {
        "tool": "tavily",
        "query": "RAG research",
        "success": True,
        "duration_ms": 845.5,
    }

    # Serialised as "metadata".
    assert response.model_dump()["metadata"]["tool"] == "tavily"


@pytest.mark.asyncio
async def test_research_execution_response(db, task_id):

    from app.db.models import ResearchTask
    from app.schemas.research import ResearchExecutionResponse

    job = await create_research_job(
        db=db,
        task_id=task_id,
    )

    task = await db.get(ResearchTask, task_id)

    response = ResearchExecutionResponse(
        task=task,
        jobs=[job],
    )

    assert response.task.id == task_id
    assert response.task.status == "pending"

    [job_response] = response.jobs

    assert job_response.id == job.id
    assert job_response.status == "pending"
    assert job_response.attempts == 0

