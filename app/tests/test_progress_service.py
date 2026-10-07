"""Following a research task live: GET /research/{task_id}/progress."""

import pytest

from app.db.models import (
    AgentRun,
    AgentRunStatus,
    JobStatus,
    ResearchJob,
    TaskStatus,
)
from app.services.progress_service import build_task_progress
from app.tests.conftest import TestSessionLocal
from app.tests.fake_agent_llm import FakeAgentLLM
from app.tests.test_research_api import client, resumable_state
from app.tests.test_timeline_service import (  # noqa: F401 (autouse fixture)
    fake_tools_and_health_file,
    run_task_through_worker,
)
from app.tests.test_workflow_service import BrokenLLM, create_task


def checkpoint(iteration: int, tool_calls: int, tokens_in: int, tokens_out: int):
    state = resumable_state()
    state.update(
        iteration=iteration,
        tool_call_count=tool_calls,
        executed_tool_calls=tool_calls,
        input_tokens=tokens_in,
        output_tokens=tokens_out,
        total_tokens=tokens_in + tokens_out,
        estimated_cost=None,
    )
    return state


@pytest.mark.asyncio
async def test_live_totals_change_while_the_run_is_going(db_session):

    task = await create_task(db_session)
    task.status = TaskStatus.RESEARCHING

    db_session.add(
        ResearchJob(
            task_id=task.id,
            status=JobStatus.RUNNING,
            attempts=1,
            worker_id="worker-1",
        )
    )

    # The run's columns are still 0 (written when it ends); its checkpoint
    # has the totals so far.
    run = AgentRun(
        task_id=task.id,
        status=AgentRunStatus.WAITING_FOR_TOOL,
        state=checkpoint(1, 1, 120, 40),
    )
    db_session.add(run)
    await db_session.commit()

    first = await build_task_progress(db_session, task.id)

    assert first["finished"] is False
    assert first["task"].summary is None
    assert first["job"].status == JobStatus.RUNNING

    assert first["run"]["live"] is True
    assert first["run"]["status"] == "waiting_for_tool"
    assert first["run"]["iterations"] == 1
    assert first["run"]["tool_calls"] == 1
    assert first["run"]["total_tokens"] == 160

    # The agent finishes another iteration...
    run.status = AgentRunStatus.RUNNING
    run.state = checkpoint(2, 3, 300, 90)
    await db_session.commit()

    second = await build_task_progress(db_session, task.id)

    # ...and the next poll shows it.
    assert second["run"]["iterations"] == 2
    assert second["run"]["tool_calls"] == 3
    assert second["run"]["total_tokens"] == 390
    assert second["run"]["elapsed_seconds"] >= first["run"]["elapsed_seconds"]


@pytest.mark.asyncio
async def test_completed_task_shows_its_summary(clean_tables, monkeypatch):

    task_id = await run_task_through_worker(monkeypatch, FakeAgentLLM)

    async with TestSessionLocal() as db:
        progress = await build_task_progress(db, task_id)

    assert progress["finished"] is True
    assert progress["can_resume"] is False
    assert progress["task"].status == TaskStatus.COMPLETED
    assert progress["task"].summary.startswith("RAG improves factuality")

    # The final totals, from the run itself.
    run = progress["run"]

    assert run["live"] is False
    assert run["status"] == "completed"
    assert run["iterations"] == 2
    assert run["tool_calls"] == 1
    assert run["total_tokens"] == 300
    assert run["completed_at"] is not None

    event_types = [event["event_type"] for event in progress["events"]]

    assert "tool_started" in event_types
    assert "tool_completed" in event_types
    assert event_types[-1] == "completed"


@pytest.mark.asyncio
async def test_failed_task_without_a_checkpoint_cant_resume(
    clean_tables,
    monkeypatch,
):

    # Fails on its first LLM call: no checkpoint to resume from.
    task_id = await run_task_through_worker(monkeypatch, BrokenLLM)

    async with TestSessionLocal() as db:
        progress = await build_task_progress(db, task_id)

    assert progress["finished"] is True
    assert progress["task"].status == TaskStatus.FAILED
    assert progress["run"]["status"] == "failed"
    assert progress["run"]["error"] == "LLM service unavailable"
    assert progress["can_resume"] is False


@pytest.mark.asyncio
async def test_failed_task_with_a_checkpoint_can_resume(db_session):

    task = await create_task(db_session)
    task.status = TaskStatus.FAILED

    db_session.add(
        AgentRun(
            task_id=task.id,
            status=AgentRunStatus.FAILED,
            state=resumable_state(),
            error="Tavily timed out",
        )
    )
    await db_session.commit()

    progress = await build_task_progress(db_session, task.id)

    assert progress["finished"] is True
    assert progress["can_resume"] is True


@pytest.mark.asyncio
async def test_progress_endpoint(clean_tables, monkeypatch):

    task_id = await run_task_through_worker(monkeypatch, FakeAgentLLM)

    response = client.get(f"/research/{task_id}/progress")

    assert response.status_code == 200

    body = response.json()

    assert body["finished"] is True
    assert body["task"]["summary"].startswith("RAG improves factuality")
    assert body["run"]["total_tokens"] == 300
    assert body["job"]["status"] == "completed"
    assert body["run"]["started_at"].endswith("Z")
    assert body["events"]


@pytest.mark.asyncio
async def test_progress_of_a_queued_task(clean_tables):

    from app.services.research_service import create_research_task

    async with TestSessionLocal() as db:
        task = await create_research_task(db=db, question="What is RAG?")

    body = client.get(f"/research/{task.id}/progress").json()

    assert body["finished"] is False
    assert body["job"]["status"] == "pending"
    # No worker has started it: no run yet.
    assert body["run"] is None
    assert [event["event_type"] for event in body["events"]] == ["created"]


def test_progress_of_missing_task():

    response = client.get("/research/999999/progress")

    assert response.status_code == 404


# -------------------------
# Tool calls
# -------------------------


@pytest.mark.asyncio
async def test_tool_calls_of_a_completed_task(clean_tables, monkeypatch):

    task_id = await run_task_through_worker(monkeypatch, FakeAgentLLM)

    async with TestSessionLocal() as db:
        progress = await build_task_progress(db, task_id)

    [call] = progress["tool_calls"]

    assert call["tool"] == "tavily"
    assert call["query"] == "retrieval augmented generation"
    assert call["iteration"] == 1
    assert call["status"] == "completed"
    assert call["success"] is True
    assert call["error"] is None
    assert call["duration_ms"] is not None
    assert call["result_preview"] == "Fake Tavily research result."
    assert call["result_truncated"] is False
    assert call["result_length"] == len("Fake Tavily research result.")


@pytest.mark.asyncio
async def test_failed_tool_call_shows_its_error(clean_tables, monkeypatch):

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

    async with TestSessionLocal() as db:
        [call] = (await build_task_progress(db, task_id))["tool_calls"]

    assert call["status"] == "failed"
    assert call["success"] is False
    assert call["error"] == "rate limited"
    assert call["result_preview"] is None


@pytest.mark.asyncio
async def test_running_and_long_tool_calls(db_session):

    from app.db.models import ResearchResult, ResearchStep, StepStatus
    from app.services.progress_service import RESULT_PREVIEW_CHARS

    task = await create_task(db_session)

    running = ResearchStep(
        task_id=task.id,
        tool="arxiv",
        query="RAG survey",
        iteration=2,
        status=StepStatus.RUNNING,
    )
    finished = ResearchStep(
        task_id=task.id,
        tool="wikipedia",
        query="GraphRAG",
        iteration=1,
        status=StepStatus.COMPLETED,
        duration_ms=530.0,
    )
    db_session.add_all([finished, running])
    await db_session.flush()

    long_content = "x" * (RESULT_PREVIEW_CHARS + 500)
    db_session.add(
        ResearchResult(
            step_id=finished.id,
            tool="wikipedia",
            query="GraphRAG",
            content=long_content,
        )
    )
    await db_session.commit()

    calls = (await build_task_progress(db_session, task.id))["tool_calls"]

    first, second = calls

    # Oldest first.
    assert first["tool"] == "wikipedia"
    assert len(first["result_preview"]) == RESULT_PREVIEW_CHARS
    assert first["result_truncated"] is True
    assert first["result_length"] == len(long_content)

    # Still running: no result, success unknown.
    assert second["status"] == "running"
    assert second["success"] is None
    assert second["duration_ms"] is None
    assert second["result_preview"] is None


@pytest.mark.asyncio
async def test_progress_endpoint_includes_tool_calls(clean_tables, monkeypatch):

    task_id = await run_task_through_worker(monkeypatch, FakeAgentLLM)

    [call] = client.get(f"/research/{task_id}/progress").json()["tool_calls"]

    assert call["tool"] == "tavily"
    assert call["success"] is True
    assert call["started_at"].endswith("Z")
