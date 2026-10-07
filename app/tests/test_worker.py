"""The research worker: claim a pending job, run the ResearchAgent through
the workflow (which records the AgentRun), mark the job completed or failed.

The worker commits through its own sessions, so these tests use
clean_tables instead of the rolled-back db_session.
"""

import asyncio

import pytest
import pytest_asyncio
from sqlalchemy import select

from app.db.models import (
    AgentRun,
    AgentRunStatus,
    JobStatus,
    ResearchJob,
    ResearchTask,
    TaskStatus,
)
from app.jobs.service import requeue_job
from app.jobs.worker import (
    ResearchWorker,
    default_worker_id,
    run_once,
    run_worker,
)
from app.services import workflow_service
from app.services.research_service import create_research_task
from app.tests.conftest import TestSessionLocal
from app.tests.fake_agent_llm import FakeAgentLLM
from app.tests.fake_agent_tools import FakeRegistry
from app.tests.test_agent_checkpoint import CrashOnThirdCallLLM
from app.tests.test_resume_workflow import AnswerLLM
from app.tests.test_workflow_service import BrokenLLM


@pytest.fixture(autouse=True)
def fake_tools(monkeypatch):
    monkeypatch.setattr(workflow_service, "ToolRegistry", FakeRegistry)


@pytest.fixture(autouse=True)
def worker_health_file_in_tmp(monkeypatch, tmp_path):
    # Don't touch the real health file (/tmp/research-worker-health.json).
    from app.core.config import settings

    monkeypatch.setattr(
        settings,
        "worker_health_file",
        str(tmp_path / "worker-health.json"),
    )


@pytest.fixture(autouse=True)
def worker_uses_test_database(monkeypatch):
    # ResearchWorker() uses the app's database by default; here it must use
    # the test database.
    monkeypatch.setattr("app.jobs.worker.AsyncSessionLocal", TestSessionLocal)


@pytest_asyncio.fixture
async def db(clean_tables):
    # A real session: the worker uses its own sessions, which only see data
    # that was actually committed (not the rolled-back db_session's).
    async with TestSessionLocal() as session:
        yield session


def use_llm(monkeypatch, llm_class):
    monkeypatch.setattr(workflow_service, "OpenAIProvider", llm_class)


async def new_task(question="What is RAG?") -> int:
    async with TestSessionLocal() as db:
        task = await create_research_task(db=db, question=question)
        return task.id


async def load(task_id):
    """The task, its latest job and its runs, fresh from the database."""

    async with TestSessionLocal() as db:

        task = await db.get(ResearchTask, task_id)
        job = await db.scalar(
            select(ResearchJob)
            .where(ResearchJob.task_id == task_id)
            .order_by(ResearchJob.id.desc())
        )
        runs = (
            await db.execute(
                select(AgentRun)
                .where(AgentRun.task_id == task_id)
                .order_by(AgentRun.id)
            )
        ).scalars().all()

        return task, job, list(runs)


# -------------------------
# One job
# -------------------------


@pytest.mark.asyncio
async def test_no_pending_job(clean_tables):

    assert await run_once(
        session_factory=TestSessionLocal,
        worker_id="worker-1",
    ) is False


@pytest.mark.asyncio
async def test_worker_completes_a_research_job(clean_tables, monkeypatch):

    use_llm(monkeypatch, FakeAgentLLM)

    task_id = await new_task()

    assert await run_once(session_factory=TestSessionLocal, worker_id="worker-1")

    task, job, [run] = await load(task_id)

    # The job: claimed by this worker, run once, completed.
    assert job.status == JobStatus.COMPLETED
    assert job.worker_id == "worker-1"
    assert job.attempts == 1
    assert job.started_at is not None
    assert job.completed_at >= job.started_at
    assert job.error is None

    # The research itself: the agent ran and the AgentRun was recorded.
    assert task.status == TaskStatus.COMPLETED
    assert task.summary.startswith("RAG improves factuality")

    assert run.status == AgentRunStatus.COMPLETED
    assert run.iteration_count == 2
    assert run.tool_call_count == 1
    assert run.total_tokens == 300


async def job_timeline(task_id: int) -> list[tuple]:

    from app.jobs.event_service import get_job_events

    _, job, _ = await load(task_id)

    async with TestSessionLocal() as db:
        events = await get_job_events(db=db, job_id=job.id)

    return [
        (event.event_type, event.tool, event.duration_ms is not None)
        for event in events
    ]


@pytest.mark.asyncio
async def test_worker_records_tool_events(clean_tables, monkeypatch):

    use_llm(monkeypatch, FakeAgentLLM)

    task_id = await new_task()

    assert await run_once(session_factory=TestSessionLocal, worker_id="worker-1")

    # (event, tool, has a duration)
    assert await job_timeline(task_id) == [
        ("created", None, False),
        ("claimed", None, False),
        ("tool_started", "tavily", False),
        ("tool_completed", "tavily", True),
        ("completed", None, False),
    ]


@pytest.mark.asyncio
async def test_worker_records_failed_tool(clean_tables, monkeypatch):

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
    use_llm(monkeypatch, FakeAgentLLM)

    task_id = await new_task()

    assert await run_once(session_factory=TestSessionLocal, worker_id="worker-1")

    from app.jobs.event_service import get_job_events

    _, job, _ = await load(task_id)

    async with TestSessionLocal() as db:
        [failed] = [
            event
            for event in await get_job_events(db=db, job_id=job.id)
            if event.event_type == "tool_failed"
        ]

    assert failed.tool == "tavily"
    assert failed.details["query"] == "retrieval augmented generation"
    assert failed.details["success"] is False
    assert failed.details["error"] == "rate limited"
    assert failed.duration_ms is not None


@pytest.mark.asyncio
async def test_worker_fails_job_when_agent_fails(clean_tables, monkeypatch):

    use_llm(monkeypatch, BrokenLLM)

    task_id = await new_task()

    assert await run_once(session_factory=TestSessionLocal, worker_id="worker-1")

    task, job, [run] = await load(task_id)

    # The run's error is the job's error.
    assert job.status == JobStatus.FAILED
    assert job.error == "LLM service unavailable"
    assert job.completed_at is not None

    assert task.status == TaskStatus.FAILED
    assert run.status == AgentRunStatus.FAILED


@pytest.mark.asyncio
async def test_failed_job_retried_resumes_same_run(clean_tables, monkeypatch):

    use_llm(monkeypatch, CrashOnThirdCallLLM)

    task_id = await new_task()

    await run_once(session_factory=TestSessionLocal, worker_id="worker-1")

    _, job, [run] = await load(task_id)

    assert job.status == JobStatus.FAILED
    assert job.error == "OpenAI connection reset"
    run_id = run.id

    # Retried: back in the queue, picked up by another worker, resumed from
    # the checkpoint (no new run, no repeated searches).
    async with TestSessionLocal() as db:
        await requeue_job(db=db, job=await db.get(ResearchJob, job.id))

    use_llm(monkeypatch, AnswerLLM)

    await run_once(session_factory=TestSessionLocal, worker_id="worker-2")

    task, job, [run] = await load(task_id)

    assert job.status == JobStatus.COMPLETED
    assert job.attempts == 2
    assert job.worker_id == "worker-2"

    assert task.status == TaskStatus.COMPLETED
    assert run.id == run_id
    assert run.iteration_count == 3


@pytest.mark.asyncio
async def test_retry_without_checkpoint_fails_with_reason(clean_tables, monkeypatch):

    # Fails on the first LLM call: no checkpoint to resume from.
    use_llm(monkeypatch, BrokenLLM)

    task_id = await new_task()

    await run_once(session_factory=TestSessionLocal, worker_id="worker-1")

    _, job, _ = await load(task_id)

    async with TestSessionLocal() as db:
        await requeue_job(db=db, job=await db.get(ResearchJob, job.id))

    await run_once(session_factory=TestSessionLocal, worker_id="worker-1")

    _, job, runs = await load(task_id)

    assert job.status == JobStatus.FAILED
    assert job.error == "Agent run has no checkpoint to resume."
    assert len(runs) == 1


# -------------------------
# The worker loop
# -------------------------


@pytest.mark.asyncio
async def test_worker_runs_queue_in_order_and_stops(clean_tables, monkeypatch):

    use_llm(monkeypatch, FakeAgentLLM)

    task_ids = [await new_task(f"Question {n}") for n in range(3)]

    stop = asyncio.Event()

    worker = asyncio.create_task(
        run_worker(
            session_factory=TestSessionLocal,
            worker_id="worker-1",
            stop=stop,
            poll_interval_seconds=0.01,
        )
    )

    # Wait (up to 10s) until every job has finished: one query per check.
    async with TestSessionLocal() as db:
        for _ in range(200):
            db.expire_all()
            jobs = (
                await db.execute(
                    select(ResearchJob)
                    .where(ResearchJob.task_id.in_(task_ids))
                    .order_by(ResearchJob.task_id)
                )
            ).scalars().all()
            if all(job.status == JobStatus.COMPLETED for job in jobs):
                break
            await asyncio.sleep(0.05)

    stop.set()
    await asyncio.wait_for(worker, timeout=5)

    # All done, oldest first.
    assert all(job.status == JobStatus.COMPLETED for job in jobs)
    assert [job.started_at for job in jobs] == sorted(
        job.started_at for job in jobs
    )


@pytest.mark.asyncio
async def test_worker_stops_promptly_when_idle(clean_tables):

    stop = asyncio.Event()

    worker = asyncio.create_task(
        run_worker(
            session_factory=TestSessionLocal,
            worker_id="worker-1",
            stop=stop,
            # Long poll: stopping must not wait for it.
            poll_interval_seconds=60,
        )
    )

    await asyncio.sleep(0.05)
    stop.set()

    await asyncio.wait_for(worker, timeout=1)


def test_default_worker_id_is_host_and_pid():

    import os
    import socket

    assert default_worker_id() == f"{socket.gethostname()}-{os.getpid()}"


# -------------------------
# ResearchWorker
# -------------------------


@pytest.mark.asyncio
async def test_worker_processes_pending_job(db, monkeypatch):

    task = ResearchTask(
        question="What is RAG?",
        status=TaskStatus.PENDING,
    )

    db.add(task)

    await db.flush()

    job = ResearchJob(
        task_id=task.id,
        status=JobStatus.PENDING,
    )

    db.add(job)

    await db.commit()

    # Like the real workflow when the research succeeds: the task ends up
    # COMPLETED (the worker completes the job only if it did).
    async def fake_workflow(
        db,
        task,
        job_id,
    ):
        task.status = TaskStatus.COMPLETED
        await db.commit()
        return task

    monkeypatch.setattr(
        "app.jobs.worker.execute_research_workflow",
        fake_workflow,
    )

    worker = ResearchWorker(
        worker_id="test-worker",
    )

    processed = await worker.process_next_job()

    assert processed is True

    await db.refresh(job)

    assert job.status == JobStatus.COMPLETED
    assert job.worker_id == "test-worker"
    assert job.attempts == 1


@pytest.mark.asyncio
async def test_worker_marks_job_failed_on_workflow_error(
    db,
    monkeypatch,
):

    task = ResearchTask(
        question="What is RAG?",
        status=TaskStatus.PENDING,
    )

    db.add(task)

    await db.flush()

    job = ResearchJob(
        task_id=task.id,
        status=JobStatus.PENDING,
    )

    db.add(job)

    await db.commit()

    async def failing_workflow(
        db,
        task,
        job_id,
    ):
        raise RuntimeError(
            "Agent execution failed."
        )

    monkeypatch.setattr(
        "app.jobs.worker.execute_research_workflow",
        failing_workflow,
    )

    worker = ResearchWorker(
        worker_id="test-worker",
    )

    processed = await worker.process_next_job()

    assert processed is True

    await db.refresh(job)

    assert job.status == JobStatus.FAILED
    assert job.error == "Agent execution failed."
    assert job.worker_id == "test-worker"
    assert job.attempts == 1


@pytest.mark.asyncio
async def test_worker_fails_job_when_workflow_fails_the_task(db, monkeypatch):

    task = ResearchTask(question="What is RAG?", status=TaskStatus.PENDING)
    db.add(task)
    await db.flush()

    job = ResearchJob(task_id=task.id, status=JobStatus.PENDING)
    db.add(job)
    await db.commit()

    # The real workflow doesn't raise when research fails: it marks the task
    # FAILED and returns it. That must still fail the job.
    async def workflow_that_fails_task(db, task, job_id):
        task.status = TaskStatus.FAILED
        await db.commit()
        return task

    monkeypatch.setattr(
        "app.jobs.worker.execute_research_workflow",
        workflow_that_fails_task,
    )

    assert await ResearchWorker(worker_id="test-worker").process_next_job()

    await db.refresh(job)

    assert job.status == JobStatus.FAILED
    assert job.error == f"Research task {task.id} ended as failed."


@pytest.mark.asyncio
async def test_worker_with_no_pending_job(db):

    assert await ResearchWorker(worker_id="test-worker").process_next_job() is False


def test_worker_defaults():

    worker = ResearchWorker()

    assert worker.worker_id == default_worker_id()
    assert worker.session_factory is TestSessionLocal


@pytest.mark.asyncio
async def test_stop_ends_run(db):

    worker = ResearchWorker(worker_id="test-worker", poll_interval_seconds=60)

    running = asyncio.create_task(worker.run(recover=False))

    await asyncio.sleep(0.05)

    worker.stop()

    # Promptly, despite the 60s poll interval.
    await asyncio.wait_for(running, timeout=2)

    assert worker.status == "stopped"


@pytest.mark.asyncio
async def test_main_stops_worker_on_sigterm(db, monkeypatch, caplog):

    import logging
    import os
    import signal

    from app.jobs import main as worker_main

    monkeypatch.setenv("WORKER_ID", "signal-test-worker")
    monkeypatch.setattr(worker_main, "configure_logging", lambda: None)

    caplog.set_level(logging.INFO, logger="app.jobs.main")

    running = asyncio.create_task(worker_main.main())

    # Let it start and install its signal handlers, then "docker stop" it.
    await asyncio.sleep(0.1)
    os.kill(os.getpid(), signal.SIGTERM)

    try:
        await asyncio.wait_for(running, timeout=5)
    finally:
        # Don't leave the handlers installed for the rest of the test run.
        loop = asyncio.get_running_loop()
        loop.remove_signal_handler(signal.SIGTERM)
        loop.remove_signal_handler(signal.SIGINT)

    assert "Shutdown signal received by signal-test-worker" in caplog.text
