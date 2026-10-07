"""Recovery when a worker starts: work a crash or restart cut off is failed
and requeued, so a worker continues it from its checkpoint.

These tests commit through their own sessions (like the worker does), so
they use clean_tables instead of the rolled-back db_session.
"""

import asyncio

import pytest
from sqlalchemy import select

from app.agents.research_agent import AgentCheckpoint
from app.db.models import AgentRun, AgentRunStatus, ResearchTask, TaskStatus
from app.jobs.models import JobStatus, ResearchJob
from app.jobs.worker import run_once
from app.services import workflow_service
from app.services.recovery_service import (
    INTERRUPTED_ERROR,
    recover_interrupted_jobs,
    recover_interrupted_runs,
)
from app.services.research_service import create_research_task
from app.tests.conftest import TestSessionLocal
from app.tests.fake_agent_llm import FakeFunctionCall, FakeResponse
from app.tests.fake_agent_tools import FakeRegistry
from app.tests.test_resume_workflow import AnswerLLM


def checkpoint_after(iteration: int) -> dict:
    return AgentCheckpoint(
        iteration=iteration,
        input_items=[{"role": "user", "content": "What is RAG?"}],
        tool_call_count=iteration,
        input_tokens=100 * iteration,
        output_tokens=50 * iteration,
        total_tokens=150 * iteration,
        estimated_cost=None,
        executed_tool_calls=iteration,
    ).to_dict()


async def add_run(
    db,
    status,
    task_status=TaskStatus.RESEARCHING,
    state=None,
    job_status=None,
    job_attempts=1,
):
    """A task, run and (optionally) job as a dead worker left them."""

    task = ResearchTask(question="What is RAG?", status=task_status)
    db.add(task)
    await db.flush()

    # Set directly (not through the state machine): this is the state the
    # database is found in, not a transition.
    if status is not None:
        db.add(AgentRun(task_id=task.id, status=status, state=state))

    if job_status is not None:
        db.add(
            ResearchJob(
                task_id=task.id,
                status=job_status,
                attempts=job_attempts,
                worker_id="dead-worker",
            )
        )

    await db.flush()

    return task.id


async def latest_job(db, task_id) -> ResearchJob | None:
    return await db.scalar(
        select(ResearchJob)
        .where(ResearchJob.task_id == task_id)
        .order_by(ResearchJob.id.desc())
    )


# -------------------------
# recover_interrupted_runs
# -------------------------


@pytest.mark.asyncio
async def test_unfinished_runs_are_marked_failed(clean_tables):

    async with TestSessionLocal() as db:

        task_ids = {
            status: await add_run(db, status, state=checkpoint_after(2))
            for status in [
                AgentRunStatus.CREATED,
                AgentRunStatus.RUNNING,
                AgentRunStatus.WAITING_FOR_TOOL,
                AgentRunStatus.PROCESSING_RESULT,
            ]
        }
        await db.commit()

        resumable = await recover_interrupted_runs(db)

        db.expire_all()

        for status, task_id in task_ids.items():

            run = await db.scalar(
                select(AgentRun).where(AgentRun.task_id == task_id)
            )
            task = await db.get(ResearchTask, task_id)

            assert run.status == AgentRunStatus.FAILED, status
            assert run.error == INTERRUPTED_ERROR
            assert task.status == TaskStatus.FAILED

            # Totals from the checkpoint, not 0; the checkpoint stays.
            assert run.iteration_count == 2
            assert run.total_tokens == 300
            assert run.state is not None

        assert sorted(resumable) == sorted(task_ids.values())


@pytest.mark.asyncio
async def test_finished_runs_are_left_alone(clean_tables):

    async with TestSessionLocal() as db:

        await add_run(db, AgentRunStatus.COMPLETED, TaskStatus.COMPLETED)
        await add_run(
            db,
            AgentRunStatus.FAILED,
            TaskStatus.FAILED,
            state=checkpoint_after(1),
        )
        await db.commit()

        # An already failed run is resumable by hand, but wasn't
        # interrupted.
        assert await recover_interrupted_runs(db) == []


# -------------------------
# recover_interrupted_jobs
# -------------------------


@pytest.mark.asyncio
async def test_interrupted_job_is_failed_then_requeued(clean_tables):

    async with TestSessionLocal() as db:

        task_id = await add_run(
            db,
            AgentRunStatus.WAITING_FOR_TOOL,
            state=checkpoint_after(2),
            job_status=JobStatus.RUNNING,
        )
        await db.commit()

        queued = await recover_interrupted_jobs(
            db=db,
            requeue=True,
            max_attempts=3,
        )

        assert queued == [task_id]

        db.expire_all()

        job = await latest_job(db, task_id)

        # Back in the queue for any worker, attempt count kept.
        assert job.status == JobStatus.PENDING
        assert job.worker_id is None
        assert job.attempts == 1

        task = await db.get(ResearchTask, task_id)

        assert task.status == TaskStatus.FAILED


@pytest.mark.asyncio
async def test_job_that_died_before_starting_is_requeued(clean_tables):

    async with TestSessionLocal() as db:

        # Claimed, but the worker died before the task got going: no run.
        task_id = await add_run(
            db,
            None,
            task_status=TaskStatus.PENDING,
            job_status=JobStatus.RUNNING,
        )
        await db.commit()

        assert await recover_interrupted_jobs(
            db=db,
            requeue=True,
            max_attempts=3,
        ) == [task_id]

        db.expire_all()

        assert (await latest_job(db, task_id)).status == JobStatus.PENDING


@pytest.mark.asyncio
async def test_job_out_of_attempts_is_left_failed(clean_tables):

    async with TestSessionLocal() as db:

        task_id = await add_run(
            db,
            AgentRunStatus.RUNNING,
            state=checkpoint_after(1),
            job_status=JobStatus.RUNNING,
            job_attempts=3,
        )
        await db.commit()

        assert await recover_interrupted_jobs(
            db=db,
            requeue=True,
            max_attempts=3,
        ) == []

        db.expire_all()

        job = await latest_job(db, task_id)

        assert job.status == JobStatus.FAILED
        assert job.error == INTERRUPTED_ERROR


@pytest.mark.asyncio
async def test_run_without_checkpoint_is_not_requeued(clean_tables):

    async with TestSessionLocal() as db:

        # Nothing to resume from, and the task is no longer PENDING.
        task_id = await add_run(
            db,
            AgentRunStatus.RUNNING,
            job_status=JobStatus.RUNNING,
        )
        await db.commit()

        assert await recover_interrupted_jobs(
            db=db,
            requeue=True,
            max_attempts=3,
        ) == []

        db.expire_all()

        assert (await latest_job(db, task_id)).status == JobStatus.FAILED


@pytest.mark.asyncio
async def test_task_from_before_jobs_gets_a_new_job(clean_tables):

    async with TestSessionLocal() as db:

        # Interrupted, resumable, and has no job at all.
        task_id = await add_run(
            db,
            AgentRunStatus.RUNNING,
            state=checkpoint_after(1),
        )
        await db.commit()

        assert await recover_interrupted_jobs(
            db=db,
            requeue=True,
            max_attempts=3,
        ) == [task_id]

        db.expire_all()

        job = await latest_job(db, task_id)

        assert job.status == JobStatus.PENDING
        assert job.attempts == 0


@pytest.mark.asyncio
async def test_requeue_off_only_marks_failed(clean_tables):

    async with TestSessionLocal() as db:

        task_id = await add_run(
            db,
            AgentRunStatus.RUNNING,
            state=checkpoint_after(1),
            job_status=JobStatus.RUNNING,
        )
        await db.commit()

        assert await recover_interrupted_jobs(
            db=db,
            requeue=False,
            max_attempts=3,
        ) == []

        db.expire_all()

        # Left for POST /research/{task_id}/resume.
        assert (await latest_job(db, task_id)).status == JobStatus.FAILED


# -------------------------
# A real worker crash, end to end
# -------------------------


class KilledOnThirdCallLLM:
    """Searches twice; then the worker process is killed during call 3.

    CancelledError is what a running coroutine sees when the process shuts
    down. It isn't an Exception, so neither the workflow nor the worker can
    clean up: the job and run are left exactly as a crash would leave them.
    """

    def __init__(self):
        self.calls = 0

    async def generate_with_tools(self, input_items, tools, instructions=None):

        self.calls += 1

        if self.calls == 3:
            raise asyncio.CancelledError()

        return FakeResponse(
            output=[
                FakeFunctionCall(
                    type="function_call",
                    name="tavily_search",
                    arguments=f'{{"query": "rag {self.calls}"}}',
                    call_id=f"call_{self.calls}",
                )
            ],
        )


@pytest.mark.asyncio
async def test_worker_crash_is_resumed_by_next_worker(clean_tables, monkeypatch):

    monkeypatch.setattr(workflow_service, "ToolRegistry", FakeRegistry)
    monkeypatch.setattr(workflow_service, "OpenAIProvider", KilledOnThirdCallLLM)

    async with TestSessionLocal() as db:
        task = await create_research_task(db=db, question="What is RAG?")
        task_id = task.id

    # 1. Worker A claims the job and is killed partway through the run.
    with pytest.raises(asyncio.CancelledError):
        await run_once(session_factory=TestSessionLocal, worker_id="worker-a")

    async with TestSessionLocal() as db:

        job = await latest_job(db, task_id)
        run = await db.scalar(select(AgentRun).where(AgentRun.task_id == task_id))

        assert job.status == JobStatus.RUNNING
        assert job.worker_id == "worker-a"
        assert run.status == AgentRunStatus.RUNNING
        assert run.state["iteration"] == 2
        run_id = run.id

    # 2. Worker B starts: recovery requeues the job, then B claims it.
    AnswerLLM.instances.clear()
    monkeypatch.setattr(workflow_service, "OpenAIProvider", AnswerLLM)

    async with TestSessionLocal() as db:
        assert await recover_interrupted_jobs(
            db=db,
            requeue=True,
            max_attempts=3,
        ) == [task_id]

    assert await run_once(session_factory=TestSessionLocal, worker_id="worker-b")

    # 3. Same job and run, resumed from the checkpoint, completed.
    async with TestSessionLocal() as db:

        task = await db.get(ResearchTask, task_id)
        job = await latest_job(db, task_id)
        [run] = (
            await db.execute(select(AgentRun).where(AgentRun.task_id == task_id))
        ).scalars().all()

        assert task.status == TaskStatus.COMPLETED
        assert task.summary == "Resumed answer."

        assert job.status == JobStatus.COMPLETED
        assert job.worker_id == "worker-b"
        assert job.attempts == 2

        assert run.id == run_id
        assert run.status == AgentRunStatus.COMPLETED
        assert run.iteration_count == 3
        assert run.total_tokens == 450
        assert run.state is None

    # The resumed model was sent the conversation from the checkpoint.
    [llm] = AnswerLLM.instances
    assert len(llm.requests[0]) == 5
