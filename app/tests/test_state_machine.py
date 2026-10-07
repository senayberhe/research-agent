import pytest
from sqlalchemy import select

from app.db.models import (
    AgentRun,
    AgentRunStatus,
    ResearchTask,
    TaskStatus,
)
from app.services import workflow_service
from app.services.agent_run_service import complete_agent_run, create_agent_run
from app.services.state_machine import (
    AGENT_RUN_TRANSITIONS,
    TASK_TRANSITIONS,
    InvalidStateTransition,
    can_transition_task,
    is_resumable,
    transition_agent_run,
    transition_task,
)
from app.tests.fake_agent_llm import FakeAgentLLM
from app.tests.fake_agent_tools import FakeRegistry
from app.tests.test_workflow_service import create_task


def make_task(status) -> ResearchTask:
    return ResearchTask(id=1, question="q", status=status)


def make_run(status) -> AgentRun:
    return AgentRun(id=1, task_id=1, status=status)


# -------------------------
# Tables
# -------------------------


def test_every_status_has_transitions():

    assert set(TASK_TRANSITIONS) == set(TaskStatus)
    assert set(AGENT_RUN_TRANSITIONS) == set(AgentRunStatus)


# -------------------------
# Task transitions
# -------------------------


def test_task_happy_path():

    task = make_task(TaskStatus.PENDING)

    for status in [
        TaskStatus.PLANNING,
        TaskStatus.RESEARCHING,
        TaskStatus.SYNTHESIZING,
        TaskStatus.COMPLETED,
    ]:
        transition_task(task, status)

    assert task.status == TaskStatus.COMPLETED


@pytest.mark.parametrize(
    "status",
    [
        TaskStatus.PENDING,
        TaskStatus.PLANNING,
        TaskStatus.RESEARCHING,
        TaskStatus.SYNTHESIZING,
    ],
)
def test_any_unfinished_task_can_fail(status):

    task = make_task(status)

    transition_task(task, TaskStatus.FAILED)

    assert task.status == TaskStatus.FAILED


def test_failed_task_resumes_into_researching():

    task = make_task(TaskStatus.FAILED)

    assert is_resumable(task)

    transition_task(task, TaskStatus.RESEARCHING)

    assert task.status == TaskStatus.RESEARCHING


@pytest.mark.parametrize(
    ("current", "new"),
    [
        # Skipping steps.
        (TaskStatus.PENDING, TaskStatus.RESEARCHING),
        (TaskStatus.PENDING, TaskStatus.COMPLETED),
        (TaskStatus.RESEARCHING, TaskStatus.COMPLETED),
        # Going backwards.
        (TaskStatus.SYNTHESIZING, TaskStatus.RESEARCHING),
        # Leaving COMPLETED.
        (TaskStatus.COMPLETED, TaskStatus.FAILED),
        (TaskStatus.COMPLETED, TaskStatus.RESEARCHING),
        # A failed task only comes back through a resume.
        (TaskStatus.FAILED, TaskStatus.COMPLETED),
        (TaskStatus.FAILED, TaskStatus.FAILED),
    ],
)
def test_invalid_task_transitions_raise(current, new):

    task = make_task(current)

    assert not can_transition_task(task, new)

    with pytest.raises(InvalidStateTransition):
        transition_task(task, new)

    # Status unchanged.
    assert task.status == current


def test_status_loaded_as_plain_string_works():

    # Rows read from the database hold "failed", not TaskStatus.FAILED.
    task = make_task("failed")

    assert is_resumable(task)
    assert can_transition_task(task, TaskStatus.RESEARCHING)


@pytest.mark.parametrize(
    "status",
    [s for s in TaskStatus if s != TaskStatus.FAILED],
)
def test_only_failed_tasks_are_resumable(status):

    assert not is_resumable(make_task(status))


def test_error_message_names_both_statuses():

    with pytest.raises(InvalidStateTransition) as exc_info:
        transition_task(make_task(TaskStatus.COMPLETED), TaskStatus.FAILED)

    assert str(exc_info.value) == (
        "Research task 1 can't go from completed to failed"
    )


# -------------------------
# Agent run transitions
# -------------------------


def test_run_lifecycle_with_two_iterations():

    run = make_run(AgentRunStatus.CREATED)

    for status in [
        AgentRunStatus.RUNNING,
        AgentRunStatus.WAITING_FOR_TOOL,
        AgentRunStatus.PROCESSING_RESULT,
        # Next iteration.
        AgentRunStatus.RUNNING,
        AgentRunStatus.COMPLETED,
    ]:
        transition_agent_run(run, status)

    assert run.status == AgentRunStatus.COMPLETED


@pytest.mark.parametrize(
    "status",
    [
        AgentRunStatus.CREATED,
        AgentRunStatus.RUNNING,
        AgentRunStatus.WAITING_FOR_TOOL,
        AgentRunStatus.PROCESSING_RESULT,
    ],
)
def test_any_unfinished_run_can_fail(status):

    run = make_run(status)

    transition_agent_run(run, AgentRunStatus.FAILED)

    assert run.status == AgentRunStatus.FAILED


def test_failed_run_resumes_into_running():

    run = make_run(AgentRunStatus.FAILED)

    transition_agent_run(run, AgentRunStatus.RUNNING)

    assert run.status == AgentRunStatus.RUNNING


@pytest.mark.parametrize(
    ("current", "new"),
    [
        # Skipping phases.
        (AgentRunStatus.CREATED, AgentRunStatus.WAITING_FOR_TOOL),
        (AgentRunStatus.CREATED, AgentRunStatus.COMPLETED),
        (AgentRunStatus.RUNNING, AgentRunStatus.PROCESSING_RESULT),
        (AgentRunStatus.WAITING_FOR_TOOL, AgentRunStatus.RUNNING),
        # Only RUNNING (the model's answer) can complete.
        (AgentRunStatus.WAITING_FOR_TOOL, AgentRunStatus.COMPLETED),
        (AgentRunStatus.PROCESSING_RESULT, AgentRunStatus.COMPLETED),
        # A failed run only comes back through a resume.
        (AgentRunStatus.FAILED, AgentRunStatus.COMPLETED),
        (AgentRunStatus.FAILED, AgentRunStatus.WAITING_FOR_TOOL),
    ],
)
def test_invalid_run_transitions_raise(current, new):

    with pytest.raises(InvalidStateTransition):
        transition_agent_run(make_run(current), new)


def test_completed_run_is_final():

    for new in AgentRunStatus:
        with pytest.raises(InvalidStateTransition):
            transition_agent_run(make_run(AgentRunStatus.COMPLETED), new)


@pytest.mark.asyncio
async def test_completing_a_completed_run_raises(db_session):

    task = await create_task(db_session)

    run = await create_agent_run(db=db_session, task_id=task.id)
    transition_agent_run(run, AgentRunStatus.RUNNING)

    usage = dict(
        iteration_count=1,
        tool_call_count=0,
        input_tokens=1,
        output_tokens=1,
        total_tokens=2,
        estimated_cost_usd=None,
    )

    await complete_agent_run(db=db_session, run=run, **usage)

    with pytest.raises(InvalidStateTransition):
        await complete_agent_run(db=db_session, run=run, **usage)


# -------------------------
# Workflow failure handling
# -------------------------


@pytest.mark.asyncio
async def test_failure_after_completion_keeps_task_and_run_completed(
    db_session,
    monkeypatch,
):

    monkeypatch.setattr(workflow_service, "OpenAIProvider", FakeAgentLLM)
    monkeypatch.setattr(workflow_service, "ToolRegistry", FakeRegistry)

    # The workflow's only db.refresh on the success path comes right after
    # the task is committed as COMPLETED; make that one fail.
    real_refresh = db_session.refresh
    refresh_calls = 0

    async def refresh_failing_once(obj, *args, **kwargs):
        nonlocal refresh_calls
        refresh_calls += 1
        if refresh_calls == 1:
            raise RuntimeError("refresh failed")
        return await real_refresh(obj, *args, **kwargs)

    # After create_task, which calls refresh itself.
    task = await create_task(db_session)
    task_id = task.id

    monkeypatch.setattr(db_session, "refresh", refresh_failing_once)

    returned = await workflow_service.execute_research_workflow(
        db=db_session,
        task=task,
    )

    # No InvalidStateTransition escaped, and nothing was "un-completed".
    assert returned is task
    assert task.status == TaskStatus.COMPLETED

    db_session.expire_all()

    run = (
        await db_session.execute(
            select(AgentRun).where(AgentRun.task_id == task_id)
        )
    ).scalar_one()

    assert run.status == AgentRunStatus.COMPLETED
    assert run.error is None
