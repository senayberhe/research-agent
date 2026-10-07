import pytest
from sqlalchemy import select

from app.db.models import (
    ResearchResult,
    ResearchStep,
    ResearchTask,
    StepStatus,
    TaskStatus,
)
from app.services import workflow_service
from app.services.workflow_service import execute_research_workflow
from app.tests.fake_agent_llm import FakeAgentLLM, FakeFunctionCall, FakeResponse
from app.tests.fake_agent_tools import FakeRegistry


async def create_task(db_session) -> ResearchTask:

    task = ResearchTask(
        question="How does retrieval augmented generation improve factuality?",
        status=TaskStatus.PENDING,
    )

    db_session.add(task)

    await db_session.commit()
    await db_session.refresh(task)

    return task


async def load_steps(db_session, task_id: int) -> list[ResearchStep]:

    result = await db_session.execute(
        select(ResearchStep)
        .where(ResearchStep.task_id == task_id)
        .order_by(ResearchStep.id)
    )

    return list(result.scalars().all())


@pytest.mark.asyncio
async def test_execute_research_workflow(db_session, monkeypatch):

    # Fake LLM and tools instead of the real OpenAI and search APIs.
    monkeypatch.setattr(workflow_service, "OpenAIProvider", FakeAgentLLM)
    monkeypatch.setattr(workflow_service, "ToolRegistry", FakeRegistry)

    task = await create_task(db_session)

    returned = await execute_research_workflow(
        db=db_session,
        task=task,
    )

    # -------------------------
    # Verify task
    # -------------------------

    assert returned is task
    assert task.status == TaskStatus.COMPLETED
    assert task.summary.startswith("RAG improves factuality")

    # -------------------------
    # Verify saved step (one per agent tool call)
    # -------------------------

    steps = await load_steps(db_session, task.id)

    assert len(steps) == 1

    step = steps[0]

    assert step.tool == "tavily"
    assert step.query == "retrieval augmented generation"
    assert step.iteration == 1
    assert step.status == StepStatus.COMPLETED

    # Timing from _execute_tool is saved on the step.
    assert step.duration_ms is not None
    assert step.duration_ms >= 0

    # -------------------------
    # Verify saved result
    # -------------------------

    results = (
        await db_session.execute(
            select(ResearchResult).where(ResearchResult.step_id == step.id)
        )
    ).scalars().all()

    assert len(results) == 1
    assert results[0].content == "Fake Tavily research result."
    assert results[0].success is True


class NoAnswerLLM:
    """Keeps calling a tool and never gives a final answer."""

    def __init__(self):
        self.calls = 0

    async def generate_with_tools(self, input_items, tools, instructions=None):
        self.calls += 1
        return FakeResponse(
            output=[
                FakeFunctionCall(
                    type="function_call",
                    name="tavily_search",
                    arguments='{"query": "rag"}',
                    call_id=f"call_{self.calls}",
                )
            ]
        )


@pytest.mark.asyncio
async def test_workflow_fails_without_answer(db_session, monkeypatch):

    monkeypatch.setattr(workflow_service, "OpenAIProvider", NoAnswerLLM)
    monkeypatch.setattr(workflow_service, "ToolRegistry", FakeRegistry)

    task = await create_task(db_session)

    await execute_research_workflow(db=db_session, task=task)

    assert task.status == TaskStatus.FAILED
    assert task.summary is None

    # Each round's tool call is still saved, numbered by round.
    steps = await load_steps(db_session, task.id)

    assert [step.iteration for step in steps] == list(range(1, 7))
    assert all(step.status == StepStatus.COMPLETED for step in steps)


class BrokenLLM:
    """Simulates the LLM API being down."""

    async def generate_with_tools(self, input_items, tools, instructions=None):
        raise RuntimeError("LLM service unavailable")


@pytest.mark.asyncio
async def test_workflow_marks_task_failed_on_error(db_session, monkeypatch):

    monkeypatch.setattr(workflow_service, "OpenAIProvider", BrokenLLM)
    monkeypatch.setattr(workflow_service, "ToolRegistry", FakeRegistry)

    task = await create_task(db_session)

    returned = await execute_research_workflow(db=db_session, task=task)

    assert returned is task

    await db_session.refresh(task)

    assert task.status == TaskStatus.FAILED
