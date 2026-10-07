import pytest

from app.db.models import (
    ResearchStep,
    ResearchTask,
    TaskStatus,
)
from app.services.result_service import save_tool_result
from app.tools.base import ToolResult


@pytest.mark.asyncio
async def test_save_tool_result(db_session):

    # -------------------------
    # Create research task
    # -------------------------

    task = ResearchTask(
        question="How does RAG work?",
        status=TaskStatus.RESEARCHING,
    )

    db_session.add(task)

    await db_session.commit()
    await db_session.refresh(task)

    # -------------------------
    # Create research step
    # -------------------------

    step = ResearchStep(
        task_id=task.id,
        tool="tavily",
        query="How does RAG work?",
        status="completed",
    )

    db_session.add(step)

    await db_session.commit()
    await db_session.refresh(step)

    # -------------------------
    # Create tool result
    # -------------------------

    tool_result = ToolResult(
        tool="tavily",
        query="How does RAG work?",
        content="RAG combines retrieval with generation.",
        success=True,
        error=None,
    )

    # -------------------------
    # Save result
    # -------------------------

    result = await save_tool_result(
        db=db_session,
        step=step,
        result=tool_result,
    )

    # -------------------------
    # Verify result
    # -------------------------

    assert result.id is not None

    assert result.step_id == step.id

    assert result.tool == "tavily"

    assert result.query == "How does RAG work?"

    assert (
        result.content
        == "RAG combines retrieval with generation."
    )

    assert result.success is True

    assert result.error is None

    assert result.created_at is not None