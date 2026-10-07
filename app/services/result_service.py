from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import ResearchResult, ResearchStep
from app.tools.base import ToolResult


async def save_tool_result(
        db: AsyncSession,
        step: ResearchStep,
        result: ToolResult
) -> ResearchResult:
    research_result = ResearchResult(
        step_id=step.id,
        tool=result.tool,
        query=result.query,
        content=result.content,
        success=result.success,
        error=result.error
    )
    db.add(research_result)
    await db.flush()
    return research_result