import pytest

from app.agents.executor import ResearchExecutor
from app.agents.planner import create_research_plan
from app.tests.fake_tools import FakeTool


class FakeRegistry:

    def __init__(self):

        self.tools = {
            "tavily": FakeTool("tavily"),
            "arxiv": FakeTool("arxiv"),
            "wikipedia": FakeTool("wikipedia"),
        }

    def get(
        self,
        name: str,
    ):

        return self.tools[name]


@pytest.mark.asyncio
async def test_executor():

    plan = create_research_plan(
        "How do modern RAG systems work?"
    )

    registry = FakeRegistry()

    executor = ResearchExecutor(
        registry=registry
    )

    results = await executor.execute(
        steps=plan.steps
    )

    assert len(results) == 3

    assert results[0].tool == "tavily"

    assert results[1].tool == "arxiv"

    assert results[2].tool == "wikipedia"

    for result in results:

        assert result.success is True