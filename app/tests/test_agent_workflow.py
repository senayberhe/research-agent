import pytest

from app.agents.research_agent import ResearchAgent
from app.tests.fake_agent_llm import FakeAgentLLM
from app.tests.fake_agent_tools import FakeRegistry


@pytest.mark.asyncio
async def test_agent_workflow_end_to_end():

    llm = FakeAgentLLM()

    registry = FakeRegistry()

    agent = ResearchAgent(
        llm=llm,
        registry=registry,
    )

    result = await agent.run(
        question=(
            "How does retrieval augmented "
            "generation improve factuality?"
        )
    )

    assert result.answer

    assert len(result.tool_results) == 1

    tool_result = result.tool_results[0]

    assert tool_result.success is True

    assert tool_result.tool == "tavily"

    assert tool_result.query == (
        "retrieval augmented generation"
    )
