import pytest

from app.agents.research_agent import ResearchAgent
from app.tests.fake_agent_llm import FakeAgentLLM
from app.tests.fake_agent_tools import FakeRegistry


@pytest.mark.asyncio
async def test_agent_tracks_iterations_and_tool_calls():

    llm = FakeAgentLLM()

    registry = FakeRegistry()

    agent = ResearchAgent(
        llm=llm,
        registry=registry,
    )

    result = await agent.run(
        question="What is retrieval augmented generation?"
    )

    assert result.iteration_count == 2

    assert result.tool_call_count == 1

    assert len(result.tool_results) == 1

    assert result.answer

@pytest.mark.asyncio
async def test_agent_counts_when_round_limit_is_hit():

    # Imported here to reuse the never-answering fake from the agent tests.
    from app.tests.test_research_agent import AlwaysCallsToolLLM

    result = await ResearchAgent(
        llm=AlwaysCallsToolLLM("tavily_search"),
        registry=FakeRegistry(),
    ).run(question="What is RAG?")

    assert result.success is False
    assert result.iteration_count == 6
    assert result.tool_call_count == 6


@pytest.mark.asyncio
async def test_agent_counts_invalid_tool_calls():

    from app.tests.test_research_agent import AlwaysCallsToolLLM

    result = await ResearchAgent(
        llm=AlwaysCallsToolLLM("google_search"),
        registry=FakeRegistry(),
    ).run(question="What is RAG?")

    # Every requested call is counted, even though none could run.
    assert result.tool_call_count == 6
    assert result.tool_results == []


@pytest.mark.asyncio
async def test_agent_tracks_token_usage():
    llm = FakeAgentLLM()
    registry = FakeRegistry()

    agent = ResearchAgent(
        llm=llm,
        registry=registry,
    )

    result = await agent.run(
        question=(
            "What is retrieval augmented generation?"
        )
    )

    assert result.usage.input_tokens == 200
    assert result.usage.output_tokens == 100
    assert result.usage.total_tokens == 300
