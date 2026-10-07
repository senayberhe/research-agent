import asyncio

import pytest

from app.agents.research_agent import ResearchAgent
from app.tools.base import ToolResult
from app.tests.fake_agent_llm import FakeAgentLLM, FakeFunctionCall, FakeResponse
from app.tests.fake_agent_tools import FakeRegistry


@pytest.mark.asyncio
async def test_research_agent_calls_tool_and_finishes():

    llm = FakeAgentLLM()
    registry = FakeRegistry()

    agent = ResearchAgent(
        llm=llm,
        registry=registry,
    )

    result = await agent.run(
        question=(
            "How does retrieval augmented generation "
            "improve factuality?"
        )
    )

    assert result.answer != ""

    assert len(result.tool_results) == 1

    assert result.tool_results[0].tool == "tavily"

    assert (
        result.tool_results[0].query
        == "retrieval augmented generation"
    )

    assert llm.calls == 2

@pytest.mark.asyncio
async def test_research_agent_sends_tool_output_back_to_model():

    llm = FakeAgentLLM()

    await ResearchAgent(
        llm=llm,
        registry=FakeRegistry(),
    ).run(question="What is RAG?")

    # Second request: the question, the model's function call, then the
    # tool's output for that same call_id.
    second_request = llm.requests[1]

    assert second_request[0] == {"role": "user", "content": "What is RAG?"}
    assert second_request[1].call_id == "call_1"
    assert second_request[2] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "Fake Tavily research result.",
    }


class AlwaysCallsToolLLM:
    """Never gives a final answer; keeps requesting the given tool."""

    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        self.calls = 0
        self.requests = []

    async def generate_with_tools(self, input_items, tools, instructions=None):
        self.calls += 1
        self.requests.append(list(input_items))
        return FakeResponse(
            output=[
                FakeFunctionCall(
                    type="function_call",
                    name=self.tool_name,
                    arguments='{"query": "rag"}',
                    call_id=f"call_{self.calls}",
                )
            ]
        )


@pytest.mark.asyncio
async def test_research_agent_stops_after_max_rounds():

    llm = AlwaysCallsToolLLM("tavily_search")

    result = await ResearchAgent(
        llm=llm,
        registry=FakeRegistry(),
    ).run(question="What is RAG?")

    assert result.success is False
    assert llm.calls == 6
    assert len(result.tool_results) == 6


@pytest.mark.asyncio
async def test_research_agent_reports_unknown_tool_to_model():

    llm = AlwaysCallsToolLLM("google_search")

    result = await ResearchAgent(
        llm=llm,
        registry=FakeRegistry(),
    ).run(question="What is RAG?")

    # No tool ran, and the model was told the call was invalid.
    assert result.tool_results == []
    assert llm.requests[1][-1]["output"].startswith(
        "Error: invalid call to google_search"
    )


class TwoCallsLLM:
    """Asks for two searches in one round, then answers."""

    def __init__(self):
        self.calls = 0

    async def generate_with_tools(self, input_items, tools, instructions=None):
        self.calls += 1
        if self.calls == 1:
            return FakeResponse(
                output=[
                    FakeFunctionCall(type="function_call", name="tavily_search", arguments='{"query": "a"}', call_id="call_a"),
                    FakeFunctionCall(type="function_call", name="tavily_search", arguments='{"query": "b"}', call_id="call_b"),
                ]
            )
        return FakeResponse(output=[], output_text="Done.")


@pytest.mark.asyncio
async def test_research_agent_callbacks_run_one_at_a_time():

    events = []
    running = 0

    async def on_tool_call(tool, query, iteration):
        nonlocal running
        running += 1
        assert running == 1, "on_tool_call ran concurrently"
        await asyncio.sleep(0)
        events.append(("call", tool, query, iteration))
        running -= 1
        return f"step-{query}"

    async def on_tool_result(step, execution):
        nonlocal running
        running += 1
        assert running == 1, "on_tool_result ran concurrently"
        await asyncio.sleep(0)
        events.append(("result", step, execution.result.query))
        running -= 1

    result = await ResearchAgent(
        llm=TwoCallsLLM(),
        registry=FakeRegistry(),
        on_tool_call=on_tool_call,
        on_tool_result=on_tool_result,
    ).run(question="q")

    assert result.answer == "Done."

    # Each result is passed the step object its own call returned.
    assert events == [
        ("call", "tavily", "a", 1),
        ("call", "tavily", "b", 1),
        ("result", "step-a", "a"),
        ("result", "step-b", "b"),
    ]


class RaisingTool:

    async def search(self, query):
        raise ConnectionError("arXiv is down")


class TavilyAndArxivLLM:
    """Asks for a Tavily and an arXiv search in one round, then answers."""

    def __init__(self):
        self.calls = 0
        self.requests = []

    async def generate_with_tools(self, input_items, tools, instructions=None):
        self.calls += 1
        self.requests.append(list(input_items))
        if self.calls == 1:
            return FakeResponse(
                output=[
                    FakeFunctionCall(type="function_call", name="tavily_search", arguments='{"query": "rag"}', call_id="call_t"),
                    FakeFunctionCall(type="function_call", name="arxiv_search", arguments='{"query": "rag"}', call_id="call_a"),
                ]
            )
        return FakeResponse(output=[], output_text="Answer from Tavily only.")


@pytest.mark.asyncio
async def test_research_agent_turns_tool_exception_into_failed_result():

    registry = FakeRegistry()
    registry.tools["arxiv"] = RaisingTool()

    llm = TavilyAndArxivLLM()

    result = await ResearchAgent(
        llm=llm,
        registry=registry,
    ).run(question="What is RAG?")

    # The run still finishes, and the other search in the round succeeded.
    assert result.answer == "Answer from Tavily only."

    by_tool = {r.tool: r for r in result.tool_results}

    assert by_tool["tavily"].success is True

    assert by_tool["arxiv"].success is False
    assert by_tool["arxiv"].error == "arXiv is down"

    # The model was told the arXiv search failed.
    outputs = {
        item["call_id"]: item["output"]
        for item in llm.requests[1]
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    }

    assert outputs["call_t"] == "Fake Tavily research result."
    assert outputs["call_a"] == "Error: arxiv search failed: arXiv is down"


class OneSearchLLM:
    """Asks for one Tavily search, then answers."""

    def __init__(self):
        self.calls = 0
        self.requests = []

    async def generate_with_tools(self, input_items, tools, instructions=None):
        self.calls += 1
        self.requests.append(list(input_items))
        if self.calls == 1:
            return FakeResponse(
                output=[
                    FakeFunctionCall(type="function_call", name="tavily_search", arguments='{"query": "rag"}', call_id="call_1"),
                ]
            )
        return FakeResponse(output=[], output_text="Answer.")


class FlakyTool:
    """Fails (raises) a set number of times, then succeeds."""

    def __init__(self, failures: int):
        self.failures = failures
        self.attempts = 0

    async def search(self, query):
        self.attempts += 1
        if self.attempts <= self.failures:
            raise ConnectionError(f"attempt {self.attempts} failed")
        return ToolResult(tool="tavily", query=query, content="Recovered result.")


class SlowTool:

    def __init__(self):
        self.attempts = 0

    async def search(self, query):
        self.attempts += 1
        await asyncio.sleep(1)
        return ToolResult(tool="tavily", query=query, content="Too late.")


@pytest.mark.asyncio
async def test_research_agent_retries_failed_tool():

    tool = FlakyTool(failures=2)
    registry = FakeRegistry()
    registry.tools["tavily"] = tool

    result = await ResearchAgent(llm=OneSearchLLM(), registry=registry).run(question="q")

    # Two failures, then success on the last allowed attempt.
    assert tool.attempts == 3
    assert result.tool_results[0].success is True
    assert result.tool_results[0].content == "Recovered result."


@pytest.mark.asyncio
async def test_research_agent_gives_up_after_retries():

    tool = FlakyTool(failures=10)
    registry = FakeRegistry()
    registry.tools["tavily"] = tool

    result = await ResearchAgent(llm=OneSearchLLM(), registry=registry).run(question="q")

    assert tool.attempts == 3
    assert result.tool_results[0].success is False
    assert result.tool_results[0].error == "attempt 3 failed"


@pytest.mark.asyncio
async def test_research_agent_times_out_slow_tool(monkeypatch):

    tool = SlowTool()
    registry = FakeRegistry()
    registry.tools["tavily"] = tool

    llm = OneSearchLLM()

    result = await ResearchAgent(
        llm=llm,
        registry=registry,
        tool_timeout_seconds=0.01,
    ).run(question="q")

    # Every attempt timed out; the run still finished with an answer.
    assert tool.attempts == 3
    assert result.answer == "Answer."
    assert result.tool_results[0].success is False
    assert result.tool_results[0].error == "Tool execution timed out after 0.01 seconds."
    assert llm.requests[1][-1]["output"] == (
        "Error: tavily search failed: Tool execution timed out after 0.01 seconds."
    )


class ManySearchesLLM:
    """Asks for 5 searches per round until no tools are offered."""

    def __init__(self):
        self.calls = 0
        self.tools_offered = []
        self.requests = []

    async def generate_with_tools(self, input_items, tools, instructions=None):
        self.calls += 1
        self.tools_offered.append(len(tools))
        self.requests.append(list(input_items))
        if not tools:
            return FakeResponse(output=[], output_text="Answer from 8 searches.")
        return FakeResponse(
            output=[
                FakeFunctionCall(
                    type="function_call",
                    name="tavily_search",
                    arguments=f'{{"query": "q{self.calls}-{i}"}}',
                    call_id=f"call_{self.calls}_{i}",
                )
                for i in range(5)
            ]
        )


@pytest.mark.asyncio
async def test_research_agent_enforces_tool_call_limit():

    llm = ManySearchesLLM()

    result = await ResearchAgent(llm=llm, registry=FakeRegistry()).run(question="q")

    # 5 searches in round 1, 3 in round 2 (2 skipped), then no tools offered.
    assert len(result.tool_results) == 8
    assert llm.tools_offered == [3, 3, 0]
    assert result.answer == "Answer from 8 searches."

    # The model requested 10 calls in total; all are counted.
    assert result.tool_call_count == 10

    skipped = [
        item["output"]
        for item in llm.requests[2]
        if isinstance(item, dict)
        and item.get("output", "").startswith("Error: tool call limit")
    ]

    assert len(skipped) == 2
