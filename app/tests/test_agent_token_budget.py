import pytest

from app.agents.research_agent import (
    ResearchAgent,
    TokenBudgetExceeded,
    TokenUsage,
)
from app.core.config import settings
from app.db.models import TaskStatus
from app.services import workflow_service
from app.tests.fake_agent_llm import (
    FakeAgentLLM,
    FakeFunctionCall,
    FakeResponse,
)
from app.tests.fake_agent_tools import FakeRegistry
from app.tests.test_workflow_service import create_task, load_steps


class SearchForeverLLM:
    """Always asks for another search; each call uses 1000 tokens."""

    def __init__(self):
        self.calls = 0

    async def generate_with_tools(self, input_items, tools, instructions=None):

        self.calls += 1

        return FakeResponse(
            output=[
                FakeFunctionCall(
                    type="function_call",
                    name="tavily_search",
                    arguments=f'{{"query": "rag {self.calls}"}}',
                    call_id=f"call_{self.calls}",
                )
            ],
            input_tokens=800,
            output_tokens=200,
        )


class CountingRegistry(FakeRegistry):

    def __init__(self):
        super().__init__()
        self.searches = 0
        tool = self.tools["tavily"]
        search = tool.search

        async def counting_search(query):
            self.searches += 1
            return await search(query)

        tool.search = counting_search


# -------------------------
# Tracking
# -------------------------


@pytest.mark.asyncio
async def test_usage_is_summed_over_llm_calls():

    agent = ResearchAgent(llm=FakeAgentLLM(), registry=FakeRegistry())

    result = await agent.run(question="What is RAG?")

    assert result.success is True

    # Two LLM calls of 100 input + 50 output each.
    assert result.usage.input_tokens == 200
    assert result.usage.output_tokens == 100
    assert result.usage.total_tokens == 300


@pytest.mark.asyncio
async def test_estimated_cost_uses_model_pricing():

    # FakeAgentLLM's model is gpt-5.4-mini: $0.75 input, $4.50 output per 1M.
    agent = ResearchAgent(llm=FakeAgentLLM(), registry=FakeRegistry())

    result = await agent.run(question="What is RAG?")

    # 200 * $0.75/1M + 100 * $4.50/1M
    assert result.usage.estimated_cost == pytest.approx(0.0006)


@pytest.mark.asyncio
async def test_estimated_cost_is_unknown_for_unpriced_model():

    agent = ResearchAgent(
        llm=FakeAgentLLM(model="gpt-5.6"),
        registry=FakeRegistry(),
    )

    result = await agent.run(question="What is RAG?")

    # Tokens are still counted; only the cost is unknown.
    assert result.usage.total_tokens == 300
    assert result.usage.estimated_cost is None


def make_agent() -> ResearchAgent:
    return ResearchAgent(llm=FakeAgentLLM(), registry=FakeRegistry())


def test_extract_usage_reads_response_usage():

    usage = make_agent()._extract_usage(
        FakeResponse(
            output=[],
            input_tokens=100,
            output_tokens=20,
        )
    )

    assert usage.input_tokens == 100
    assert usage.output_tokens == 20
    assert usage.total_tokens == 120
    # gpt-5.4-mini: 100 * $0.75/1M + 20 * $4.50/1M
    assert usage.estimated_cost == pytest.approx(0.000165)


def test_extract_usage_charges_cached_tokens_at_cached_price():

    usage = make_agent()._extract_usage(
        FakeResponse(
            output=[],
            input_tokens=1_000,
            cached_tokens=800,
            output_tokens=100,
        )
    )

    # gpt-5.4-mini: 200 uncached * $0.75 + 800 cached * $0.075
    # + 100 output * $4.50, per 1M.
    assert usage.estimated_cost == pytest.approx(0.00066)


def test_extract_usage_prefers_model_reported_by_response():

    # The agent's provider says gpt-5.4-mini, but the response names the
    # snapshot that actually ran: gpt-5.4 ($2.50 input, $15 output).
    usage = make_agent()._extract_usage(
        FakeResponse(
            output=[],
            input_tokens=100,
            output_tokens=20,
            model="gpt-5.4-2026-03-05",
        )
    )

    assert usage.estimated_cost == pytest.approx(0.00055)


def test_missing_usage_counts_as_zero():

    # FakeResponse always has usage, so remove it to simulate a response
    # that doesn't report any.
    response = FakeResponse(output=[])
    response.usage = None

    usage = make_agent()._extract_usage(response)

    assert usage == TokenUsage()


def test_total_is_input_plus_output_when_not_reported():

    response = FakeResponse(output=[], input_tokens=10, output_tokens=5)
    response.usage.total_tokens = 0

    usage = make_agent()._extract_usage(response)

    assert usage.total_tokens == 15


def test_add_usage_sums_tokens_and_cost():

    agent = make_agent()

    usage = TokenUsage()

    agent._add_usage(
        usage,
        TokenUsage(100, 20, 120, estimated_cost=0.001),
    )
    agent._add_usage(
        usage,
        TokenUsage(200, 50, 250, estimated_cost=0.002),
    )

    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (
        300,
        70,
        370,
    )
    assert usage.estimated_cost == pytest.approx(0.003)


def test_one_unpriced_call_makes_run_cost_unknown():

    agent = make_agent()

    usage = TokenUsage()

    agent._add_usage(usage, TokenUsage(100, 20, 120, estimated_cost=0.001))
    agent._add_usage(usage, TokenUsage(100, 20, 120, estimated_cost=None))
    agent._add_usage(usage, TokenUsage(100, 20, 120, estimated_cost=0.001))

    # A partial total would understate the cost, so it stays unknown.
    assert usage.estimated_cost is None
    assert usage.total_tokens == 360


# -------------------------
# Budget protection
# -------------------------


@pytest.mark.asyncio
async def test_run_stops_when_token_budget_is_exceeded():

    llm = SearchForeverLLM()
    registry = CountingRegistry()

    agent = ResearchAgent(llm=llm, registry=registry, max_total_tokens=2500)

    result = await agent.run(question="What is RAG?")

    # 1000 tokens per call: the 3rd call reaches 3000 >= 2500.
    assert llm.calls == 3
    assert result.success is False
    assert result.answer == ""
    assert result.error == "Agent token budget exceeded: 3000 >= 2500"
    assert result.iteration_count == 3
    assert result.usage.total_tokens == 3000

    # The 3rd round's search was never run; earlier results are kept.
    assert registry.searches == 2
    assert [r.query for r in result.tool_results] == ["rag 1", "rag 2"]


@pytest.mark.asyncio
async def test_budget_stop_skips_tool_callbacks():

    calls = []

    async def on_tool_call(tool, query, iteration):
        calls.append(query)

    agent = ResearchAgent(
        llm=SearchForeverLLM(),
        registry=FakeRegistry(),
        on_tool_call=on_tool_call,
        max_total_tokens=1000,
    )

    result = await agent.run(question="What is RAG?")

    # The first response already uses the whole budget: no step is created.
    assert result.success is False
    assert calls == []


@pytest.mark.asyncio
async def test_final_answer_over_budget_is_still_returned():

    # FakeAgentLLM uses 150 tokens per call; the answer takes the run to 300.
    agent = ResearchAgent(
        llm=FakeAgentLLM(),
        registry=FakeRegistry(),
        max_total_tokens=200,
    )

    result = await agent.run(question="What is RAG?")

    assert result.success is True
    assert result.answer.startswith("RAG improves factuality")
    assert result.usage.total_tokens == 300


@pytest.mark.asyncio
async def test_run_under_budget_is_unaffected():

    agent = ResearchAgent(
        llm=FakeAgentLLM(),
        registry=FakeRegistry(),
        max_total_tokens=10_000,
    )

    result = await agent.run(question="What is RAG?")

    assert result.success is True
    assert result.error is None
    assert len(result.tool_results) == 1


@pytest.mark.asyncio
async def test_workflow_fails_task_when_budget_exceeded(db_session, monkeypatch):

    monkeypatch.setattr(workflow_service, "OpenAIProvider", SearchForeverLLM)
    monkeypatch.setattr(workflow_service, "ToolRegistry", FakeRegistry)
    monkeypatch.setattr(settings, "agent_max_total_tokens", 2500)

    task = await create_task(db_session)

    await workflow_service.execute_research_workflow(db=db_session, task=task)

    assert task.status == TaskStatus.FAILED
    assert task.summary is None

    # Steps from the rounds before the budget ran out are saved.
    steps = await load_steps(db_session, task.id)

    assert [step.query for step in steps] == ["rag 1", "rag 2"]


# -------------------------
# _check_token_budget
# -------------------------


def test_check_token_budget_allows_usage_under_budget():

    agent = ResearchAgent(
        llm=FakeAgentLLM(),
        registry=FakeRegistry(),
        max_total_tokens=20_000,
    )

    # No exception.
    agent._check_token_budget(TokenUsage(total_tokens=19_999))


def test_check_token_budget_refuses_at_budget():

    agent = ResearchAgent(
        llm=FakeAgentLLM(),
        registry=FakeRegistry(),
        max_total_tokens=20_000,
    )

    with pytest.raises(TokenBudgetExceeded) as exc_info:
        agent._check_token_budget(TokenUsage(total_tokens=20_000))

    assert str(exc_info.value) == "Agent token budget exceeded: 20000 >= 20000"

    # Still a RuntimeError, for code that catches that.
    assert isinstance(exc_info.value, RuntimeError)


@pytest.mark.asyncio
async def test_agent_makes_no_llm_call_after_budget_is_reached():

    llm = SearchForeverLLM()

    agent = ResearchAgent(
        llm=llm,
        registry=FakeRegistry(),
        max_iterations=25,
        max_tool_calls=25,
        max_total_tokens=20_000,
    )

    result = await agent.run(question="What is RAG?")

    # 1000 tokens per call: the 20th call reaches 20000, and with 25
    # iterations allowed, the budget (not the iteration limit) stops it.
    assert llm.calls == 20
    assert result.success is False
    assert result.usage.total_tokens == 20_000
    assert result.error == "Agent token budget exceeded: 20000 >= 20000"

    # The 20th call's search never ran.
    assert len(result.tool_results) == 19
