import pytest

from app.agents.planner import (
    LLMResearchPlanner,
    ResearchPlan,
    ResearchStep,
    ToolName,
    create_research_plan
)
from app.agents.planner_prompt import SYSTEM_PROMPT, build_planning_prompt
from app.agents.planner_schema import RESEARCH_PLAN_SCHEMA

def test_create_research_plan():
    question = "How do modern RAG systems work?"
    plan = create_research_plan(question)
    assert isinstance(plan, ResearchPlan)
    assert all(isinstance(step, ResearchStep) for step in plan.steps)
    assert all(step.query == question for step in plan.steps)

    for step in plan.steps:
        assert step.query==question
        assert step.tool in ToolName

class FakePlannerLLM:
    """Returns a fixed plan from generate_structured, or raises."""

    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.kwargs = None

    async def generate_structured(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.response


@pytest.mark.asyncio
async def test_llm_planner_builds_plan_from_llm_response():
    llm = FakePlannerLLM(
        response={
            "steps": [
                {"tool": "arxiv", "query": "retrieval augmented generation survey"},
                {"tool": "wikipedia", "query": "Retrieval-augmented generation"},
                # Duplicate (case-insensitive) and empty queries are dropped.
                {"tool": "arxiv", "query": "Retrieval Augmented Generation Survey"},
                {"tool": "tavily", "query": "   "},
            ]
        }
    )

    plan = await LLMResearchPlanner(llm=llm).create_plan("What is RAG?")

    assert plan.steps == [
        ResearchStep(tool=ToolName.ARXIV, query="retrieval augmented generation survey"),
        ResearchStep(tool=ToolName.WIKIPEDIA, query="Retrieval-augmented generation"),
    ]

    assert llm.kwargs["instructions"] == SYSTEM_PROMPT
    assert llm.kwargs["prompt"] == build_planning_prompt("What is RAG?")
    assert llm.kwargs["schema"] == RESEARCH_PLAN_SCHEMA


@pytest.mark.asyncio
async def test_llm_planner_caps_steps():
    llm = FakePlannerLLM(
        response={
            "steps": [
                {"tool": "tavily", "query": f"query {i}"}
                for i in range(8)
            ]
        }
    )

    plan = await LLMResearchPlanner(llm=llm).create_plan("q")

    assert len(plan.steps) == LLMResearchPlanner.MAX_STEPS


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "llm",
    [
        FakePlannerLLM(error=RuntimeError("LLM service unavailable")),
        FakePlannerLLM(response={"steps": []}),
        FakePlannerLLM(response={"steps": [{"tool": "google", "query": "q"}]}),
    ],
    ids=["llm-error", "empty-plan", "unknown-tool"],
)
async def test_llm_planner_falls_back_to_default_plan(llm):
    question = "What is RAG?"

    plan = await LLMResearchPlanner(llm=llm).create_plan(question)

    assert plan == create_research_plan(question)
