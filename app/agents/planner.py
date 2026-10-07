import logging
from dataclasses import dataclass
from enum import Enum

from app.agents.planner_prompt import SYSTEM_PROMPT, build_planning_prompt
from app.agents.planner_schema import RESEARCH_PLAN_SCHEMA
from app.llm.base import LLMProvider


logger = logging.getLogger(__name__)


class ToolName(str, Enum):
    TAVILY = "tavily"
    ARXIV = "arxiv"
    WIKIPEDIA = "wikipedia"


@dataclass
class ResearchStep:
    tool: ToolName
    query: str


@dataclass
class ResearchPlan:
    steps: list[ResearchStep]


def create_research_plan(question: str) -> ResearchPlan:
    return ResearchPlan(
        steps=[
            ResearchStep(
                tool=ToolName.TAVILY,
                query=question
            ),
            ResearchStep(
                tool=ToolName.ARXIV,
                query=question
            ),
            ResearchStep(
                tool=ToolName.WIKIPEDIA,
                query=question
            ),
        ]
    )


class LLMResearchPlanner:

    # Matches the "no more than 5 research steps" rule in SYSTEM_PROMPT.
    MAX_STEPS = 5

    def __init__(
            self,
            llm: LLMProvider,
    ):
        self.llm = llm

    async def create_plan(
        self,
        question: str,
    ) -> ResearchPlan:

        # If the LLM call fails or returns nothing usable, fall back to the
        # fixed plan so research still runs.
        try:
            response = await self.llm.generate_structured(
                prompt=build_planning_prompt(question),
                schema=RESEARCH_PLAN_SCHEMA,
                instructions=SYSTEM_PROMPT,
            )

            steps = self._parse_steps(response)

        except Exception as exc:
            logger.warning(
                "LLM planning failed, using default plan: %s",
                exc,
            )
            return create_research_plan(question)

        if not steps:
            logger.warning(
                "LLM returned an empty plan, using default plan"
            )
            return create_research_plan(question)

        return ResearchPlan(
            steps=steps,
        )

    def _parse_steps(
        self,
        response: dict,
    ) -> list[ResearchStep]:

        steps = []
        seen = set()

        for item in response["steps"]:

            # Raises ValueError for a tool name we don't know.
            tool = ToolName(item["tool"])

            query = item["query"].strip()

            # Skip empty queries and repeats of the same search.
            key = (tool, query.lower())

            if not query or key in seen:
                continue

            seen.add(key)

            steps.append(
                ResearchStep(
                    tool=tool,
                    query=query,
                )
            )

        return steps[:self.MAX_STEPS]
