import json
import logging
from dataclasses import dataclass

from app.agents.tool_definitions import (
    RESEARCH_TOOL_DEFINITIONS,
)
from app.agents.tool_mapping import (
    TOOL_NAME_MAPPING,
)
from app.llm.base import LLMProvider
from app.tools.base import ToolResult
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    answer: str
    tool_results: list[ToolResult]


class ResearchAgent:

    MAX_ITERATIONS = 6

    def __init__(
        self,
        llm: LLMProvider,
        registry: ToolRegistry,
    ):
        self.llm = llm
        self.registry = registry

    async def run(
        self,
        question: str,
    ) -> AgentResult:

        input_items = [
            {
                "role": "system",
                "content": (
                    "You are a research agent. "
                    "Use the available research tools to gather "
                    "reliable evidence before answering. "
                    "Do not invent facts. "
                    "Use additional tools when the available "
                    "evidence is insufficient."
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ]

        tool_results: list[ToolResult] = []

        for iteration in range(self.MAX_ITERATIONS):

            logger.info(
                "Agent iteration=%s",
                iteration + 1,
            )

            response = await self.llm.generate_with_tools(
                input_items=input_items,
                tools=RESEARCH_TOOL_DEFINITIONS,
            )

            input_items.extend(response.output)

            tool_calls = [
                item
                for item in response.output
                if item.type == "function_call"
            ]

            if not tool_calls:
                logger.info(
                    "Agent completed after iteration=%s",
                    iteration + 1,
                )

                return AgentResult(
                    answer=response.output_text,
                    tool_results=tool_results,
                )

            for tool_call in tool_calls:

                tool_name = tool_call.name

                if tool_name not in TOOL_NAME_MAPPING:
                    raise ValueError(
                        f"Unknown agent tool: {tool_name}"
                    )

                arguments = json.loads(
                    tool_call.arguments
                )

                query = arguments["query"]

                registry_name = TOOL_NAME_MAPPING[
                    tool_name
                ]

                logger.info(
                    "Agent requested tool=%s query=%s",
                    registry_name,
                    query,
                )

                tool = self.registry.get(
                    registry_name
                )

                result = await tool.search(query)

                tool_results.append(result)

                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call.call_id,
                        "output": result.content,
                    }
                )

        raise RuntimeError(
            "Research agent exceeded maximum iterations."
        )