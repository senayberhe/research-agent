import asyncio
import logging

from app.tools.base import ToolResult
from app.tools.registry import ToolRegistry
logger = logging.getLogger(__name__)

class ResearchExecutor:

    MAX_RETRIES = 3

    def __init__(
        self,
        registry: ToolRegistry,
    ):
        self.registry = registry

    async def execute_step(
        self,
        tool_name: str,
        query: str,
    ) -> ToolResult:

        tool = self.registry.get(
            tool_name
        )

        result = None

        # -------------------------
        # Retry tool
        # -------------------------

        for attempt in range(
            self.MAX_RETRIES
        ):

            result = await tool.search(
                query
            )

            if result.success:
                break

            logger.warning(
                "Tool %s failed (attempt %d/%d): %s",
                tool_name,
                attempt + 1,
                self.MAX_RETRIES,
                result.error,
            )

            if attempt < self.MAX_RETRIES - 1:

                delay = 2 ** attempt

                await asyncio.sleep(
                    delay
                )

        if result.success:

            logger.info(
                "Tool %s succeeded for query: %s",
                tool_name,
                query,
            )

        else:

            logger.error(
                "Tool %s gave up after %d attempts for query: %s",
                tool_name,
                self.MAX_RETRIES,
                query,
            )

        return result

    async def execute(
        self,
        steps,
    ) -> list[ToolResult]:

        tasks = [
            self.execute_step(
                tool_name=step.tool,
                query=step.query,
            )
            for step in steps
        ]

        results = await asyncio.gather(
            *tasks
        )

        succeeded = sum(
            1
            for result in results
            if result.success
        )

        logger.info(
            "Executed %d steps: %d succeeded, %d failed",
            len(results),
            succeeded,
            len(results) - succeeded,
        )

        return results