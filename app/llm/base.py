from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        instructions: str | None = None,
    ) -> str:
        raise NotImplementedError

    # Optional capabilities: not abstract, so simple providers (and test
    # fakes) only need generate(). Calling one that isn't implemented raises.

    async def generate_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        instructions: str | None = None,
        name: str = "research_plan",
    ) -> dict[str, Any]:
        raise NotImplementedError(
            f"{type(self).__name__} does not support structured output."
        )

    async def generate_with_tools(
        self,
        input_items: list[Any],
        tools: list[dict[str, Any]],
        instructions: str | None = None,
    ) -> Any:
        raise NotImplementedError(
            f"{type(self).__name__} does not support tool calling."
        )
