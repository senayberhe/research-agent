from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ToolResult:
    tool: str
    query: str
    content: str
    success: bool = True
    error: str | None = None


class ResearchTool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    async def search(self, query: str) -> ToolResult:
        pass
