from app.tools.arxiv import ArxivTool
from app.tools.tavily import TavilyTool
from app.tools.wikipedia import WikipediaTool
from app.tools.base import ResearchTool


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str,ResearchTool]={
            "tavily": TavilyTool(),
            "arxiv": ArxivTool(),
            "wikipedia": WikipediaTool()
        }

    def get(self, name: str) -> ResearchTool:
        try:
            return self._tools[name]
        except KeyError:
            raise ValueError(f"Tool '{name}' not found in the registry.")

    def all(self) -> list[ResearchTool]:
        return list(self._tools.values())