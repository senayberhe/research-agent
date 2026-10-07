from app.tools.base import ResearchTool, ToolResult

class FakeTool(ResearchTool):
    def __init__(
            self,
            tool_name: str,
    ):
        self.tool_name = tool_name

    @property
    def name(self) -> str:
        return self.tool_name

    async def search(
            self,
            query: str,
    ) -> ToolResult:
        return ToolResult(
            tool=self.tool_name,
            query=query,
            content=f"Fake result for query: {query}",
            success = True,
            error=None,
        )