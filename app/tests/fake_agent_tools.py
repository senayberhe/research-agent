from app.tools.base import ToolResult


class FakeTavilyTool:

    @property
    def name(self):
        return "tavily"

    async def search(
        self,
        query: str,
    ) -> ToolResult:

        return ToolResult(
            tool="tavily",
            query=query,
            content=(
                "Fake Tavily research result."
            ),
            success=True,
        )


class FakeRegistry:

    def __init__(self):
        self.tools = {
            "tavily": FakeTavilyTool(),
        }

    def get(self, name: str):
        return self.tools[name]