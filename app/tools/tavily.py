import asyncio

import httpx

from app.core.config import settings

from app.tools.base import ResearchTool, ToolResult


class TavilyTool(ResearchTool):

    MAX_RETRIES = 3

    @property
    def name(self) -> str:
        return "tavily"

    async def search(self, query: str) -> ToolResult:

        url = "https://api.tavily.com/search"

        payload = {
            "api_key": settings.tavily_api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": 5,
        }

        for attempt in range(self.MAX_RETRIES + 1):

            try:

                async with httpx.AsyncClient(
                    timeout=30.0
                ) as client:

                    response = await client.post(
                        url,
                        json=payload,
                    )

                    response.raise_for_status()

                    data = response.json()

                break

            except httpx.TimeoutException:

                if attempt == self.MAX_RETRIES:
                    return ToolResult(
                        tool=self.name,
                        query=query,
                        content="",
                        success=False,
                        error="Tavily request timed out",
                    )

                delay = 2 ** attempt

                await asyncio.sleep(delay)

            except httpx.RequestError as exc:

                if attempt == self.MAX_RETRIES:
                    return ToolResult(
                        tool=self.name,
                        query=query,
                        content="",
                        success=False,
                        error=f"Tavily request failed: {exc}",
                    )

                delay = 2 ** attempt

                await asyncio.sleep(delay)

            except httpx.HTTPStatusError as exc:

                status_code = exc.response.status_code

                if status_code in {429, 500, 502, 503, 504}:

                    if attempt == self.MAX_RETRIES:
                        return ToolResult(
                            tool=self.name,
                            query=query,
                            content="",
                            success=False,
                            error=(
                                f"Tavily returned HTTP "
                                f"{status_code} after retries"
                            ),
                        )

                    delay = 2 ** attempt

                    await asyncio.sleep(delay)

                else:

                    return ToolResult(
                        tool=self.name,
                        query=query,
                        content="",
                        success=False,
                        error=(
                            f"Tavily returned HTTP "
                            f"{status_code}"
                        ),
                    )

            except ValueError:

                return ToolResult(
                    tool=self.name,
                    query=query,
                    content="",
                    success=False,
                    error="Tavily returned invalid JSON",
                )

        results = data.get("results", [])

        content_parts = []

        for result in results:

            title = result.get("title", "")
            content = result.get("content", "")
            result_url = result.get("url", "")

            content_parts.append(
                f"Title: {title}\n"
                f"URL: {result_url}\n"
                f"Content: {content}"
            )

        content = "\n\n".join(content_parts)

        return ToolResult(
            tool=self.name,
            query=query,
            content=content,
            success=True,
        )