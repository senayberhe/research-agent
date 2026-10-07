import httpx

from app.tools.base import ResearchTool, ToolResult


class WikipediaTool(ResearchTool):

    API_URL = "https://en.wikipedia.org/w/api.php"
    # Wikipedia's API policy requires a descriptive User-Agent.
    USER_AGENT = "research-agent/0.1 (personal research project)"

    MAX_RESULTS = 3
    MAX_CONTENT_LENGTH = 8000

    @property
    def name(self) -> str:
        return "wikipedia"

    async def search(self, query: str) -> ToolResult:

        try:
            pages = await self._search(query)

        except Exception as exc:

            return ToolResult(
                tool=self.name,
                query=query,
                content="",
                success=False,
                error=f"Wikipedia search failed: {exc}",
            )

        content_parts = []

        for page in pages:

            content_parts.append(
                f"Title: {page['title']}\n"
                f"URL: {page['fullurl']}\n"
                f"Summary: {page['extract']}"
            )

        content = "\n\n".join(content_parts)

        content = content[:self.MAX_CONTENT_LENGTH]

        return ToolResult(
            tool=self.name,
            query=query,
            content=content,
            success=True,
        )

    async def _search(self, query: str) -> list[dict]:

        # One request: search for matching pages and fetch each page's
        # intro text, URL and disambiguation flag at the same time.
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "generator": "search",
            "gsrsearch": query,
            "gsrlimit": self.MAX_RESULTS,
            "prop": "extracts|info|pageprops",
            "exintro": "1",
            "explaintext": "1",
            "inprop": "url",
            "ppprop": "disambiguation",
            "redirects": "1",
        }

        async with httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": self.USER_AGENT},
        ) as client:

            response = await client.get(
                self.API_URL,
                params=params,
            )

            response.raise_for_status()

            data = response.json()

        pages = data.get("query", {}).get("pages", [])

        pages = [
            page
            for page in pages
            if "disambiguation" not in page.get("pageprops", {})
        ]

        # The API returns pages unordered; "index" is the search rank.
        return sorted(
            pages,
            key=lambda page: page.get("index", 0),
        )
