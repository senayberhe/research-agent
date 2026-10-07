import asyncio

import arxiv

from app.tools.base import ResearchTool, ToolResult


class ArxivTool(ResearchTool):

    MAX_RESULTS = 5

    @property
    def name(self) -> str:
        return "arxiv"

    async def search(self, query: str) -> ToolResult:

        try:
            results = await asyncio.to_thread(
                self._search,
                query,
            )

        except Exception as exc:

            return ToolResult(
                tool=self.name,
                query=query,
                content="",
                success=False,
                error=f"arXiv search failed: {exc}",
            )

        content_parts = []

        for paper in results:

            authors = ", ".join(
                author.name
                for author in paper.authors
            )

            content_parts.append(
                f"Title: {paper.title}\n"
                f"Authors: {authors}\n"
                f"Published: {paper.published}\n"
                f"URL: {paper.entry_id}\n"
                f"Summary: {paper.summary}"
            )

        content = "\n\n".join(content_parts)

        return ToolResult(
            tool=self.name,
            query=query,
            content=content,
            success=True,
        )

    def _search(self, query: str):

        client = arxiv.Client(
            page_size=self.MAX_RESULTS,
            num_retries=3,
            delay_seconds=3,
        )

        search = arxiv.Search(
            query=query,
            max_results=self.MAX_RESULTS,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        return list(
            client.results(search)
        )