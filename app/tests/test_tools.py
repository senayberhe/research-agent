
from types import SimpleNamespace

import httpx
import pytest
from app.tools.arxiv import ArxivTool
from app.tools.tavily import TavilyTool
from app.tools.wikipedia import WikipediaTool
from app.tools.registry import ToolRegistry

@pytest.mark.asyncio
async def test_tavily_tool(monkeypatch):
    async def fake_post(self, url, json):
        assert json["query"] == query
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "RAG Overview",
                        "url": "https://example.com/rag",
                        "content": "Retrieval-augmented generation explained.",
                    }
                ]
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    tool = TavilyTool()
    query = "test query"
    result = await tool.search(query)
    assert result.tool == tool.name
    assert result.query == query
    assert "Title: RAG Overview" in result.content
    assert "URL: https://example.com/rag" in result.content
    assert "Content: Retrieval-augmented generation explained." in result.content


@pytest.mark.asyncio
async def test_arxiv_tool(monkeypatch):
    fake_paper = SimpleNamespace(
        title="Retrieval-Augmented Generation",
        authors=[SimpleNamespace(name="Ada Lovelace"), SimpleNamespace(name="Alan Turing")],
        published="2020-05-22",
        entry_id="http://arxiv.org/abs/2005.11401",
        summary="We explore RAG models.",
    )

    def fake_search(self, query):
        assert query == "test query"
        return [fake_paper]

    monkeypatch.setattr(ArxivTool, "_search", fake_search)

    tool = ArxivTool()
    query = "test query"
    result = await tool.search(query)
    assert result.success is True
    assert result.tool == tool.name
    assert result.query == query
    assert "Title: Retrieval-Augmented Generation" in result.content
    assert "Authors: Ada Lovelace, Alan Turing" in result.content
    assert "URL: http://arxiv.org/abs/2005.11401" in result.content
    assert "Summary: We explore RAG models." in result.content


@pytest.mark.asyncio
async def test_arxiv_tool_failure(monkeypatch):
    def failing_search(self, query):
        raise RuntimeError("arXiv is down")

    monkeypatch.setattr(ArxivTool, "_search", failing_search)

    result = await ArxivTool().search("test query")
    assert result.success is False
    assert result.content == ""
    assert "arXiv is down" in result.error

@pytest.mark.asyncio
async def test_wikipedia_tool(monkeypatch):
    async def fake_get(self, url, params):
        assert params["gsrsearch"] == query
        return httpx.Response(
            200,
            json={
                "query": {
                    "pages": [
                        {
                            "title": "Vector database",
                            "index": 2,
                            "fullurl": "https://en.wikipedia.org/wiki/Vector_database",
                            "extract": "A vector database stores embeddings.",
                        },
                        {
                            "title": "RAG (disambiguation)",
                            "index": 3,
                            "fullurl": "https://en.wikipedia.org/wiki/RAG",
                            "extract": "RAG may refer to:",
                            "pageprops": {"disambiguation": ""},
                        },
                        {
                            "title": "Retrieval-augmented generation",
                            "index": 1,
                            "fullurl": "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
                            "extract": "RAG is a technique that grounds LLM output in retrieved documents.",
                        },
                    ]
                }
            },
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

    tool = WikipediaTool()
    query = "test query"
    result = await tool.search(query)
    assert result.success is True
    assert result.tool == tool.name
    assert result.query == query
    assert "Title: Retrieval-augmented generation" in result.content
    assert "URL: https://en.wikipedia.org/wiki/Retrieval-augmented_generation" in result.content
    assert "Summary: RAG is a technique" in result.content
    # Pages are ordered by search rank, and disambiguation pages are dropped.
    assert result.content.index("Retrieval-augmented generation") < result.content.index("Vector database")
    assert "may refer to" not in result.content


@pytest.mark.asyncio
async def test_wikipedia_tool_truncates_content(monkeypatch):
    long_page = {"title": "Long", "fullurl": "https://example.com", "extract": "x" * 20000}

    async def fake_search(self, query):
        return [long_page]

    monkeypatch.setattr(WikipediaTool, "_search", fake_search)

    result = await WikipediaTool().search("test query")
    assert len(result.content) == WikipediaTool.MAX_CONTENT_LENGTH


@pytest.mark.asyncio
async def test_wikipedia_tool_failure(monkeypatch):
    async def failing_get(self, url, params):
        raise httpx.ConnectError("Wikipedia is down")

    monkeypatch.setattr(httpx.AsyncClient, "get", failing_get)

    result = await WikipediaTool().search("test query")
    assert result.success is False
    assert result.content == ""
    assert "Wikipedia is down" in result.error

def test_tool_registry():
    registry = ToolRegistry()
    all_tools = registry.all()
    assert any(isinstance(tool, TavilyTool) for tool in all_tools)
    assert any(isinstance(tool, ArxivTool) for tool in all_tools)
    assert any(isinstance(tool, WikipediaTool) for tool in all_tools)

    tavily_tool = registry.get("tavily")
    assert isinstance(tavily_tool, TavilyTool)

    arxiv_tool = registry.get("arxiv")
    assert isinstance(arxiv_tool, ArxivTool)

    wikipedia_tool = registry.get("wikipedia")
    assert isinstance(wikipedia_tool, WikipediaTool)

    with pytest.raises(ValueError):
        registry.get("nonexistent_tool")
