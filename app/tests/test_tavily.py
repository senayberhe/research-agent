import asyncio
from app.tools.tavily import TavilyTool


async def main():
    tool = TavilyTool()
    result = await tool.search("How do modern RAG systems work?")

    print("\nTOOL:")
    print(result.tool)

    print("\nQUERY:")
    print(result.query)

    print("\nSUCCESS:")
    print(result.success)

    print("\nERROR:")
    print(result.error)

    print("\nCONTENT:")
    print(result.content)

    assert result.tool=="tavily"
    assert result.query=="How do modern RAG systems work?"
    assert result.success is True or result.success is False
    assert result.error is None or isinstance(result.error, str)
    assert isinstance(result.content, str)

if __name__ == "__main__":
    asyncio.run(main())
