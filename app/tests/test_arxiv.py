import asyncio

from app.tools.arxiv import ArxivTool


async def main():

    tool = ArxivTool()

    result = await tool.search(
        "retrieval augmented generation"
    )

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

    assert result.tool == "arxiv"
    assert result.query == "retrieval augmented generation"
    assert result.success is True
    assert result.error is None
    assert result.content


if __name__ == "__main__":
    asyncio.run(main())