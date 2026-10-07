import pytest

from app.agents.synthesizer import ResearchSynthesizer
from app.tools.base import ToolResult
from app.tests.fake_llm import FailingLLMProvider, FakeLLMProvider


@pytest.mark.asyncio
async def test_synthesizer_uses_llm():
    llm = FakeLLMProvider()

    synthesizer = ResearchSynthesizer(
        llm=llm,
    )

    results = [
        ToolResult(
            tool="tavily",
            query="What is RAG?",
            content="RAG retrieves external information.",
            success=True,
        ),
        ToolResult(
            tool="arxiv",
            query="What is RAG?",
            content="RAG combines retrieval and generation.",
            success=True,
        ),
        ToolResult(
            tool="wikipedia",
            query="What is RAG?",
            content="",
            success=False,
            error="timeout",
        ),
    ]

    result = await synthesizer.synthesize(
        question="What is RAG?",
        results=results,
    )

    assert result.success is True

    assert (
        result.summary
        == "This is a fake synthesized research summary."
    )

    # The prompt includes the question and every successful source,
    # and leaves out the failed one.
    prompt = llm.prompts[0]

    assert "What is RAG?" in prompt
    assert "RAG retrieves external information." in prompt
    assert "RAG combines retrieval and generation." in prompt
    assert "SOURCE: wikipedia" not in prompt


@pytest.mark.asyncio
async def test_synthesizer_without_successful_results_skips_llm():
    llm = FakeLLMProvider()

    result = await ResearchSynthesizer(llm=llm).synthesize(
        question="What is RAG?",
        results=[
            ToolResult(tool="tavily", query="q", content="", success=False, error="timeout"),
        ],
    )

    assert result.success is False
    assert result.error == "No successful research results available."
    assert llm.prompts == []


@pytest.mark.asyncio
async def test_synthesizer_handles_llm_failure():
    result = await ResearchSynthesizer(llm=FailingLLMProvider()).synthesize(
        question="What is RAG?",
        results=[
            ToolResult(tool="tavily", query="q", content="Some content."),
        ],
    )

    assert result.success is False
    assert result.summary == ""
    assert result.error == "LLM service unavailable"
