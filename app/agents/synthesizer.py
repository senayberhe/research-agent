from dataclasses import dataclass

from app.llm.base import LLMProvider
from app.tools.base import ToolResult


@dataclass
class SynthesisResult:
    summary: str
    success: bool = True
    error: str | None = None


class ResearchSynthesizer:

    def __init__(
        self,
        llm: LLMProvider,
    ):
        self.llm = llm

    async def synthesize(
        self,
        question: str,
        results: list[ToolResult],
    ) -> SynthesisResult:

        successful_results = [
            result
            for result in results
            if result.success and result.content
        ]

        if not successful_results:
            return SynthesisResult(
                summary="",
                success=False,
                error="No successful research results available.",
            )

        sections = []

        for result in successful_results:
            sections.append(
                f"""
SOURCE: {result.tool}

QUERY:
{result.query}

CONTENT:
{result.content}
"""
            )

        research_context = "\n\n".join(sections)

        prompt = f"""
You are a research synthesis assistant.

Research question:
{question}

Below are research results collected from multiple sources.

{research_context}

Instructions:

1. Answer the research question directly.
2. Synthesize the sources rather than simply copying them.
3. Identify important agreements between sources.
4. Identify meaningful disagreements or limitations.
5. Do not invent facts that are not supported by the research.
6. Clearly distinguish evidence from inference.
7. Write a concise but informative answer.
"""

        try:
            summary = await self.llm.generate(prompt)

            return SynthesisResult(
                summary=summary,
                success=True,
            )

        except Exception as exc:
            return SynthesisResult(
                summary="",
                success=False,
                error=str(exc),
            )