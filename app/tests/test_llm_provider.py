from types import SimpleNamespace

import pytest

from app.agents.planner_prompt import SYSTEM_PROMPT, build_planning_prompt
from app.llm.provider import OpenAIProvider


class FakeResponses:
    """Stands in for client.responses; records each create() call."""

    def __init__(self, output_text: str):
        self.output_text = output_text
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            output_text=self.output_text,
            status="completed",
        )


def make_provider(output_text: str) -> tuple[OpenAIProvider, FakeResponses]:
    provider = OpenAIProvider(model="test-model")
    responses = FakeResponses(output_text)
    provider.client = SimpleNamespace(responses=responses)
    return provider, responses


@pytest.mark.asyncio
async def test_generate_uses_responses_api():
    provider, responses = make_provider("A summary.")

    text = await provider.generate(
        "Summarize this.",
        instructions="Be concise.",
    )

    assert text == "A summary."
    assert responses.calls == [
        {
            "model": "test-model",
            "instructions": "Be concise.",
            "input": "Summarize this.",
        }
    ]


@pytest.mark.asyncio
async def test_generate_structured_parses_json():
    provider, responses = make_provider(
        '{"steps": [{"tool": "arxiv", "query": "retrieval augmented generation"}]}'
    )

    schema = {
        "type": "object",
        "properties": {"steps": {"type": "array"}},
        "required": ["steps"],
        "additionalProperties": False,
    }

    plan = await provider.generate_structured(
        build_planning_prompt("What is RAG?"),
        schema=schema,
        instructions=SYSTEM_PROMPT,
    )

    assert plan == {
        "steps": [{"tool": "arxiv", "query": "retrieval augmented generation"}]
    }

    call = responses.calls[0]
    assert call["instructions"] == SYSTEM_PROMPT
    assert "What is RAG?" in call["input"]
    assert call["text"]["format"] == {
        "type": "json_schema",
        "name": "research_plan",
        "strict": True,
        "schema": schema,
    }


@pytest.mark.asyncio
async def test_empty_output_raises():
    provider, _ = make_provider("")

    with pytest.raises(ValueError, match="returned no text"):
        await provider.generate("Summarize this.")


def test_provider_uses_configured_model_by_default():

    from app.core.config import settings

    assert OpenAIProvider().model == settings.llm_model
    assert OpenAIProvider(model="other-model").model == "other-model"


def test_configured_model_has_pricing():

    # A typo or unpriced model would record every run's cost as unknown.
    from app.core.config import settings
    from app.llm.pricing import get_model_pricing

    assert get_model_pricing(settings.llm_model) is not None, (
        f"LLM_MODEL={settings.llm_model!r} is not in app/llm/pricing.py"
    )
