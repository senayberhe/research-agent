from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any


class FakeUsage:
    """Like response.usage from the Responses API."""

    def __init__(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
    ):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = (
            input_tokens + output_tokens
        )

        # Part of input_tokens served from the prompt cache.
        self.input_tokens_details = SimpleNamespace(
            cached_tokens=cached_tokens,
        )


class FakeResponse:
    """Like a Responses API response; always reports token usage."""

    def __init__(
        self,
        output,
        output_text="",
        input_tokens=100,
        output_tokens=50,
        cached_tokens=0,
        model=None,
    ):
        self.output = output
        self.output_text = output_text

        # The model snapshot that ran; None leaves the agent to use the
        # provider's model.
        self.model = model

        self.usage = FakeUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
        )


@dataclass
class FakeFunctionCall:
    type: str
    name: str
    arguments: str
    call_id: str


class FakeAgentLLM:
    """Searches once, then answers. Each call uses the FakeResponse default
    of 100 input + 50 output tokens, so a run uses 200 + 100 = 300."""

    def __init__(self, model: str = "gpt-5.4-mini"):
        # Like OpenAIProvider.model; used for pricing.
        self.model = model
        self.calls = 0
        self.requests = []

    async def generate_with_tools(
        self,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        instructions: str | None = None,
    ):

        self.calls += 1

        # Copy: the agent keeps appending to the same list.
        self.requests.append(list(input_items))

        if self.calls == 1:

            return FakeResponse(
                output=[
                    FakeFunctionCall(
                        type="function_call",
                        name="tavily_search",
                        arguments=(
                            '{"query": '
                            '"retrieval augmented generation"}'
                        ),
                        call_id="call_1",
                    )
                ],
            )

        return FakeResponse(
            output=[],
            output_text=(
                "RAG improves factuality by allowing "
                "language models to retrieve external evidence "
                "before generating an answer."
            ),
        )