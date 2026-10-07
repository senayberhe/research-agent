import json
from typing import Any

from openai import AsyncOpenAI
from openai.types.responses import Response

from app.core.config import settings
from app.llm.base import LLMProvider


class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        model: str | None = None,
    ):
        self.client = AsyncOpenAI(
            api_key=settings.openai_api_key,
        )
        # LLM_MODEL in .env unless a model is passed in.
        self.model = model or settings.llm_model

    # These use the Responses API (client.responses.create): input=, text=,
    # instructions= and response.output_text don't exist on
    # client.chat.completions.create.

    async def generate(
        self,
        prompt: str,
        instructions: str | None = None,
    ) -> str:
        response = await self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=prompt,
        )

        return self._output_text(response)

    async def generate_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        instructions: str | None = None,
        name: str = "research_plan",
    ) -> dict[str, Any]:
        response = await self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": name,
                    "strict": True,
                    "schema": schema,
                }
            },
        )

        # output_text is the JSON as a string; parse it into a dict.
        return json.loads(
            self._output_text(response)
        )

    def _output_text(self, response) -> str:
        # output_text is empty when the model returns no text (e.g. a
        # refusal). Raise so callers report a failure instead of saving an
        # empty result as a success.
        if not response.output_text:
            raise ValueError(
                f"{self.model} returned no text (status={response.status})"
            )

        return response.output_text


    async def generate_with_tools(
        self,
        input_items: list[Any],
        tools: list[dict[str, Any]],
        instructions: str | None = None,
    ) -> Response:
        # Returns the full Response: callers need response.output (function
        # calls to run) as well as response.output_text (the final answer).
        return await self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=input_items,
            tools=tools,
        )