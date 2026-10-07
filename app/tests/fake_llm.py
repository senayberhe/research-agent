from app.llm.base import LLMProvider


class FakeLLMProvider(LLMProvider):
    """Returns a fixed summary and records the prompts it was given."""

    RESPONSE = "This is a fake synthesized research summary."

    def __init__(self):
        self.prompts: list[str] = []

    async def generate(self, prompt: str, instructions: str | None = None) -> str:
        self.prompts.append(prompt)
        return self.RESPONSE


class FailingLLMProvider(LLMProvider):
    """Simulates the LLM API failing."""

    async def generate(self, prompt: str, instructions: str | None = None) -> str:
        raise RuntimeError("LLM service unavailable")
