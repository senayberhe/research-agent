import logging

from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelPricing:
    """USD per 1M tokens."""

    input: float
    output: float
    # Price for input tokens served from OpenAI's prompt cache; None when the
    # model has no cached-input discount (they cost the normal input price).
    cached_input: float | None = None


# OpenAI Standard tier, short context (<= 272K tokens), USD per 1M tokens.
# Source: https://developers.openai.com/api/docs/pricing (checked 2026-10-05).
# Not covered: Batch / Flex / Priority tiers, long-context (> 272K) prices,
# and the 10% uplift for regional (data residency) endpoints.
MODEL_PRICING: dict[str, ModelPricing] = {
    "gpt-5.6-sol": ModelPricing(input=4.00, cached_input=0.40, output=20.00),
    "gpt-5.6-terra": ModelPricing(input=2.00, cached_input=0.20, output=12.00),
    "gpt-5.6-luna": ModelPricing(input=0.20, cached_input=0.02, output=1.20),
    "gpt-5.5": ModelPricing(input=5.00, cached_input=0.50, output=30.00),
    "gpt-5.5-pro": ModelPricing(input=30.00, output=180.00),
    "gpt-5.4": ModelPricing(input=2.50, cached_input=0.25, output=15.00),
    "gpt-5.4-mini": ModelPricing(input=0.75, cached_input=0.075, output=4.50),
    "gpt-5.4-nano": ModelPricing(input=0.20, cached_input=0.02, output=1.25),
    "gpt-5.4-pro": ModelPricing(input=30.00, output=180.00),
    "gpt-5.2": ModelPricing(input=1.75, cached_input=0.175, output=14.00),
    "gpt-5.2-pro": ModelPricing(input=21.00, output=168.00),
    "gpt-5.1": ModelPricing(input=1.25, cached_input=0.125, output=10.00),
    "gpt-5": ModelPricing(input=1.25, cached_input=0.125, output=10.00),
    "gpt-5-mini": ModelPricing(input=0.25, cached_input=0.025, output=2.00),
    "gpt-5-nano": ModelPricing(input=0.05, cached_input=0.005, output=0.40),
    "gpt-5-pro": ModelPricing(input=15.00, output=120.00),
}


def get_model_pricing(model: str | None) -> ModelPricing | None:
    """Pricing for a model name, or None if it isn't in MODEL_PRICING.

    Responses report dated snapshots (e.g. "gpt-5.4-mini-2026-03-17"), so a
    name also matches a table entry it starts with followed by "-". The
    longest match wins, so "gpt-5.4-mini-..." uses gpt-5.4-mini, not gpt-5.4.
    """

    if not model:
        return None

    matches = [
        name
        for name in MODEL_PRICING
        if model == name or model.startswith(name + "-")
    ]

    if not matches:
        return None

    return MODEL_PRICING[max(matches, key=len)]


def estimate_cost_usd(
    model: str | None,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> float | None:
    """Estimated USD cost of one LLM call, or None for an unknown model.

    input_tokens includes cached_input_tokens (as in the Responses API), so
    cached tokens are charged at the cached price and the rest at the normal
    input price.
    """

    pricing = get_model_pricing(model)

    if pricing is None:
        logger.warning("No pricing for model %r; cost is unknown", model)
        return None

    cached_input_tokens = min(cached_input_tokens, input_tokens)
    uncached_input_tokens = input_tokens - cached_input_tokens

    cached_price = (
        pricing.cached_input
        if pricing.cached_input is not None
        else pricing.input
    )

    return (
        uncached_input_tokens * pricing.input
        + cached_input_tokens * cached_price
        + output_tokens * pricing.output
    ) / 1_000_000
