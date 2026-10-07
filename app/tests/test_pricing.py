import pytest

from app.llm.pricing import (
    MODEL_PRICING,
    ModelPricing,
    estimate_cost_usd,
    get_model_pricing,
)


def test_exact_model_name():

    assert get_model_pricing("gpt-5.6-terra") == ModelPricing(
        input=2.00,
        cached_input=0.20,
        output=12.00,
    )


def test_dated_snapshot_matches_base_model():

    assert get_model_pricing("gpt-5.6-terra-2026-08-01") == MODEL_PRICING[
        "gpt-5.6-terra"
    ]


def test_longest_match_wins():

    # Starts with both "gpt-5.4" and "gpt-5.4-mini".
    assert get_model_pricing("gpt-5.4-mini-2026-03-17") == MODEL_PRICING[
        "gpt-5.4-mini"
    ]


def test_similar_names_do_not_match():

    # "gpt-5" must not price gpt-5.6-*, and "gpt-5.4" must not price
    # "gpt-5.40" (only a "-" may follow the base name).
    assert get_model_pricing("gpt-5.6-terra") != MODEL_PRICING["gpt-5"]
    assert get_model_pricing("gpt-5.40") is None


@pytest.mark.parametrize("model", [None, "", "gpt-5.6", "gpt-4o", "unknown"])
def test_unknown_models_have_no_pricing(model):

    assert get_model_pricing(model) is None
    assert estimate_cost_usd(model, 1_000, 1_000) is None


def test_cost_for_one_million_tokens_each():

    # gpt-5.6-terra: $2 input + $12 output.
    assert estimate_cost_usd("gpt-5.6-terra", 1_000_000, 1_000_000) == (
        pytest.approx(14.00)
    )


def test_cached_tokens_use_cached_price():

    # gpt-5.6-sol: 600K uncached * $4 + 400K cached * $0.40 + 0 output.
    cost = estimate_cost_usd(
        "gpt-5.6-sol",
        input_tokens=1_000_000,
        output_tokens=0,
        cached_input_tokens=400_000,
    )

    assert cost == pytest.approx(2.40 + 0.16)


def test_model_without_cache_discount_charges_full_input_price():

    # gpt-5.5-pro has no cached-input price.
    cost = estimate_cost_usd(
        "gpt-5.5-pro",
        input_tokens=1_000_000,
        output_tokens=0,
        cached_input_tokens=1_000_000,
    )

    assert cost == pytest.approx(30.00)


def test_cached_tokens_never_exceed_input_tokens():

    # A bad usage report can't make the cost go negative.
    cost = estimate_cost_usd(
        "gpt-5.4-mini",
        input_tokens=100,
        output_tokens=0,
        cached_input_tokens=500,
    )

    assert cost == pytest.approx(100 * 0.075 / 1_000_000)
