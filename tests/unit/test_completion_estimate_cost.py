from __future__ import annotations

from typing import Any
from unittest.mock import patch

from aios.harness.completion import estimate_cost_usd


def test_estimate_cost_usd_builds_cache_aware_usage_object() -> None:
    captured: dict[str, Any] = {}

    def fake_cost_per_token(*, model: str, usage_object: object) -> tuple[float, float]:
        captured["model"] = model
        captured["usage"] = usage_object
        return 0.001, 0.002

    with patch("litellm.cost_per_token", fake_cost_per_token):
        cost = estimate_cost_usd(
            "mapped-model",
            {
                "input_tokens": 100,
                "output_tokens": 20,
                "cache_read_input_tokens": 30,
                "cache_creation_input_tokens": 40,
            },
        )

    assert cost == 0.003
    assert captured["model"] == "mapped-model"
    usage = captured["usage"]
    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 20
    assert usage.prompt_tokens_details.cached_tokens == 30
    assert usage.cache_creation_input_tokens == 40


def test_estimate_cost_usd_returns_none_for_unmapped_model() -> None:
    with patch("litellm.cost_per_token", side_effect=Exception("unknown")):
        assert estimate_cost_usd("not-in-map", {"input_tokens": 1, "output_tokens": 2}) is None
