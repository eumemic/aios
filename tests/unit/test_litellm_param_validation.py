from __future__ import annotations

from typing import Any

import litellm

from aios.services.litellm_params import unsupported_openai_params


def test_stale_capability_map_is_reported_at_config_save(monkeypatch: Any) -> None:
    # Model metadata may refresh independently of the locked LiteLLM package. Exercise a
    # deliberately stale snapshot instead of depending on today's remote capability map.
    monkeypatch.setattr(
        litellm,
        "get_supported_openai_params",
        lambda _model: ["temperature"],
    )

    assert unsupported_openai_params("xai/grok-4.6", {"reasoning_effort": "high"}) == [
        "reasoning_effort"
    ]


def test_provider_specific_kwargs_are_not_mislabeled() -> None:
    assert unsupported_openai_params("xai/grok-4.6", {"vendor_bogus": True}) == []
