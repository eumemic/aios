from __future__ import annotations

from aios.services.litellm_params import unsupported_openai_params


def test_stale_capability_map_is_reported_at_config_save() -> None:
    assert unsupported_openai_params("xai/grok-4.6", {"reasoning_effort": "high"}) == [
        "reasoning_effort"
    ]


def test_provider_specific_kwargs_are_not_mislabeled() -> None:
    assert unsupported_openai_params("xai/grok-4.6", {"vendor_bogus": True}) == []
