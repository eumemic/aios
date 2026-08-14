from __future__ import annotations

import litellm
import pytest
from litellm.exceptions import UnsupportedParamsError

from aios.harness import completion
from aios.services.litellm_params import unsupported_openai_params


def test_stale_capability_map_is_reported_at_config_save() -> None:
    assert unsupported_openai_params("xai/grok-4.6", {"reasoning_effort": "high"}) == [
        "reasoning_effort"
    ]


def test_provider_specific_kwargs_are_not_mislabeled() -> None:
    assert unsupported_openai_params("xai/grok-4.6", {"vendor_bogus": True}) == []


def test_drop_params_cannot_silently_discard_unsupported_sampling_param() -> None:
    kwargs = completion._build_litellm_kwargs(
        model="anthropic/claude-opus-4-8",
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        auth=None,
        extra={"temperature": 0.8, "drop_params": True},
        session_id=None,
        stream=False,
    )

    assert kwargs["drop_params"] is False
    with pytest.raises(UnsupportedParamsError, match="does not support temperature"):
        litellm.get_optional_params(
            model="claude-opus-4-8",
            custom_llm_provider="anthropic",
            temperature=kwargs["temperature"],
            drop_params=kwargs["drop_params"],
        )
