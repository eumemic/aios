from __future__ import annotations

import pytest

from aios.models.model_providers import ProviderAuth
from aios.services.model_providers import merge_provider_config


def _auth() -> ProviderAuth:
    return ProviderAuth(
        api_key="row-key",
        api_base="https://row.example",
        owner_account_id="acc_parent",
        credentials={"api_key": "row-key", "extra_headers": {"x-row": "yes"}},
        litellm_defaults={
            "temperature": 0.2,
            "thinking": {"type": "adaptive", "display": "summarized"},
            "extra_body": {"provider": {"order": ["anthropic"]}, "row": True},
        },
    )


def test_merge_precedence_reasoning_shadow_and_extra_body_union() -> None:
    merged = merge_provider_config(
        harness_kwargs={"temperature": 0.1, "max_tokens": 100},
        resolved=_auth(),
        agent_extra={
            "temperature": 0.9,
            "reasoning_effort": "high",
            "extra_body": {"agent": True},
        },
    )
    assert merged["temperature"] == 0.9
    assert merged["max_tokens"] == 100
    assert merged["reasoning_effort"] == "high"
    assert "thinking" not in merged
    assert merged["api_key"] == "row-key"
    assert merged["extra_body"] == {
        "provider": {"order": ["anthropic"]},
        "row": True,
        "agent": True,
    }

    # Every returned container is owned; later cache-hint mutation cannot alter rows/agents.
    merged["extra_body"]["row"] = False
    assert _auth().litellm_defaults["extra_body"]["row"] is True


def test_agent_credentials_suppress_entire_row_credential_source() -> None:
    merged = merge_provider_config(
        harness_kwargs={},
        resolved=_auth(),
        agent_extra={"api_key": "agent-key", "api_base": "https://agent.example"},
    )
    assert merged["api_key"] == "agent-key"
    assert merged["api_base"] == "https://agent.example"
    assert "extra_headers" not in merged


def test_model_is_forbidden_in_agent_extra() -> None:
    with pytest.raises(ValueError, match="model"):
        merge_provider_config(
            harness_kwargs={}, resolved=None, agent_extra={"model": "openai/other"}
        )
