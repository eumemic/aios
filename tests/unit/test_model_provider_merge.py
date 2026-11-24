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
        inline_credentials=True,
    )
    assert merged["api_key"] == "agent-key"
    assert merged["api_base"] == "https://agent.example"
    assert "extra_headers" not in merged


def test_model_is_forbidden_in_agent_extra() -> None:
    with pytest.raises(ValueError, match="model"):
        merge_provider_config(
            harness_kwargs={}, resolved=None, agent_extra={"model": "openai/other"}
        )


def test_account_only_strips_inline_credentials_and_endpoint() -> None:
    merged = merge_provider_config(
        harness_kwargs={},
        resolved=_auth(),
        agent_extra={"api_key": "agent-key", "base_url": "https://agent.example"},
        inline_credentials=False,
    )
    assert merged["api_key"] == "row-key"
    assert merged["api_base"] == "https://row.example"
    assert "base_url" not in merged
    assert merged["extra_headers"] == {"x-row": "yes"}


def test_defaults_apply_when_agent_sets_no_reasoning_knob() -> None:
    merged = merge_provider_config(
        harness_kwargs={"temperature": 0.1},
        resolved=_auth(),
        agent_extra={"extra_body": {"provider": {"order": ["x"]}}},
        inline_credentials=True,
    )
    assert merged["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert merged["temperature"] == 0.2  # account defaults beat harness kwargs
    # extra_body unions per key; agent key wins whole-key (no deep merge)
    assert merged["extra_body"] == {"provider": {"order": ["x"]}, "row": True}


@pytest.mark.parametrize("family_key", ["reasoning_effort", "thinking", "output_config"])
def test_any_reasoning_family_key_shadows_whole_family(family_key: str) -> None:
    auth = ProviderAuth(
        api_key="k",
        api_base=None,
        owner_account_id="acc",
        litellm_defaults={
            "reasoning_effort": "low",
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "low"},
        },
    )
    merged = merge_provider_config(
        harness_kwargs={}, resolved=auth, agent_extra={family_key: {"x": 1}}
    )
    assert merged[family_key] == {"x": 1}
    assert (set(merged) & {"reasoning_effort", "thinking", "output_config"}) == {family_key}


def test_no_row_passes_agent_extra_through_as_owned_copy() -> None:
    agent = {"extra_body": {"a": {"b": 1}}}
    merged = merge_provider_config(
        harness_kwargs={}, resolved=None, agent_extra=agent, inline_credentials=True
    )
    merged["extra_body"]["a"]["b"] = 2
    assert agent["extra_body"]["a"]["b"] == 1


def test_defaults_only_row_has_no_credentials() -> None:
    auth = ProviderAuth(
        api_key="", api_base=None, owner_account_id="acc", litellm_defaults={"seed": 1}
    )
    assert auth.has_credentials is False
    merged = merge_provider_config(
        harness_kwargs={}, resolved=auth, agent_extra=None, inline_credentials=False
    )
    assert merged == {"seed": 1}
