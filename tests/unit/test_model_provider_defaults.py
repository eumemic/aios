from __future__ import annotations

import pytest
from pydantic import ValidationError

from aios.models.model_providers import (
    ModelProviderCreate,
    ProviderAuth,
    merge_litellm_defaults,
)


def test_defaults_only_anthropic_config_is_valid() -> None:
    body = ModelProviderCreate(
        provider="anthropic",
        litellm_defaults={"thinking": {"type": "adaptive", "display": "summarized"}},
    )
    assert body.api_key is None


def test_config_requires_credentials_or_defaults() -> None:
    with pytest.raises(ValidationError):
        ModelProviderCreate(provider="anthropic")


@pytest.mark.parametrize("key", ["api_key", "api_base", "base_url", "model"])
def test_defaults_reject_binding_keys(key: str) -> None:
    with pytest.raises(ValidationError):
        ModelProviderCreate(provider="anthropic", litellm_defaults={key: "bad"})


def test_explicit_reasoning_family_shadows_entire_default_family() -> None:
    defaults = {
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "high"},
        "temperature": 0.2,
    }
    merged = merge_litellm_defaults(defaults, {"reasoning_effort": "low"})
    assert merged == {"temperature": 0.2, "reasoning_effort": "low"}


def test_extra_body_merges_and_returns_owned_containers() -> None:
    defaults = {"extra_body": {"provider": {"order": ["anthropic"]}, "x": 1}}
    explicit = {"extra_body": {"x": 2}}
    merged = merge_litellm_defaults(defaults, explicit)
    assert merged["extra_body"] == {"provider": {"order": ["anthropic"]}, "x": 2}
    assert merged["extra_body"] is not defaults["extra_body"]


def test_resolved_defaults_are_applied_below_explicit_params() -> None:
    auth = ProviderAuth(
        api_key=None,
        api_base=None,
        owner_account_id="acc_root",
        litellm_defaults={"thinking": {"type": "adaptive", "display": "summarized"}},
    )
    merged = merge_litellm_defaults(auth.litellm_defaults, {"temperature": 0.3})
    assert merged["thinking"]["display"] == "summarized"
    assert merged["temperature"] == 0.3


def test_defaults_only_own_row_does_not_exempt_redirect_conflict_guard() -> None:
    """A key-less (defaults-only) own row must not unlock an api_base redirect.

    Without a key, LiteLLM falls back to the worker's env key, so the own-row
    exemption would reopen the exfiltration guard on a non-root account.
    """
    from aios.models.model_providers import provider_auth_conflict

    auth = ProviderAuth(
        api_key=None,
        api_base=None,
        owner_account_id="acc_child",
        litellm_defaults={"thinking": {"type": "adaptive"}},
    )
    extra = {"api_base": "https://attacker.example"}
    assert provider_auth_conflict(
        litellm_extra=extra, resolved=auth, account_id="acc_child", account_is_root=False
    )
    assert not provider_auth_conflict(
        litellm_extra=extra, resolved=auth, account_id="acc_child", account_is_root=True
    )
