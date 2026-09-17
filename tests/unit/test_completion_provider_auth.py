"""Injection of the resolved per-account ``ProviderAuth`` into litellm kwargs.

Pins the wire-level behavior of ``_build_litellm_kwargs`` and its two callers
(``call_litellm``/``stream_litellm``): ``auth`` (when present) supplies
``api_key``/``api_base``, but a per-agent ``litellm_extra`` redirect still
wins — the account row only fills in a default underneath. Also pins the
dual-key-kwargs guard: an extra redirect via the ``base_url`` alias must
suppress the ``auth.api_base`` injection, not just the ``api_base`` key,
or litellm's api_base-over-base_url precedence would silently invert "extra
wins".
"""

from __future__ import annotations

import litellm
import pytest

from aios.harness import completion
from aios.models.model_providers import ProviderAuth


class _DictResponse(dict[str, object]):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._hidden_params: dict[str, object] = {}


@pytest.fixture(autouse=True)
def _nonlegacy_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Settings:
        inference_credential_policy = "account_only"
        model_call_deadline_s = 300.0

    monkeypatch.setattr(completion, "get_settings", lambda: _Settings())


def _ok_response() -> _DictResponse:
    return _DictResponse(
        choices=[{"message": {"role": "assistant", "content": ""}}],
        usage={},
    )


def _capture(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    captured: dict[str, object] = {}

    async def fake_acompletion(**kwargs: object) -> _DictResponse:
        captured.update(kwargs)
        return _ok_response()

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    return captured


async def test_auth_injects_api_key_and_api_base(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture(monkeypatch)
    auth = ProviderAuth(
        api_key="sk-resolved", api_base="https://proxy.example", owner_account_id="acc_x"
    )

    await completion.call_litellm(
        completion.LlmRequest(messages=[{"role": "user", "content": "hi"}]),
        model="anthropic/claude-x",
        auth=auth,
    )

    assert captured["api_key"] == "sk-resolved"
    assert captured["api_base"] == "https://proxy.example"


async def test_auth_none_injects_neither_key(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture(monkeypatch)

    await completion.call_litellm(
        completion.LlmRequest(messages=[{"role": "user", "content": "hi"}]),
        model="anthropic/claude-x",
    )

    assert "api_key" not in captured
    assert "api_base" not in captured


async def test_resolved_api_base_overrides_extra_api_base(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture(monkeypatch)
    auth = ProviderAuth(
        api_key="sk-resolved", api_base="https://account.example", owner_account_id="acc_x"
    )

    await completion.call_litellm(
        completion.LlmRequest(
            messages=[{"role": "user", "content": "hi"}],
            params={"api_base": "https://agent.example"},
        ),
        model="anthropic/claude-x",
        auth=auth,
    )

    assert captured["api_base"] == "https://account.example"
    assert (
        captured["api_key"] == "sk-resolved"
    )  # extra doesn't carry a key here — auth's still lands


async def test_resolved_api_base_suppresses_extra_base_url_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A redirect via the base_url alias must ALSO suppress auth.api_base —
    not just the literal `api_base` key — or both `api_base` (from auth) and
    `base_url` (from extra) would land in kwargs simultaneously, and litellm's
    api_base-over-base_url precedence would silently invert "extra wins".
    """
    captured = _capture(monkeypatch)
    auth = ProviderAuth(
        api_key="sk-resolved", api_base="https://account.example", owner_account_id="acc_x"
    )

    await completion.call_litellm(
        completion.LlmRequest(
            messages=[{"role": "user", "content": "hi"}],
            params={"base_url": "https://agent.example"},
        ),
        model="anthropic/claude-x",
        auth=auth,
    )

    assert captured["api_base"] == "https://account.example"
    assert "base_url" not in captured


async def test_resolved_api_key_overrides_extra_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture(monkeypatch)
    auth = ProviderAuth(api_key="sk-resolved", api_base=None, owner_account_id="acc_x")

    await completion.call_litellm(
        completion.LlmRequest(
            messages=[{"role": "user", "content": "hi"}],
            params={"api_key": "sk-agent-supplied"},
        ),
        model="anthropic/claude-x",
        auth=auth,
    )

    assert captured["api_key"] == "sk-resolved"


async def test_cache_hints_unaffected_by_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture(monkeypatch)
    auth = ProviderAuth(api_key="sk-resolved", api_base=None, owner_account_id="acc_x")

    await completion.call_litellm(
        completion.LlmRequest(
            messages=[{"role": "user", "content": "hi"}], session_id="sess_abc123"
        ),
        model="openai/gpt-5.5",
        auth=auth,
    )

    extra_body = captured.get("extra_body")
    assert isinstance(extra_body, dict)
    assert extra_body.get("prompt_cache_key") == "sess_abc123"


async def test_stream_litellm_also_injects_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _EmptyResponse:
        def __aiter__(self) -> _EmptyResponse:
            return self

        async def __anext__(self) -> object:
            raise StopAsyncIteration

    async def fake_acompletion(**kwargs: object) -> _EmptyResponse:
        captured.update(kwargs)
        return _EmptyResponse()

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(
        litellm,
        "stream_chunk_builder",
        lambda chunks, **_kwargs: {
            "usage": {},
            "choices": [{"message": {"role": "assistant", "content": ""}}],
        },
    )

    from tests.unit.test_completion_timeouts import _StubPool

    auth = ProviderAuth(
        api_key="sk-resolved", api_base="https://proxy.example", owner_account_id="acc_x"
    )
    await completion.stream_litellm(
        completion.LlmRequest(
            messages=[{"role": "user", "content": "hi"}], session_id="sess_xyz789"
        ),
        model="anthropic/claude-x",
        pool=_StubPool(),
        auth=auth,
    )

    assert captured["api_key"] == "sk-resolved"
    assert captured["api_base"] == "https://proxy.example"


async def test_legacy_env_preserves_inline_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    class _LegacySettings:
        inference_credential_policy = "legacy_env"
        model_call_deadline_s = 300.0

    monkeypatch.setattr(completion, "get_settings", lambda: _LegacySettings())
    captured = _capture(monkeypatch)
    auth = ProviderAuth(api_key="sk-row", api_base="https://row.example", owner_account_id="acc_x")
    await completion.call_litellm(
        completion.LlmRequest(
            messages=[{"role": "user", "content": "hi"}],
            params={"api_key": "sk-inline", "api_base": "https://inline.example"},
        ),
        model="anthropic/claude-x",
        auth=auth,
    )
    assert captured["api_key"] == "sk-inline"
    assert captured["api_base"] == "https://inline.example"


async def test_resolved_api_base_suppresses_extra_url_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A redirect via the watsonx-text ``url`` alias must suppress the injection of
    ``auth.api_base`` (the account row's endpoint) just as a ``base_url`` redirect
    does — or both the row's ``api_base`` (top-level kwarg, which LiteLLM resolves
    first) and the caller's ``url`` (in optional_params) would land in kwargs
    simultaneously and LiteLLM would route to the row's endpoint while the guard
    believed the caller's. With ``url`` stripped (next test), the row's endpoint
    wins outright, as the account-only policy intends; this test pins the
    companion invariant that no stale ``url`` survives to compete with it."""
    captured = _capture(monkeypatch)
    auth = ProviderAuth(
        api_key="sk-resolved", api_base="https://account.example", owner_account_id="acc_x"
    )

    await completion.call_litellm(
        completion.LlmRequest(
            messages=[{"role": "user", "content": "hi"}],
            params={"url": "https://agent.example"},
        ),
        model="watsonx_text/ibm/granite-13b-instruct-v2",
        auth=auth,
    )

    # ``url`` is stripped under the account-only policy (it's a redirect alias, not a
    # legitimate passthrough), so the account row's endpoint is the sole authority.
    assert "url" not in captured
    assert captured["api_key"] == "sk-resolved"
    # The row's api_base is injected because, after stripping ``url``,
    # ``api_base_of(effective_extra)`` is None (no caller redirect remains).
    assert captured.get("api_base") == "https://account.example"


async def test_url_redirect_stripped_when_row_api_base_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defense-in-depth for the reported exploit's completion.py arm: an ancestor row
    with ``api_base=None`` (endpoint sourced from env/defaults) plus a caller-supplied
    ``url`` redirect. The conflict guard (guard 3) catches this before completion.py
    on the production path; this test pins that completion.py ALSO strips the ``url``
    so, even if a guard missed it, the caller's endpoint could not reach LiteLLM —
    the call falls back to env/defaults rather than routing to the attacker."""
    captured = _capture(monkeypatch)
    auth = ProviderAuth(api_key="sk-ancestor", api_base=None, owner_account_id="acc_parent")

    await completion.call_litellm(
        completion.LlmRequest(
            messages=[{"role": "user", "content": "hi"}],
            params={"url": "https://attacker.example/wx"},
        ),
        model="watsonx_text/ibm/granite-13b-instruct-v2",
        auth=auth,
    )

    assert "url" not in captured
    assert captured["api_key"] == "sk-ancestor"
    assert "api_base" not in captured  # row's api_base is None → nothing injected


async def test_wx_credentials_blob_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    """The nested ``wx_credentials`` blob (which LiteLLM reads for a nested ``url``
    redirect AND an ``apikey`` override) is stripped wholesale under non-legacy
    policy — an inline credentials blob is agent metadata, not account config."""
    captured = _capture(monkeypatch)
    auth = ProviderAuth(
        api_key="sk-resolved", api_base="https://account.example", owner_account_id="acc_x"
    )

    await completion.call_litellm(
        completion.LlmRequest(
            messages=[{"role": "user", "content": "hi"}],
            params={"wx_credentials": {"url": "https://agent.example", "apikey": "sk-inline"}},
        ),
        model="watsonx_text/ibm/granite-13b-instruct-v2",
        auth=auth,
    )

    assert "wx_credentials" not in captured
    assert "url" not in captured
    assert captured["api_key"] == "sk-resolved"  # inline apikey override stripped
    assert captured.get("api_base") == "https://account.example"


async def test_watsonx_credentials_blob_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture(monkeypatch)
    auth = ProviderAuth(
        api_key="sk-resolved", api_base="https://account.example", owner_account_id="acc_x"
    )

    await completion.call_litellm(
        completion.LlmRequest(
            messages=[{"role": "user", "content": "hi"}],
            params={"watsonx_credentials": {"url": "https://agent.example"}},
        ),
        model="watsonx_text/ibm/granite-13b-instruct-v2",
        auth=auth,
    )

    assert "watsonx_credentials" not in captured
    assert "url" not in captured
    assert captured.get("api_base") == "https://account.example"


async def test_legacy_env_preserves_watsonx_url_inline_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under the legacy_env policy, inline auth fields are preserved exactly as for
    ``api_base`` — the account-only strip is what removes caller redirect/key
    overrides, and legacy_env deliberately does not apply it."""

    class _LegacySettings:
        inference_credential_policy = "legacy_env"
        model_call_deadline_s = 300.0

    monkeypatch.setattr(completion, "get_settings", lambda: _LegacySettings())
    captured = _capture(monkeypatch)
    auth = ProviderAuth(api_key="sk-row", api_base="https://row.example", owner_account_id="acc_x")
    await completion.call_litellm(
        completion.LlmRequest(
            messages=[{"role": "user", "content": "hi"}],
            params={"api_key": "sk-inline", "url": "https://inline.example"},
        ),
        model="watsonx_text/ibm/granite-13b-instruct-v2",
        auth=auth,
    )
    assert captured["api_key"] == "sk-inline"
    assert captured["url"] == "https://inline.example"


def test_stale_model_map_params_are_centrally_allowed() -> None:
    kwargs = completion._build_litellm_kwargs(
        model="xai/grok-4.6",
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        auth=None,
        extra={"reasoning_effort": "high"},
        session_id=None,
        stream=False,
    )

    assert "reasoning_effort" in kwargs["allowed_openai_params"]


def test_explicit_allowed_params_are_merged_with_central_passthrough() -> None:
    kwargs = completion._build_litellm_kwargs(
        model="xai/grok-4.6",
        messages=[],
        tools=None,
        auth=None,
        extra={"reasoning_effort": "high", "allowed_openai_params": ["seed"]},
        session_id=None,
        stream=False,
    )

    assert kwargs["allowed_openai_params"] == ["reasoning_effort", "seed"]
