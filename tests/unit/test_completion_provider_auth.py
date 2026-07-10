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


async def test_extra_api_base_overrides_auth_api_base(monkeypatch: pytest.MonkeyPatch) -> None:
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

    assert captured["api_base"] == "https://agent.example"
    assert (
        captured["api_key"] == "sk-resolved"
    )  # extra doesn't carry a key here — auth's still lands


async def test_extra_base_url_alias_suppresses_auth_api_base(
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

    assert "api_base" not in captured  # auth's api_base was suppressed
    assert captured["base_url"] == "https://agent.example"  # extra's redirect lands unmodified


async def test_extra_api_key_overrides_auth_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
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

    assert captured["api_key"] == "sk-agent-supplied"


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
