"""Outbound ``prompt_cache_key`` forwarding for the openai-provider path.

Production observation: an aios session that hits Anthropic shows ~90%
prompt-cache reads, while the same session against an OpenAI-compatible
endpoint (oai-proxy → auth2api → Codex/ChatGPT) sits near 7%. The
Anthropic path gets ``cache_control`` markers from
:func:`inject_cache_breakpoints`; the openai path was getting nothing.

OpenAI's Responses / Chat Completions APIs cache by the explicit
``prompt_cache_key`` field — callers pass a stable identifier and the
provider groups requests by that key for cache eligibility. The natural
scope in aios is the session id: every turn in a session shares a
prefix, and distinct sessions don't collide on cache lookups.

These tests pin the wire-level behavior: the openai path *must* send
``prompt_cache_key`` set to the session id, and the anthropic path
*must not* (cache_control markers already do the job there, and adding
the field would either be ignored or — worse — confuse a non-OpenAI
adapter).
"""

from __future__ import annotations

import pytest

from aios.harness import completion


class _DictResponse(dict[str, object]):
    """Subscriptable + ``.get`` + ``_hidden_params`` like a real LiteLLM response.

    The wrapper's ``_extract_cost`` reads ``_hidden_params``; a bare dict
    raises ``AttributeError`` on that lookup, which masks the kwarg we're
    actually trying to assert on.
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._hidden_params: dict[str, object] = {}


def _ok_response() -> _DictResponse:
    return _DictResponse(
        choices=[{"message": {"role": "assistant", "content": ""}}],
        usage={},
    )


@pytest.mark.asyncio
async def test_call_litellm_openai_path_sends_prompt_cache_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The non-streaming wrapper must forward a stable ``prompt_cache_key``
    on the openai provider path, keyed on ``session_id``.

    Without this, OpenAI's cache lookup can't group successive turns of
    the same session — producing the 7%-reads anomaly observed in prod.
    """
    captured: dict[str, object] = {}

    async def fake_acompletion(**kwargs: object) -> _DictResponse:
        captured.update(kwargs)
        return _ok_response()

    monkeypatch.setattr(completion.litellm, "acompletion", fake_acompletion)

    await completion.call_litellm(
        model="openai/gpt-5.5",
        messages=[{"role": "user", "content": "hi"}],
        session_id="sess_abc123",
    )

    assert captured.get("prompt_cache_key") == "sess_abc123"


@pytest.mark.asyncio
async def test_stream_litellm_openai_path_sends_prompt_cache_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streaming path must forward the same ``prompt_cache_key`` it would
    on the non-streaming path. The harness picks streaming vs. non-streaming
    based on whether a subscriber is attached — caching must not depend on
    that pick."""
    captured: dict[str, object] = {}

    class _EmptyResponse:
        def __aiter__(self) -> _EmptyResponse:
            return self

        async def __anext__(self) -> object:
            raise StopAsyncIteration

    async def fake_acompletion(**kwargs: object) -> _EmptyResponse:
        captured.update(kwargs)
        return _EmptyResponse()

    monkeypatch.setattr(completion.litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(
        completion.litellm,
        "stream_chunk_builder",
        lambda chunks: {
            "usage": {},
            "choices": [{"message": {"role": "assistant", "content": ""}}],
        },
    )

    from tests.unit.test_completion_timeouts import _StubPool

    await completion.stream_litellm(
        model="openai/gpt-5.5",
        messages=[{"role": "user", "content": "hi"}],
        pool=_StubPool(),  # type: ignore[arg-type]
        session_id="sess_xyz789",
    )

    assert captured.get("prompt_cache_key") == "sess_xyz789"


@pytest.mark.asyncio
async def test_call_litellm_anthropic_path_omits_prompt_cache_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anthropic-routed calls must NOT carry ``prompt_cache_key``.

    Cache control on that path is driven by content-block markers
    (:func:`inject_cache_breakpoints`). The OpenAI cache-key field has
    no defined behavior for the Anthropic adapter and could trip
    parameter validation in some LiteLLM versions; keep the channels
    cleanly separated."""
    captured: dict[str, object] = {}

    async def fake_acompletion(**kwargs: object) -> _DictResponse:
        captured.update(kwargs)
        return _ok_response()

    monkeypatch.setattr(completion.litellm, "acompletion", fake_acompletion)

    await completion.call_litellm(
        model="anthropic/claude-opus-4-6",
        messages=[{"role": "user", "content": "hi"}],
        session_id="sess_anthropic",
    )

    assert "prompt_cache_key" not in captured


@pytest.mark.asyncio
async def test_call_litellm_openai_path_session_id_optional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``session_id`` is optional — when omitted, no ``prompt_cache_key``
    is sent (rather than e.g. a synthetic one that would create a
    different cache bucket every call)."""
    captured: dict[str, object] = {}

    async def fake_acompletion(**kwargs: object) -> _DictResponse:
        captured.update(kwargs)
        return _ok_response()

    monkeypatch.setattr(completion.litellm, "acompletion", fake_acompletion)

    await completion.call_litellm(
        model="openai/gpt-5.5",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert "prompt_cache_key" not in captured


@pytest.mark.asyncio
async def test_call_litellm_extra_overrides_prompt_cache_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit ``prompt_cache_key`` in ``extra`` (e.g. agent-level
    override via ``litellm_extra``) wins over the harness default.

    Agents may want a custom scope — e.g. share a cache bucket across
    multiple sessions of the same conversation. The harness default is
    the safe fallback, not a ceiling."""
    captured: dict[str, object] = {}

    async def fake_acompletion(**kwargs: object) -> _DictResponse:
        captured.update(kwargs)
        return _ok_response()

    monkeypatch.setattr(completion.litellm, "acompletion", fake_acompletion)

    await completion.call_litellm(
        model="openai/gpt-5.5",
        messages=[{"role": "user", "content": "hi"}],
        session_id="sess_default",
        extra={"prompt_cache_key": "shared-bucket"},
    )

    assert captured.get("prompt_cache_key") == "shared-bucket"
