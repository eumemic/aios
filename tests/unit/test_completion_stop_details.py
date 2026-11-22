"""Anthropic refusal ``stop_details`` survive LiteLLM (#2092).

A refusal from Anthropic carries ``stop_details`` (``{"type": "refusal",
"category", "explanation"}``); LiteLLM 1.96's Anthropic adapter drops it on both
the streaming and non-streaming paths. ``aios.harness.completion`` patches the
adapter to keep it in ``provider_specific_fields``. These tests drive LiteLLM's
real Anthropic handler over a fake HTTP client, so an upgrade that moves the
patched internals fails here instead of silently losing the field — and pin
that ``stream_litellm`` / ``call_litellm`` surface it on ``LlmResponse``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import litellm
import pytest

from aios.harness import completion
from tests.unit.test_completion_timeouts import _StubPool

STOP_DETAILS = {
    "type": "refusal",
    "category": "reasoning_extraction",
    "explanation": "This request was declined.",
}
_REQUEST = httpx.Request("POST", "https://anthropic.test/v1/messages")


def _sse(event: dict[str, Any]) -> list[str]:
    return [f"event: {event['type']}", f"data: {json.dumps(event)}", ""]


_STREAM_EVENTS: list[dict[str, Any]] = [
    {
        "type": "message_start",
        "message": {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 1},
        },
    },
    {
        "type": "message_delta",
        "delta": {"stop_reason": "refusal", "stop_sequence": None, "stop_details": STOP_DETAILS},
        "usage": {"output_tokens": 1},
    },
    {"type": "message_stop"},
]

_COMPLETION: dict[str, Any] = {
    "id": "msg_2",
    "type": "message",
    "role": "assistant",
    "model": "claude-opus-5-5",
    "content": [],
    "stop_reason": "refusal",
    "stop_sequence": None,
    "stop_details": STOP_DETAILS,
    "usage": {"input_tokens": 10, "output_tokens": 0},
}


class _StreamingResponse:
    status_code = 200
    headers = httpx.Headers({"content-type": "text/event-stream"})
    request = _REQUEST

    async def aiter_lines(self) -> AsyncIterator[str]:
        for event in _STREAM_EVENTS:
            for line in _sse(event):
                yield line


class _FakeClient:
    """Stands in for LiteLLM's ``AsyncHTTPHandler``: the handler's only call is ``post``."""

    def __init__(self, *, stream: bool) -> None:
        self._stream = stream

    async def post(self, url: str, *args: Any, **kwargs: Any) -> Any:
        if self._stream:
            return _StreamingResponse()
        return httpx.Response(200, json=_COMPLETION, request=_REQUEST)


_CALL = {
    "model": "anthropic/claude-opus-5-5",
    "messages": [{"role": "user", "content": "hi"}],
    "api_key": "sk-test",
    "api_base": "https://anthropic.test",
    "max_tokens": 16,
}


def _use_client(monkeypatch: pytest.MonkeyPatch, client: _FakeClient) -> None:
    """Route LiteLLM's Anthropic handler to ``client``: the streaming path falls
    back to ``litellm.module_level_aclient``, the non-streaming one asks
    ``get_async_httpx_client``."""
    from litellm.llms.anthropic.chat import handler

    monkeypatch.setattr(litellm, "module_level_aclient", client)
    monkeypatch.setattr(handler, "get_async_httpx_client", lambda *a, **k: client)


@pytest.mark.asyncio
async def test_streaming_message_delta_keeps_stop_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_client(monkeypatch, _FakeClient(stream=True))
    stream = await litellm.acompletion(**_CALL, stream=True)
    seen: list[Any] = []
    async for chunk in stream:
        if chunk.choices:
            seen.append(getattr(chunk.choices[0].delta, "provider_specific_fields", None))
    assert any(
        isinstance(fields, dict) and fields.get("stop_details") == STOP_DETAILS for fields in seen
    ), seen


@pytest.mark.asyncio
async def test_non_streaming_response_keeps_stop_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_client(monkeypatch, _FakeClient(stream=False))
    response = await litellm.acompletion(**_CALL)
    assert response.choices[0].finish_reason == "content_filter"
    fields = response.choices[0].message.provider_specific_fields
    assert fields["stop_details"] == STOP_DETAILS


class _Delta:
    def __init__(self, provider_specific_fields: dict[str, Any] | None) -> None:
        self.content = None
        self.provider_specific_fields = provider_specific_fields


class _Choice:
    def __init__(self, finish_reason: str | None, fields: dict[str, Any] | None) -> None:
        self.finish_reason = finish_reason
        self.delta = _Delta(fields)


class _Chunk:
    def __init__(self, finish_reason: str | None, fields: dict[str, Any] | None = None) -> None:
        self.choices = [_Choice(finish_reason, fields)]


class _FakeStream:
    def __init__(self, chunks: list[_Chunk]) -> None:
        self._it = iter(chunks)

    def __aiter__(self) -> _FakeStream:
        return self

    async def __anext__(self) -> _Chunk:
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration from None


def _assembled(chunks: list[Any], **_kwargs: Any) -> dict[str, Any]:
    return {
        "usage": {},
        "choices": [
            {"message": {"role": "assistant", "content": ""}, "finish_reason": "content_filter"}
        ],
    }


@pytest.mark.asyncio
async def test_stream_litellm_surfaces_stop_details(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = [_Chunk(None), _Chunk("content_filter", {"stop_details": STOP_DETAILS}), _Chunk("")]

    async def fake_acompletion(**_kwargs: object) -> _FakeStream:
        return _FakeStream(chunks)

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(litellm, "stream_chunk_builder", _assembled)
    response = await completion.stream_litellm(
        completion.LlmRequest(messages=[{"role": "user", "content": "hi"}], session_id="sess_sd"),
        model="anthropic/claude-opus-5-5",
        pool=_StubPool(),
    )
    assert response.finish_reason == "content_filter"
    assert response.stop_details == STOP_DETAILS


@pytest.mark.asyncio
async def test_call_litellm_surfaces_stop_details(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_acompletion(**_kwargs: object) -> dict[str, Any]:
        return {
            "usage": {},
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "provider_specific_fields": {"stop_details": STOP_DETAILS},
                    },
                    "finish_reason": "content_filter",
                }
            ],
        }

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    response = await completion.call_litellm(
        completion.LlmRequest(messages=[{"role": "user", "content": "hi"}]),
        model="anthropic/claude-opus-5-5",
    )
    assert response.stop_details == STOP_DETAILS


@pytest.mark.asyncio
async def test_ordinary_responses_carry_none(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_acompletion(**_kwargs: object) -> dict[str, Any]:
        return {
            "usage": {},
            "choices": [
                {"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
            ],
        }

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    response = await completion.call_litellm(
        completion.LlmRequest(messages=[{"role": "user", "content": "hi"}]),
        model="anthropic/claude-opus-5-5",
    )
    assert response.stop_details is None
