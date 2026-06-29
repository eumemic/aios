"""Unit tests for the internal ``LlmRequest`` / ``LlmResponse`` types.

Foundational types for the Workflows-as-Models epic (issue #1632): the named,
payload-shaped shapes reused by the session model-call path and the future
``call_llm()`` builtin — deliberately *not* OpenAI wire format.

These tests pin:

* the type shapes (fields + defaults),
* that ``call_litellm`` consumes an ``LlmRequest`` and produces an
  ``LlmResponse`` projecting ``content``/``tool_calls`` while retaining the
  opaque provider message,
* that ``request.params`` round-trips ``litellm_extra`` verbatim into the
  litellm call kwargs (the acceptance criterion), and
* that the refactor preserves behavior on the openai cache-key path.
"""

from __future__ import annotations

from typing import Any

import litellm
import pytest

from aios.harness import completion
from aios.harness.completion import LlmRequest, LlmResponse


class _DictResponse(dict):  # type: ignore[type-arg]
    """A dict-shaped litellm envelope (matches the e2e harness mock)."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._hidden_params: dict[str, object] = {}


def _envelope(message: dict[str, Any], finish_reason: str | None = "stop") -> _DictResponse:
    return _DictResponse(
        choices=[{"message": message, "finish_reason": finish_reason}],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )


class TestLlmRequestShape:
    def test_minimal_request_defaults(self) -> None:
        req = LlmRequest(messages=[{"role": "user", "content": "hi"}])
        assert req.messages == [{"role": "user", "content": "hi"}]
        assert req.tools is None
        assert req.params is None
        assert req.session_id is None

    def test_full_request_fields(self) -> None:
        tools = [{"type": "function", "function": {"name": "bash"}}]
        params = {"api_base": "https://proxy.example", "thinking": {"budget": 1}}
        req = LlmRequest(
            messages=[{"role": "user", "content": "hi"}],
            tools=tools,
            params=params,
            session_id="sess_1",
        )
        assert req.tools is tools
        assert req.params is params
        assert req.session_id == "sess_1"


class TestLlmResponseShape:
    def test_from_message_projects_content_and_tool_calls(self) -> None:
        tc = {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}
        msg = {"role": "assistant", "content": "hello", "tool_calls": [tc]}
        resp = LlmResponse.from_message(
            msg, usage={"input_tokens": 1}, cost=0.25, finish_reason="tool_calls"
        )
        assert resp.content == "hello"
        assert resp.tool_calls == [tc]
        assert resp.finish_reason == "tool_calls"
        assert resp.usage == {"input_tokens": 1}
        assert resp.cost == 0.25
        # The full provider message is retained opaquely for persistence.
        assert resp.message is msg

    def test_from_message_normalizes_absent_fields(self) -> None:
        # A tool-calls-only turn carries ``content`` already normalized to "".
        resp = LlmResponse.from_message(
            {"role": "assistant", "content": ""}, usage={}, cost=None, finish_reason="stop"
        )
        assert resp.content == ""
        assert resp.tool_calls == []
        assert resp.cost is None

    def test_from_message_retains_thinking_blocks_in_message(self) -> None:
        msg = {
            "role": "assistant",
            "content": "ok",
            "thinking_blocks": [{"thinking": "t", "signature": "s"}],
        }
        resp = LlmResponse.from_message(msg, usage={}, cost=None, finish_reason="stop")
        # The opaque message survives intact (no behavior change for persistence).
        assert resp.message["thinking_blocks"] == [{"thinking": "t", "signature": "s"}]


@pytest.mark.asyncio
async def test_call_litellm_consumes_request_produces_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``call_litellm`` takes an ``LlmRequest`` and returns an ``LlmResponse``."""

    async def fake_acompletion(**_kwargs: object) -> _DictResponse:
        return _envelope({"role": "assistant", "content": "hi there"}, "stop")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    resp = await completion.call_litellm(
        LlmRequest(messages=[{"role": "user", "content": "hi"}], session_id="sess_a"),
        model="openai/gpt-5.5",
    )
    assert isinstance(resp, LlmResponse)
    assert resp.content == "hi there"
    assert resp.finish_reason == "stop"
    assert resp.usage == {
        "input_tokens": 10,
        "output_tokens": 5,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    assert resp.message["role"] == "assistant"


@pytest.mark.asyncio
async def test_params_round_trips_litellm_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acceptance: ``request.params`` carries ``litellm_extra`` verbatim into
    the litellm call kwargs (e.g. an agent's ``api_base`` + a custom knob)."""
    captured: dict[str, object] = {}

    async def fake_acompletion(**kwargs: object) -> _DictResponse:
        captured.update(kwargs)
        return _envelope({"role": "assistant", "content": ""}, "stop")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    litellm_extra = {"api_base": "https://proxy.example/v1", "temperature": 0.3}
    await completion.call_litellm(
        LlmRequest(
            messages=[{"role": "user", "content": "hi"}],
            params=litellm_extra,
            session_id="sess_extra",
        ),
        model="anthropic/claude-opus-4-6",
    )

    assert captured.get("api_base") == "https://proxy.example/v1"
    assert captured.get("temperature") == 0.3


@pytest.mark.asyncio
async def test_call_litellm_openai_cache_key_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No behavior change: the openai prompt-cache-key path still keys on
    ``request.session_id`` exactly as before the refactor."""
    captured: dict[str, object] = {}

    async def fake_acompletion(**kwargs: object) -> _DictResponse:
        captured.update(kwargs)
        return _envelope({"role": "assistant", "content": ""}, "stop")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    await completion.call_litellm(
        LlmRequest(messages=[{"role": "user", "content": "hi"}], session_id="sess_cache"),
        model="openai/gpt-5.5",
    )

    extra_body = captured.get("extra_body")
    assert isinstance(extra_body, dict)
    assert extra_body.get("prompt_cache_key") == "sess_cache"
