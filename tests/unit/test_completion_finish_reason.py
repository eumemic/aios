"""Unit tests for ``finish_reason`` threading in ``_unpack_litellm_response``.

The refusal-surfacing logic in ``loop.py`` keys on the standardized
``finish_reason`` litellm puts on each choice. These tests pin that the unpack
helper extracts it for a normal completion, a tool-call completion, and a
``content_filter`` refusal, and that absence degrades to ``None`` — and that the
streaming path's sticky-capture restores a refusal that
``stream_chunk_builder``'s last-wins assembly clobbers.
"""

from __future__ import annotations

from typing import Any

import litellm
import pytest

from aios.harness import completion
from aios.harness.completion import _unpack_litellm_response
from tests.unit.test_completion_timeouts import _StubPool


def _envelope(message: dict[str, Any], finish_reason: str | None) -> dict[str, Any]:
    """Minimal dict-shaped litellm envelope (matches the e2e harness mock)."""
    choice: dict[str, Any] = {"message": message}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    return {
        "choices": [choice],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


class TestUnpackFinishReason:
    def test_threads_stop_for_normal_completion(self) -> None:
        msg, _usage, _cost, finish_reason = _unpack_litellm_response(
            _envelope({"role": "assistant", "content": "hello"}, "stop"),
            source="test",
        )
        assert finish_reason == "stop"
        assert msg["content"] == "hello"

    def test_threads_tool_calls_finish_reason(self) -> None:
        tc = {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}
        _msg, _usage, _cost, finish_reason = _unpack_litellm_response(
            _envelope({"role": "assistant", "content": None, "tool_calls": [tc]}, "tool_calls"),
            source="test",
        )
        assert finish_reason == "tool_calls"

    def test_threads_content_filter_refusal(self) -> None:
        _msg, _usage, _cost, finish_reason = _unpack_litellm_response(
            _envelope({"role": "assistant", "content": ""}, "content_filter"),
            source="test",
        )
        assert finish_reason == "content_filter"

    def test_absent_finish_reason_is_none(self) -> None:
        _msg, _usage, _cost, finish_reason = _unpack_litellm_response(
            _envelope({"role": "assistant", "content": "hi"}, None),
            source="test",
        )
        assert finish_reason is None


class _FakeDelta:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, finish_reason: str | None, content: str | None) -> None:
        self.finish_reason = finish_reason
        self.delta = _FakeDelta(content)


class _FakeChunk:
    def __init__(self, finish_reason: str | None, content: str | None = None) -> None:
        self.choices = [_FakeChoice(finish_reason, content)]


class _FakeStream:
    """Async-iterable of streaming chunks, mirroring litellm's ``acompletion``."""

    def __init__(self, chunks: list[_FakeChunk]) -> None:
        self._it = iter(chunks)

    def __aiter__(self) -> _FakeStream:
        return self

    async def __anext__(self) -> _FakeChunk:
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration from None


def _clobbered_builder(finish_reason: str) -> Any:
    """A ``stream_chunk_builder`` stand-in returning a given assembled
    ``finish_reason`` — used to simulate litellm's last-wins clobber."""
    return lambda chunks: {
        "usage": {},
        "choices": [
            {"message": {"role": "assistant", "content": ""}, "finish_reason": finish_reason}
        ],
    }


class TestStreamStickyContentFilter:
    """``stream_litellm`` must preserve a ``content_filter`` refusal across
    ``stream_chunk_builder``'s last-wins assembly.

    Reproduced against litellm 1.83.4 during PR review, and the behavior is
    unchanged as of 1.91.1: its
    ``stream_chunk_builder`` overwrites ``finish_reason`` from every
    choice-bearing chunk, and the Anthropic streaming adapter defaults
    ``finish_reason=""`` on all but the ``message_delta`` chunk — so a trailing
    usage/stop chunk silently clobbers ``content_filter`` back to ``"stop"``,
    defeating ``loop.REFUSAL_FINISH_REASON`` gating on the SSE/streaming path.
    ``stream_litellm`` captures the refusal off the wire and re-asserts it.
    """

    @pytest.mark.asyncio
    async def test_refusal_survives_trailing_clobber(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # content_filter rides a non-final chunk; a trailing "stop" chunk is
        # what litellm's assembler would last-wins onto the result.
        chunks = [
            _FakeChunk(finish_reason="content_filter"),
            _FakeChunk(finish_reason="stop"),
        ]

        async def fake_acompletion(**_kwargs: object) -> _FakeStream:
            return _FakeStream(chunks)

        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
        monkeypatch.setattr(litellm, "stream_chunk_builder", _clobbered_builder("stop"))

        response = await completion.stream_litellm(
            completion.LlmRequest(
                messages=[{"role": "user", "content": "hi"}],
                session_id="sess_refusal",
            ),
            model="anthropic/claude-fable-5",
            pool=_StubPool(),
        )
        # Without the sticky-capture this would be "stop" (the clobbered value).
        assert response.finish_reason == "content_filter"

    @pytest.mark.asyncio
    async def test_no_spurious_override_on_normal_stream(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The capture must not fabricate a refusal: a stream whose wire never
        carried ``content_filter`` returns the assembled value unchanged."""
        chunks = [
            _FakeChunk(finish_reason=None, content="hi"),
            _FakeChunk(finish_reason="stop"),
        ]

        async def fake_acompletion(**_kwargs: object) -> _FakeStream:
            return _FakeStream(chunks)

        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
        monkeypatch.setattr(litellm, "stream_chunk_builder", _clobbered_builder("stop"))

        response = await completion.stream_litellm(
            completion.LlmRequest(
                messages=[{"role": "user", "content": "hi"}],
                session_id="sess_ok",
            ),
            model="anthropic/claude-fable-5",
            pool=_StubPool(),
        )
        assert response.finish_reason == "stop"
