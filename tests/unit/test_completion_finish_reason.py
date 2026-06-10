"""Unit tests for ``finish_reason`` threading in ``_unpack_litellm_response``.

The refusal-surfacing logic in ``loop.py`` keys on the standardized
``finish_reason`` litellm puts on each choice. These tests pin that the unpack
helper extracts it for a normal completion, a tool-call completion, and a
``content_filter`` refusal, and that absence degrades to ``None``.
"""

from __future__ import annotations

from typing import Any

from aios.harness.completion import _unpack_litellm_response


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
