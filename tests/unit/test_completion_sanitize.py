"""Unit tests for assistant message normalization in completion.py."""

from __future__ import annotations

from aios.harness.completion import _normalize_message


class TestNormalizeMessage:
    """Tests for _normalize_message which strips provider quirks."""

    def test_tool_calls_null_stripped(self) -> None:
        """tool_calls: None (from providers like kimi-k2.5) is removed."""
        msg: dict[str, object] = {
            "role": "assistant",
            "content": "hello",
            "tool_calls": None,
        }
        result = _normalize_message(msg)
        assert "tool_calls" not in result
        assert result["content"] == "hello"

    def test_tool_calls_empty_list_preserved(self) -> None:
        """tool_calls: [] is a valid (if unusual) value — keep it."""
        msg: dict[str, object] = {
            "role": "assistant",
            "content": "",
            "tool_calls": [],
        }
        result = _normalize_message(msg)
        assert result["tool_calls"] == []

    def test_tool_calls_with_calls_preserved(self) -> None:
        """Normal tool_calls list passes through unchanged."""
        tc = {"id": "call_1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}
        msg: dict[str, object] = {
            "role": "assistant",
            "content": None,
            "tool_calls": [tc],
        }
        result = _normalize_message(msg)
        assert result["tool_calls"] == [tc]

    def test_tool_calls_absent_unchanged(self) -> None:
        """Message without tool_calls key is not modified."""
        msg: dict[str, object] = {"role": "assistant", "content": "just text"}
        result = _normalize_message(msg)
        assert result == {"role": "assistant", "content": "just text"}
        assert "tool_calls" not in result
