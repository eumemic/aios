"""Unit tests for assistant message normalization in completion.py."""

from __future__ import annotations

import litellm

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

    def test_content_null_becomes_empty_string(self) -> None:
        """content: None (from providers like GPT-5.4-mini) becomes ""."""
        msg: dict[str, object] = {"role": "assistant", "content": None}
        result = _normalize_message(msg)
        assert result["content"] == ""

    def test_content_absent_becomes_empty_string(self) -> None:
        """Message missing content key entirely gets content: ""."""
        msg: dict[str, object] = {"role": "assistant"}
        result = _normalize_message(msg)
        assert result["content"] == ""

    def test_content_nonempty_preserved(self) -> None:
        """Non-null content passes through unchanged."""
        msg: dict[str, object] = {"role": "assistant", "content": "hello"}
        result = _normalize_message(msg)
        assert result["content"] == "hello"

    def test_content_null_with_tool_calls_null_both_fixed(self) -> None:
        """Both null content and null tool_calls are normalized together."""
        msg: dict[str, object] = {
            "role": "assistant",
            "content": None,
            "tool_calls": None,
        }
        result = _normalize_message(msg)
        assert result["content"] == ""
        assert "tool_calls" not in result


def test_modify_params_enabled_on_import() -> None:
    """Importing aios.harness.completion sets litellm.modify_params = True.

    Without this flag, replaying an event log with ``content: ""`` +
    ``tool_calls`` assistant turns (emitted by some OpenRouter models)
    against an Anthropic-routed model fails with "text content blocks
    must be non-empty". See
    https://docs.litellm.ai/docs/completion/message_sanitization.
    """
    assert litellm.modify_params is True
