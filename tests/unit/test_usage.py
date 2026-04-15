"""Unit tests for completion.py: usage normalization and cache breakpoints."""

from __future__ import annotations

from typing import Any

from aios.harness.completion import (
    _CACHE_CONTROL,
    _normalize_usage,
    inject_cache_breakpoints,
)


class TestNormalizeUsage:
    """Tests for _normalize_usage which maps LiteLLM fields to our names."""

    def test_standard_openai_fields(self) -> None:
        raw = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        result = _normalize_usage(raw)
        assert result == {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }

    def test_anthropic_cache_fields(self) -> None:
        """Anthropic passes cache fields at the top level via LiteLLM."""
        raw = {
            "prompt_tokens": 200,
            "completion_tokens": 80,
            "total_tokens": 280,
            "cache_read_input_tokens": 50,
            "cache_creation_input_tokens": 30,
        }
        result = _normalize_usage(raw)
        assert result == {
            "input_tokens": 200,
            "output_tokens": 80,
            "cache_read_input_tokens": 50,
            "cache_creation_input_tokens": 30,
        }

    def test_openai_cached_tokens_in_details(self) -> None:
        """OpenAI puts cache reads in prompt_tokens_details.cached_tokens."""
        raw = {
            "prompt_tokens": 300,
            "completion_tokens": 100,
            "total_tokens": 400,
            "prompt_tokens_details": {"cached_tokens": 120},
        }
        result = _normalize_usage(raw)
        assert result == {
            "input_tokens": 300,
            "output_tokens": 100,
            "cache_read_input_tokens": 120,
            "cache_creation_input_tokens": 0,
        }

    def test_anthropic_cache_read_takes_precedence(self) -> None:
        """Top-level cache_read_input_tokens wins over prompt_tokens_details."""
        raw = {
            "prompt_tokens": 200,
            "completion_tokens": 80,
            "cache_read_input_tokens": 50,
            "prompt_tokens_details": {"cached_tokens": 999},
        }
        result = _normalize_usage(raw)
        assert result["cache_read_input_tokens"] == 50

    def test_empty_dict(self) -> None:
        result = _normalize_usage({})
        assert result == {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }

    def test_none_values_treated_as_zero(self) -> None:
        raw = {
            "prompt_tokens": None,
            "completion_tokens": None,
            "prompt_tokens_details": None,
        }
        result = _normalize_usage(raw)
        assert result == {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }

    def test_model_dump_flattened_prompt_tokens_details(self) -> None:
        """Regression: model_dump() flattens Pydantic objects to dicts with extra keys."""
        raw = {
            "prompt_tokens": 400,
            "completion_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 200, "audio_tokens": None},
        }
        result = _normalize_usage(raw)
        assert result == {
            "input_tokens": 400,
            "output_tokens": 120,
            "cache_read_input_tokens": 200,
            "cache_creation_input_tokens": 0,
        }

    def test_zero_values_preserved(self) -> None:
        raw = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
        result = _normalize_usage(raw)
        assert result == {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }


# ─── inject_cache_breakpoints ─────────────────────────────────────────────


def _msg(role: str, content: str = "") -> dict[str, Any]:
    """Build a minimal message dict."""
    return {"role": role, "content": content}


def _tool_def(name: str) -> dict[str, Any]:
    """Build a minimal OpenAI-format tool definition."""
    return {
        "type": "function",
        "function": {"name": name, "description": f"{name} tool", "parameters": {}},
    }


class TestInjectCacheBreakpoints:
    def test_system_message_annotated(self) -> None:
        msgs = [_msg("system", "you are helpful"), _msg("user", "hi")]
        inject_cache_breakpoints(msgs, None)
        assert msgs[0]["content"] == [
            {"type": "text", "text": "you are helpful", "cache_control": _CACHE_CONTROL}
        ]

    def test_last_tool_annotated(self) -> None:
        msgs = [_msg("system", "sys"), _msg("user", "hi")]
        tools = [_tool_def("bash"), _tool_def("read")]
        inject_cache_breakpoints(msgs, tools)
        assert "cache_control" not in tools[0]
        assert tools[1]["cache_control"] == _CACHE_CONTROL

    def test_last_conversation_message_annotated(self) -> None:
        msgs = [_msg("system", "sys"), _msg("user", "hi")]
        inject_cache_breakpoints(msgs, None)
        assert msgs[1]["content"] == [
            {"type": "text", "text": "hi", "cache_control": _CACHE_CONTROL}
        ]

    def test_no_system_message(self) -> None:
        """First non-system message is not annotated; only last is."""
        msgs = [_msg("user", "hi"), _msg("assistant", "hello")]
        inject_cache_breakpoints(msgs, None)
        assert msgs[0]["content"] == "hi"  # untouched
        assert msgs[1]["content"] == [
            {"type": "text", "text": "hello", "cache_control": _CACHE_CONTROL}
        ]

    def test_no_tools(self) -> None:
        msgs = [_msg("system", "sys"), _msg("user", "hi")]
        inject_cache_breakpoints(msgs, None)
        # No crash; system and last message still annotated via content blocks.
        assert msgs[0]["content"][0]["cache_control"] == _CACHE_CONTROL
        assert msgs[1]["content"][0]["cache_control"] == _CACHE_CONTROL

    def test_empty_messages(self) -> None:
        inject_cache_breakpoints([], None)  # no crash

    def test_system_only_no_double_annotate(self) -> None:
        """When the only message is the system message, it gets one
        annotation from the system-message rule.  The last-message rule
        skips it to avoid redundancy."""
        msgs = [_msg("system", "sys")]
        inject_cache_breakpoints(msgs, None)
        assert msgs[0]["content"] == [
            {"type": "text", "text": "sys", "cache_control": _CACHE_CONTROL}
        ]

    def test_tool_result_as_last_message(self) -> None:
        msgs = [
            _msg("system", "sys"),
            _msg("user", "do it"),
            {"role": "assistant", "content": "", "tool_calls": [{"id": "a"}]},
            {"role": "tool", "tool_call_id": "a", "content": "done"},
        ]
        inject_cache_breakpoints(msgs, None)
        assert msgs[3]["content"] == [
            {"type": "text", "text": "done", "cache_control": _CACHE_CONTROL}
        ]

    def test_all_three_breakpoints(self) -> None:
        msgs = [_msg("system", "sys"), _msg("user", "hi")]
        tools = [_tool_def("bash")]
        inject_cache_breakpoints(msgs, tools)
        assert msgs[0]["content"][0]["cache_control"] == _CACHE_CONTROL
        assert tools[0]["cache_control"] == _CACHE_CONTROL
        assert msgs[1]["content"][0]["cache_control"] == _CACHE_CONTROL

    def test_content_already_list(self) -> None:
        """When content is already a list of blocks, annotate the last block."""
        msgs = [
            _msg("system", "sys"),
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "part 1"},
                    {"type": "text", "text": "part 2"},
                ],
            },
        ]
        inject_cache_breakpoints(msgs, None)
        assert "cache_control" not in msgs[1]["content"][0]
        assert msgs[1]["content"][1]["cache_control"] == _CACHE_CONTROL
