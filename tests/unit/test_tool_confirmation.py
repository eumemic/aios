"""Unit tests for tool confirmation helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from aios.harness.loop import _resolve_permission
from aios.models.agents import ToolSpec
from aios.models.events import Event
from aios.services.sessions import _find_tool_call


def _event(seq: int, data: dict[str, Any], kind: str = "message") -> Event:
    return Event(
        id=f"evt_{seq}",
        session_id="sess_test",
        seq=seq,
        kind=kind,  # type: ignore[arg-type]
        data=data,
        created_at=datetime.now(timezone.utc),
    )


class TestFindToolCall:
    def test_finds_existing_call(self) -> None:
        events = [
            _event(1, {"role": "user", "content": "hello"}),
            _event(2, {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "bash", "arguments": '{"command": "echo hi"}'}},
                    {"id": "call_2", "type": "function", "function": {"name": "read", "arguments": '{"file_path": "/tmp/x"}'}},
                ],
            }),
        ]
        tc = _find_tool_call(events, "call_2")
        assert tc is not None
        assert tc["id"] == "call_2"
        assert tc["function"]["name"] == "read"

    def test_returns_none_for_missing(self) -> None:
        events = [
            _event(1, {"role": "user", "content": "hello"}),
            _event(2, {"role": "assistant", "content": "bye"}),
        ]
        assert _find_tool_call(events, "call_missing") is None

    def test_searches_in_reverse(self) -> None:
        """If the same call_id appears in multiple messages, the most recent wins."""
        events = [
            _event(1, {
                "role": "assistant",
                "tool_calls": [{"id": "call_x", "function": {"name": "bash", "arguments": "{}"}}],
            }),
            _event(2, {"role": "tool", "tool_call_id": "call_x", "content": "done"}),
            _event(3, {
                "role": "assistant",
                "tool_calls": [{"id": "call_x", "function": {"name": "read", "arguments": "{}"}}],
            }),
        ]
        tc = _find_tool_call(events, "call_x")
        assert tc is not None
        assert tc["function"]["name"] == "read"

    def test_empty_events(self) -> None:
        assert _find_tool_call([], "call_1") is None

    def test_no_tool_calls_in_assistant(self) -> None:
        events = [
            _event(1, {"role": "assistant", "content": "just text"}),
        ]
        assert _find_tool_call(events, "call_1") is None


class TestResolvePermission:
    def test_builtin_with_permission(self) -> None:
        tools = [
            ToolSpec(type="bash", permission="always_ask"),
            ToolSpec(type="read"),
        ]
        assert _resolve_permission("bash", tools) == "always_ask"

    def test_builtin_without_permission(self) -> None:
        tools = [ToolSpec(type="bash")]
        assert _resolve_permission("bash", tools) is None

    def test_custom_tool(self) -> None:
        tools = [
            ToolSpec(
                type="custom",
                name="get_weather",
                description="weather",
                input_schema={"type": "object"},
                permission="always_ask",
            ),
        ]
        assert _resolve_permission("get_weather", tools) == "always_ask"

    def test_unknown_tool_returns_none(self) -> None:
        tools = [ToolSpec(type="bash")]
        assert _resolve_permission("nonexistent", tools) is None

    def test_empty_tools(self) -> None:
        assert _resolve_permission("bash", []) is None
