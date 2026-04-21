"""Unit tests for ``aios.tail.format_event`` — the one-line session-event renderer."""

from __future__ import annotations

from typing import Any

from aios.cli.tail_format import format_event


def _msg(
    seq: int,
    role: str,
    *,
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    tool_call_id: str | None = None,
    is_error: bool = False,
    orig_channel: str | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {"role": role}
    if content is not None:
        data["content"] = content
    if tool_calls is not None:
        data["tool_calls"] = tool_calls
    if tool_call_id is not None:
        data["tool_call_id"] = tool_call_id
    if is_error:
        data["is_error"] = True
    event: dict[str, Any] = {"seq": seq, "kind": "message", "data": data}
    if orig_channel is not None:
        event["orig_channel"] = orig_channel
    return event


class TestUserEvents:
    def test_plain_user_message(self) -> None:
        e = _msg(144, "user", content="hey")
        assert format_event(e) == "#144 USER: hey"

    def test_user_message_with_channel(self) -> None:
        e = _msg(144, "user", content="hey", orig_channel="signal/dm/alice")
        assert format_event(e) == "#144 USER[signal/dm/alice]: hey"

    def test_user_message_truncates_long_content(self) -> None:
        e = _msg(1, "user", content="x" * 500)
        out = format_event(e)
        assert out is not None
        assert len(out) < 500
        assert out.endswith("…")


class TestAssistantEvents:
    def test_bare_text_assistant(self) -> None:
        e = _msg(150, "assistant", content="hello world")
        assert format_event(e) == "#150 AGENT(bare): hello world"

    def test_monologue_assistant(self) -> None:
        e = _msg(
            151,
            "assistant",
            content="INTERNAL_MONOLOGUE_NOT_SEEN_BY_USER: thinking...",
        )
        assert format_event(e) == "#151 AGENT(mono): thinking..."

    def test_tool_call_assistant(self) -> None:
        e = _msg(
            147,
            "assistant",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command":"ls"}'},
                }
            ],
        )
        assert format_event(e) == '#147 AGENT→bash: {"command":"ls"}'

    def test_multiple_tool_calls_concatenated(self) -> None:
        e = _msg(
            149,
            "assistant",
            tool_calls=[
                {"function": {"name": "read", "arguments": "{}"}},
                {"function": {"name": "write", "arguments": "{}"}},
            ],
        )
        assert format_event(e) == "#149 AGENT→read: {}, write: {}"

    def test_silent_assistant_flagged(self) -> None:
        e = _msg(1764, "assistant", content="")
        assert format_event(e) == "#1764 AGENT(silent) ⚠ no tool, no text"

    def test_silent_assistant_with_null_content(self) -> None:
        e = _msg(1765, "assistant", content=None)
        assert format_event(e) == "#1765 AGENT(silent) ⚠ no tool, no text"

    def test_mixed_content_and_tool_calls_prefers_tool_calls(self) -> None:
        e = _msg(
            200,
            "assistant",
            content="here's my plan",
            tool_calls=[{"function": {"name": "bash", "arguments": "{}"}}],
        )
        assert format_event(e) == "#200 AGENT→bash: {}"


class TestToolEvents:
    def test_normal_tool_result(self) -> None:
        e = _msg(148, "tool", content="file contents...", tool_call_id="call_1")
        assert format_event(e) == "#148 TOOL[call_1]: file contents..."

    def test_error_tool_result(self) -> None:
        e = _msg(
            1303,
            "tool",
            content="MCP server 'conn_abc' not found",
            tool_call_id="call_9",
            is_error=True,
        )
        assert format_event(e) == "#1303 TOOL⚠ ERROR[call_9]: MCP server 'conn_abc' not found"

    def test_tool_result_without_tool_call_id(self) -> None:
        e = _msg(149, "tool", content="ok")
        assert format_event(e) == "#149 TOOL[?]: ok"


class TestLifecycleEvents:
    def test_turn_ended(self) -> None:
        e: dict[str, Any] = {
            "seq": 200,
            "kind": "lifecycle",
            "data": {"event": "turn_ended", "status": "idle", "stop_reason": "end_turn"},
        }
        out = format_event(e)
        assert out is not None
        assert out.startswith("#200 LIFECYCLE ")
        assert "turn_ended" in out

    def test_interrupted(self) -> None:
        e: dict[str, Any] = {
            "seq": 201,
            "kind": "lifecycle",
            "data": {"event": "interrupted", "status": "idle", "stop_reason": "interrupt"},
        }
        out = format_event(e)
        assert out is not None
        assert "interrupted" in out


class TestSkippableEvents:
    def test_span_events_return_none(self) -> None:
        e: dict[str, Any] = {
            "seq": 10,
            "kind": "span",
            "data": {"event": "model_request_start"},
        }
        assert format_event(e) is None

    def test_unknown_kind_returns_none(self) -> None:
        e: dict[str, Any] = {"seq": 99, "kind": "weird_new_kind", "data": {}}
        assert format_event(e) is None
