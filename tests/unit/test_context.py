"""Unit tests for the context builder (should_call_model + build_messages).

Uses lightweight FakeEvent objects to avoid touching the DB.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from aios.harness.context import build_messages, should_call_model
from aios.models.events import Event


def _evt(
    seq: int,
    role: str,
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    tool_call_id: str | None = None,
    content: str = "",
) -> Event:
    """Build a minimal message Event for testing."""
    data: dict[str, Any] = {"role": role, "content": content}
    if tool_calls is not None:
        data["tool_calls"] = tool_calls
    if tool_call_id is not None:
        data["tool_call_id"] = tool_call_id
    return Event(
        id=f"evt_{seq}",
        session_id="sess_01TEST",
        seq=seq,
        kind="message",
        data=data,
        created_at=datetime.now(tz=UTC),
    )


def _tc(call_id: str, name: str = "bash") -> dict[str, Any]:
    """Build a tool_call dict."""
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}


# ─── should_call_model ──────────────────────────────────────────────────────


class TestShouldCallModel:
    def test_empty_events_returns_false(self) -> None:
        assert should_call_model([]) is False

    def test_first_user_message_returns_true(self) -> None:
        events = [_evt(1, "user", content="hello")]
        assert should_call_model(events) is True

    def test_duplicate_wake_no_new_events(self) -> None:
        events = [
            _evt(1, "user", content="hello"),
            _evt(2, "assistant", content="hi"),
        ]
        assert should_call_model(events) is False

    def test_user_injection_returns_true(self) -> None:
        events = [
            _evt(1, "user", content="do X"),
            _evt(2, "assistant", tool_calls=[_tc("a")]),
            _evt(3, "user", content="also do Y"),
        ]
        assert should_call_model(events) is True

    def test_all_tools_resolved_returns_true(self) -> None:
        events = [
            _evt(1, "user", content="do X"),
            _evt(2, "assistant", tool_calls=[_tc("a"), _tc("b")]),
            _evt(3, "tool", tool_call_id="a", content="result a"),
            _evt(4, "tool", tool_call_id="b", content="result b"),
        ]
        assert should_call_model(events) is True

    def test_partial_tools_returns_false(self) -> None:
        events = [
            _evt(1, "user", content="do X"),
            _evt(2, "assistant", tool_calls=[_tc("a"), _tc("b"), _tc("c")]),
            _evt(3, "tool", tool_call_id="a", content="result a"),
        ]
        assert should_call_model(events) is False

    def test_batch_from_earlier_assistant_completes(self) -> None:
        """The scenario from the design conversation: assistant at seq=2 requested
        tool X. A later assistant at seq=4 has no tool_calls. X completes at seq=5.
        should_call_model should see that batch (seq=2) is fully resolved."""
        events = [
            _evt(1, "user", content="do X"),
            _evt(2, "assistant", tool_calls=[_tc("x")]),
            _evt(3, "user", content="how's it going?"),
            _evt(4, "assistant", content="still working on it"),
            _evt(5, "tool", tool_call_id="x", content="done"),
        ]
        assert should_call_model(events) is True

    def test_stale_tool_result_only_returns_false(self) -> None:
        """Tool result for an already-responded-to batch doesn't trigger."""
        events = [
            _evt(1, "user", content="do X"),
            _evt(2, "assistant", tool_calls=[_tc("a")]),
            _evt(3, "tool", tool_call_id="a", content="done"),
            _evt(4, "assistant", content="X is done"),
            # No new events after seq=4 that need a response
        ]
        assert should_call_model(events) is False

    def test_two_batches_one_complete(self) -> None:
        """Two assistant messages with tool_calls. Only batch 2 is fully resolved."""
        events = [
            _evt(1, "user", content="do X and Y"),
            _evt(2, "assistant", tool_calls=[_tc("x1"), _tc("x2")]),
            _evt(3, "tool", tool_call_id="x1", content="done"),
            _evt(4, "tool", tool_call_id="x2", content="done"),
            _evt(5, "assistant", tool_calls=[_tc("y1")]),
            _evt(6, "tool", tool_call_id="y1", content="done"),
        ]
        assert should_call_model(events) is True


# ─── build_messages ──────────────────────────────────────────────────────────


class TestBuildMessages:
    def test_simple_conversation(self) -> None:
        events = [
            _evt(1, "user", content="hello"),
            _evt(2, "assistant", content="hi there"),
        ]
        msgs = build_messages(
            events, system_prompt="you are helpful", window_min=50_000, window_max=150_000
        )
        assert msgs[0] == {"role": "system", "content": "you are helpful"}
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"

    def test_tool_results_grouped_with_assistant(self) -> None:
        events = [
            _evt(1, "user", content="do it"),
            _evt(2, "assistant", tool_calls=[_tc("a"), _tc("b")]),
            _evt(3, "tool", tool_call_id="a", content="result a"),
            _evt(4, "tool", tool_call_id="b", content="result b"),
        ]
        msgs = build_messages(events, system_prompt=None, window_min=50_000, window_max=150_000)
        # Order: user, assistant, tool_a, tool_b
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        assert msgs[2]["role"] == "tool"
        assert msgs[2]["tool_call_id"] == "a"
        assert msgs[3]["role"] == "tool"
        assert msgs[3]["tool_call_id"] == "b"

    def test_pending_tool_synthesized(self) -> None:
        events = [
            _evt(1, "user", content="do it"),
            _evt(2, "assistant", tool_calls=[_tc("a"), _tc("b")]),
            _evt(3, "tool", tool_call_id="a", content="result a"),
            # b is still pending
        ]
        msgs = build_messages(events, system_prompt=None, window_min=50_000, window_max=150_000)
        assert msgs[2]["tool_call_id"] == "a"
        assert "result a" in msgs[2]["content"]
        # b should be a synthetic pending result
        assert msgs[3]["tool_call_id"] == "b"
        assert "pending" in msgs[3]["content"]

    def test_out_of_order_tool_result_reordered(self) -> None:
        """Tool result arrives after a user message in seq order,
        but gets placed right after its assistant message in the prompt."""
        events = [
            _evt(1, "user", content="do X"),
            _evt(2, "assistant", tool_calls=[_tc("x")]),
            _evt(3, "user", content="how goes?"),  # user injection before tool completes
            _evt(4, "tool", tool_call_id="x", content="X done"),
        ]
        msgs = build_messages(events, system_prompt=None, window_min=50_000, window_max=150_000)
        # Should be: user, assistant+tool_calls, tool_result_x, user_injection
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "do X"
        assert msgs[1]["role"] == "assistant"
        assert msgs[2]["role"] == "tool"
        assert msgs[2]["tool_call_id"] == "x"
        assert msgs[3]["role"] == "user"
        assert msgs[3]["content"] == "how goes?"

    def test_no_system_prompt_when_none(self) -> None:
        events = [_evt(1, "user", content="hi")]
        msgs = build_messages(events, system_prompt=None, window_min=50_000, window_max=150_000)
        assert msgs[0]["role"] == "user"

    def test_multiple_tool_batches(self) -> None:
        events = [
            _evt(1, "user", content="start"),
            _evt(2, "assistant", tool_calls=[_tc("a")]),
            _evt(3, "tool", tool_call_id="a", content="done a"),
            _evt(4, "assistant", tool_calls=[_tc("b")]),
            _evt(5, "tool", tool_call_id="b", content="done b"),
            _evt(6, "assistant", content="all done"),
        ]
        msgs = build_messages(events, system_prompt=None, window_min=50_000, window_max=150_000)
        roles = [m["role"] for m in msgs]
        assert roles == ["user", "assistant", "tool", "assistant", "tool", "assistant"]
