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
        ctx = build_messages(
            events,
            system_prompt="you are helpful",
        )
        msgs = ctx.messages
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
        msgs = build_messages(
            events,
            system_prompt=None,
        ).messages
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
        msgs = build_messages(
            events,
            system_prompt=None,
        ).messages
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
        msgs = build_messages(
            events,
            system_prompt=None,
        ).messages
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
        msgs = build_messages(
            events,
            system_prompt=None,
        ).messages
        assert msgs[0]["role"] == "user"

    def test_monotonic_blind_spot_shows_pending_then_injects_real(self) -> None:
        """When a tool result arrives during inference (seq between reacting_to
        and the assistant's own seq), the paired position shows PENDING (what the
        assistant actually saw) and the real result is injected as a user message
        after the horizon-setting assistant. This preserves prompt cache
        stability (monotonicity)."""
        events = [
            _evt(1, "user", content="run sleep 15"),
            _evt(2, "assistant", tool_calls=[_tc("bash_1")], content=""),
            # seq=3: user injection (before tool completes)
            _evt(3, "user", content="status?"),
            # seq=4: tool result (arrived DURING inference for assistant at seq=5)
            _evt(4, "tool", tool_call_id="bash_1", content='{"stdout": "DONE"}'),
            # seq=5: stale assistant response (reacting_to=3, saw pending)
            _evt(5, "assistant", content="still running"),
        ]
        # Simulate reacting_to on the assistant messages
        events[1].data["reacting_to"] = 1  # assistant at seq=2 reacted to user at seq=1
        events[4].data["reacting_to"] = 3  # assistant at seq=5 reacted to user at seq=3
        # (tool result at seq=4 has seq > reacting_to=3, so assistant saw pending)

        msgs = build_messages(
            events,
            system_prompt=None,
        ).messages

        # The paired position for bash_1 should show PENDING (not real)
        # because the assistant at seq=5 saw it as pending
        paired_tool = next(
            m for m in msgs if m.get("tool_call_id") == "bash_1" and m["role"] == "tool"
        )
        assert "pending" in paired_tool["content"], (
            "paired position should show pending (what the assistant actually saw)"
        )

        # The stale assistant should appear coherently after the pending result
        stale_asst = next(m for m in msgs if m.get("content") == "still running")
        assert stale_asst is not None

        # The real result should be injected as a user message after the stale assistant
        injected = next(
            (m for m in msgs if m["role"] == "user" and "DONE" in m.get("content", "")),
            None,
        )
        assert injected is not None, "real tool result should be injected as a user message"
        # The injected message should come AFTER the stale assistant
        injected_idx = msgs.index(injected)
        stale_idx = msgs.index(stale_asst)
        assert injected_idx > stale_idx, (
            "injected result should come after the stale assistant response"
        )

    def test_monotonic_no_rewrite_when_result_seen_as_real(self) -> None:
        """When the assistant saw the real result (result.seq <= reacting_to),
        the paired position shows the real result normally. No injection needed."""
        events = [
            _evt(1, "user", content="do it"),
            _evt(2, "assistant", tool_calls=[_tc("a")], content=""),
            _evt(3, "tool", tool_call_id="a", content="done"),
            _evt(4, "assistant", content="got it"),
        ]
        events[1].data["reacting_to"] = 1
        events[3].data["reacting_to"] = 3  # saw tool result at seq=3

        msgs = build_messages(
            events,
            system_prompt=None,
        ).messages
        # Paired position should show REAL result (not pending)
        paired_tool = next(m for m in msgs if m.get("tool_call_id") == "a")
        assert "done" in paired_tool["content"]
        assert "pending" not in paired_tool["content"]
        # No injected user messages for the tool result
        user_msgs = [m for m in msgs if m["role"] == "user"]
        assert len(user_msgs) == 1  # only the original user message

    def test_multiple_tool_batches(self) -> None:
        events = [
            _evt(1, "user", content="start"),
            _evt(2, "assistant", tool_calls=[_tc("a")]),
            _evt(3, "tool", tool_call_id="a", content="done a"),
            _evt(4, "assistant", tool_calls=[_tc("b")]),
            _evt(5, "tool", tool_call_id="b", content="done b"),
            _evt(6, "assistant", content="all done"),
        ]
        msgs = build_messages(
            events,
            system_prompt=None,
        ).messages
        roles = [m["role"] for m in msgs]
        assert roles == ["user", "assistant", "tool", "assistant", "tool", "assistant"]

    def test_prune_orphan_tool_results_at_start(self) -> None:
        """If DB windowing drops an assistant but keeps its tool results,
        those orphan tool results should be pruned from the start."""
        # Simulate pre-windowed events where the assistant at seq=2 was
        # dropped but its tool result at seq=3 was kept.
        events = [
            _evt(3, "tool", tool_call_id="a", content="result a"),
            _evt(4, "user", content="next question"),
            _evt(5, "assistant", content="answer"),
        ]
        msgs = build_messages(events, system_prompt=None).messages
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "next question"
        assert msgs[1]["role"] == "assistant"

    def test_prune_partial_assistant_tool_group(self) -> None:
        """If DB windowing keeps an assistant with tool_calls but dropped
        one of its paired results, the incomplete group should be pruned."""
        # Simulate pre-windowed events where tool result "a" was dropped
        # by the window boundary but "b" was kept.
        events = [
            _evt(10, "assistant", tool_calls=[_tc("a"), _tc("b")]),
            _evt(12, "tool", tool_call_id="b", content="small"),
            _evt(13, "user", content="next"),
            _evt(14, "assistant", content="response"),
        ]
        events[0].data["reacting_to"] = 0
        events[3].data["reacting_to"] = 12
        msgs = build_messages(events, system_prompt=None).messages
        # The incomplete group (assistant + orphan tool "b") should be pruned.
        for m in msgs:
            if m.get("role") == "tool":
                tc_id = m.get("tool_call_id")
                has_parent = any(
                    tc_id in {tc["id"] for tc in prior.get("tool_calls") or []}
                    for prior in msgs[: msgs.index(m)]
                    if prior.get("role") == "assistant"
                )
                assert has_parent, f"orphan tool result for {tc_id}"


# ─── monotonicity ──────────────────────────────────────────────────────────


def _assert_prefix(short: list[dict], long: list[dict]) -> None:
    """Assert that *short* is a message-for-message prefix of *long*."""
    assert len(short) <= len(long), (
        f"short ({len(short)} msgs) is longer than long ({len(long)} msgs)"
    )
    for i, (a, b) in enumerate(zip(short, long, strict=False)):
        assert a == b, (
            f"monotonicity violation at index {i}:\n  short[{i}] = {a!r}\n  long[{i}]  = {b!r}"
        )


class TestMonotonicity:
    """build_messages(L1) must be a prefix of build_messages(L2) whenever L1
    is a prefix of L2.  This is the property that keeps the prompt prefix
    cache stable between successive inference calls."""

    @staticmethod
    def _build(events: list[Event]) -> list[dict]:
        return build_messages(
            events,
            system_prompt=None,
        ).messages

    def test_injection_stable_when_assistant_appended(self) -> None:
        """A blind-spot tool result is injected as a user message after
        the horizon-setter.  When the model responds (new assistant
        appended), the injection must not shift — the new assistant
        should appear AFTER the injection, preserving the prefix."""
        # L1: blind-spot result exists, model is about to be called.
        #   seq 4 (tool result) > reacting_to=3 of asst at seq 5
        #   → paired position shows PENDING, real result injected inline.
        l1 = [
            _evt(1, "user", content="run sleep 15"),
            _evt(2, "assistant", tool_calls=[_tc("bash_1")]),
            _evt(3, "user", content="status?"),
            _evt(4, "tool", tool_call_id="bash_1", content="DONE"),
            _evt(5, "assistant", content="still running"),
        ]
        l1[1].data["reacting_to"] = 1
        l1[4].data["reacting_to"] = 3  # blind to tool at seq 4

        # L2: model saw the injection and responded.
        l2 = [*l1, _evt(6, "assistant", content="ah it finished")]
        l2[5].data["reacting_to"] = 4

        ctx1 = self._build(l1)
        ctx2 = self._build(l2)

        # ctx1 should be a prefix of ctx2.
        _assert_prefix(ctx1, ctx2)

    def test_inline_injection_position(self) -> None:
        """Blind-spot injection appears right after the horizon-setter
        assistant, not at the absolute tail of the message list."""
        events = [
            _evt(1, "user", content="run it"),
            _evt(2, "assistant", tool_calls=[_tc("t1")]),
            _evt(3, "tool", tool_call_id="t1", content="RESULT"),
            # asst at seq=4 is the horizon-setter for seq=2.
            # Its reacting_to=1, so tool at seq=3 > horizon=1 → blind spot.
            _evt(4, "assistant", content="checking..."),
            _evt(5, "user", content="anything else?"),
            _evt(6, "assistant", content="nope"),
        ]
        events[1].data["reacting_to"] = 1
        events[3].data["reacting_to"] = 1  # didn't see tool at seq=3
        events[5].data["reacting_to"] = 5

        msgs = self._build(events)
        roles = [m["role"] for m in msgs]
        # Injection (user) sits between horizon-setter and the following user msg.
        assert roles == ["user", "assistant", "tool", "assistant", "user", "user", "assistant"]
        assert "RESULT" in msgs[4]["content"]

    def test_horizon_setter_with_tool_calls_injection_after(self) -> None:
        """When the horizon-setter itself has tool_calls, the blind-spot
        injection goes after the horizon-setter's own tool results."""
        events = [
            _evt(1, "user", content="do A and B"),
            _evt(2, "assistant", tool_calls=[_tc("a1")]),
            _evt(3, "tool", tool_call_id="a1", content="A done"),
            # Asst at seq=4: horizon-setter for seq=2, with its own tool_calls.
            _evt(4, "assistant", tool_calls=[_tc("b1")]),
            _evt(5, "tool", tool_call_id="b1", content="B done"),
            _evt(6, "assistant", content="all done"),
        ]
        events[1].data["reacting_to"] = 1
        events[3].data["reacting_to"] = 1  # blind to a1 at seq=3
        events[5].data["reacting_to"] = 5

        msgs = self._build(events)
        roles = [m["role"] for m in msgs]
        assert roles == ["user", "assistant", "tool", "assistant", "tool", "user", "assistant"]
        assert "pending" in msgs[2]["content"]
        assert msgs[4]["tool_call_id"] == "b1"
        assert "B done" in msgs[4]["content"]
        assert "A done" in msgs[5]["content"]

    def test_multiple_blind_spot_tools_same_assistant(self) -> None:
        """Multiple blind-spot tools from the same assistant are all injected
        inline after the same horizon-setter."""
        events = [
            _evt(1, "user", content="run two"),
            _evt(2, "assistant", tool_calls=[_tc("x"), _tc("y")]),
            _evt(3, "tool", tool_call_id="x", content="X done"),
            _evt(4, "tool", tool_call_id="y", content="Y done"),
            _evt(5, "assistant", content="both pending..."),
        ]
        events[1].data["reacting_to"] = 1
        events[4].data["reacting_to"] = 1  # blind to both tools

        msgs = self._build(events)
        roles = [m["role"] for m in msgs]
        assert roles == ["user", "assistant", "tool", "tool", "assistant", "user", "user"]
        assert "pending" in msgs[2]["content"]
        assert "pending" in msgs[3]["content"]
        assert "X done" in msgs[5]["content"]
        assert "Y done" in msgs[6]["content"]

    def test_multiple_assistants_with_blind_spots(self) -> None:
        """Two different assistants each with blind-spot tools inject after
        their respective horizon-setters."""
        events = [
            _evt(1, "user", content="go"),
            _evt(2, "assistant", tool_calls=[_tc("a")]),
            _evt(3, "tool", tool_call_id="a", content="A done"),
            # asst at seq=4 is horizon-setter for seq=2, and itself has tool_calls.
            _evt(4, "assistant", tool_calls=[_tc("b")]),
            _evt(5, "tool", tool_call_id="b", content="B done"),
            # asst at seq=6 is horizon-setter for seq=4.
            _evt(6, "assistant", content="wrapping up"),
        ]
        events[1].data["reacting_to"] = 1
        events[3].data["reacting_to"] = 1  # blind to a (seq=3 > horizon=1)
        events[5].data["reacting_to"] = 3  # blind to b (seq=5 > horizon=3)

        msgs = self._build(events)
        roles = [m["role"] for m in msgs]
        assert roles == [
            "user",
            "assistant",
            "tool",
            "assistant",
            "tool",
            "user",
            "assistant",
            "user",
        ]
        assert "pending" in msgs[2]["content"]
        assert "pending" in msgs[4]["content"]
        assert "A done" in msgs[5]["content"]
        assert "B done" in msgs[7]["content"]

    def test_monotonicity_across_three_successive_appends(self) -> None:
        """L1 ⊂ L2 ⊂ L3: prefix preserved at each step."""
        base = [
            _evt(1, "user", content="run sleep 15"),
            _evt(2, "assistant", tool_calls=[_tc("bash_1")]),
            _evt(3, "user", content="status?"),
            _evt(4, "tool", tool_call_id="bash_1", content="DONE"),
            _evt(5, "assistant", content="still running"),
        ]
        base[1].data["reacting_to"] = 1
        base[4].data["reacting_to"] = 3

        l1 = list(base)
        l2 = [*l1, _evt(6, "assistant", content="ah it finished")]
        l2[5].data["reacting_to"] = 4
        l3 = [*l2, _evt(7, "user", content="great, now do Y")]

        ctx1, ctx2, ctx3 = self._build(l1), self._build(l2), self._build(l3)
        _assert_prefix(ctx1, ctx2)
        _assert_prefix(ctx2, ctx3)

    def test_reacting_to_includes_inline_injection_seq(self) -> None:
        """ContextResult.reacting_to must account for the seq of blind-spot
        tool results that are injected inline."""
        events = [
            _evt(1, "user", content="run it"),
            _evt(2, "assistant", tool_calls=[_tc("t1")]),
            _evt(3, "tool", tool_call_id="t1", content="RESULT"),
            _evt(4, "assistant", content="checking..."),
        ]
        events[1].data["reacting_to"] = 1
        events[3].data["reacting_to"] = 1

        ctx = build_messages(events, system_prompt=None)
        assert ctx.reacting_to >= 3
