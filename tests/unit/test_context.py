"""Unit tests for the context builder (should_call_model + build_messages).

Uses lightweight FakeEvent objects to avoid touching the DB.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from aios.harness.channels import build_channels_tail_block
from aios.harness.context import (
    build_messages,
    separate_adjacent_user_messages,
    should_call_model,
)
from aios.models.channel_bindings import ChannelBinding
from aios.models.events import Event


def _binding(address: str, session_id: str = "sess_01TEST") -> ChannelBinding:
    """Minimal ChannelBinding for tail-block construction."""
    now = datetime(2026, 4, 17, tzinfo=UTC)
    parts = address.split("/", 2)
    connector, account, path = parts[0], parts[1], parts[2] if len(parts) > 2 else ""
    return ChannelBinding(
        id=f"cbnd_{abs(hash(address)) & 0xFFFF:04x}",
        connection_id=f"conn_{abs(hash((connector, account))) & 0xFFFF:04x}",
        path=path,
        address=address,
        session_id=session_id,
        created_at=now,
        updated_at=now,
        notification_mode="focal_candidate",
    )


def _full_pipeline(
    events: list[Event],
    bindings: list[ChannelBinding],
    focal_channel: str | None = None,
) -> list[dict[str, Any]]:
    """Compose ``build_messages`` → tail-block append → separator — the
    same sequence ``loop.py:run_session_step`` runs before handing the
    message list to LiteLLM."""
    ctx = build_messages(events, system_prompt=None)
    tail = build_channels_tail_block(bindings, events, focal_channel)
    if tail is not None:
        ctx.messages.append(tail)
    return separate_adjacent_user_messages(ctx.messages)


def _evt(
    seq: int,
    role: str,
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    tool_call_id: str | None = None,
    content: str = "",
    metadata: dict[str, Any] | None = None,
    orig_channel: str | None = None,
    focal_channel_at_arrival: str | None = None,
) -> Event:
    """Build a minimal message Event for testing.

    When a user event is constructed with ``metadata["channel"]`` but
    no explicit ``orig_channel``, the helper auto-stamps it the same
    way :func:`aios.services.sessions.append_user_message` does at the
    real append site, so focal-aware rendering kicks in for these
    events just as it would in production.
    """
    data: dict[str, Any] = {"role": role, "content": content}
    if tool_calls is not None:
        data["tool_calls"] = tool_calls
    if tool_call_id is not None:
        data["tool_call_id"] = tool_call_id
    if metadata is not None:
        data["metadata"] = metadata
    if orig_channel is None and role == "user" and isinstance(metadata, dict):
        ch = metadata.get("channel")
        if isinstance(ch, str):
            orig_channel = ch
    return Event(
        id=f"evt_{seq}",
        session_id="sess_01TEST",
        seq=seq,
        kind="message",
        data=data,
        created_at=datetime.now(tz=UTC),
        orig_channel=orig_channel,
        focal_channel_at_arrival=focal_channel_at_arrival,
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

    def test_user_metadata_excluded_from_messages(self) -> None:
        """Metadata on user message events must not leak into the
        chat-completions message list sent to the model."""
        e = _evt(1, "user", content="hello")
        e.data["metadata"] = {"run_id": "abc123"}
        msgs = build_messages([e], system_prompt=None).messages
        assert msgs[0] == {"role": "user", "content": "hello"}

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

    def test_separator_insertion_preserves_monotonicity(self) -> None:
        """Full pipeline (build_messages → tail-block → separator) must
        keep the prefix-stability invariant: output(L1) is a prefix of
        output(L2) when L1 ⊂ L2.  Pins the "insertions only at the
        volatile suffix" claim — a refactor that inserted separators
        into the cache-stable prefix would fail this."""
        bindings = [_binding("signal/test/1")]

        l1 = [
            _evt(1, "user", content="do A"),
            _evt(2, "assistant", content="done A"),
        ]
        l2 = [*l1, _evt(3, "user", content="do B")]
        l3 = [*l2, _evt(4, "assistant", content="done B")]

        out1 = _full_pipeline(l1, bindings)
        out2 = _full_pipeline(l2, bindings)
        out3 = _full_pipeline(l3, bindings)

        # The tail block mutates per step, so compare prefixes only up
        # to (but not including) the tail and any separator before it.
        def _strip_tail(msgs: list[dict]) -> list[dict]:
            for i in range(len(msgs) - 1, -1, -1):
                m = msgs[i]
                if m.get("role") == "user" and str(m.get("content", "")).startswith(
                    "━━━ Channels ━━━"
                ):
                    stop = i
                    if i > 0 and msgs[i - 1] == {"role": "assistant", "content": ""}:
                        stop = i - 1
                    return msgs[:stop]
            return msgs

        _assert_prefix(_strip_tail(out1), _strip_tail(out2))
        _assert_prefix(_strip_tail(out2), _strip_tail(out3))

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


# ─── field stripping ────────────────────────────────────────────────────────


class TestFieldStripping:
    """build_messages strips provider-specific fields from the output,
    keeping only chat-completions spec fields per role."""

    def test_assistant_reasoning_content_stripped(self) -> None:
        """Provider-specific reasoning_content is excluded from context."""
        events = [
            _evt(1, "user", content="hello"),
            _evt(2, "assistant", content="hi"),
        ]
        events[1].data["reasoning_content"] = "I think the user wants..."
        msgs = build_messages(events, system_prompt=None).messages
        assert msgs[1]["content"] == "hi"
        assert "reasoning_content" not in msgs[1]

    def test_assistant_reacting_to_stripped(self) -> None:
        """Internal reacting_to field is excluded from context output."""
        events = [
            _evt(1, "user", content="hello"),
            _evt(2, "assistant", content="hi"),
        ]
        events[1].data["reacting_to"] = 1
        msgs = build_messages(events, system_prompt=None).messages
        assert "reacting_to" not in msgs[1]

    def test_assistant_tool_calls_preserved(self) -> None:
        """tool_calls is a spec field and must survive stripping."""
        events = [
            _evt(1, "user", content="do it"),
            _evt(2, "assistant", tool_calls=[_tc("a")]),
            _evt(3, "tool", tool_call_id="a", content="done"),
        ]
        msgs = build_messages(events, system_prompt=None).messages
        assert msgs[1]["tool_calls"] == [_tc("a")]

    def test_tool_message_extra_fields_stripped(self) -> None:
        """Unknown fields on tool messages are excluded."""
        events = [
            _evt(1, "user", content="go"),
            _evt(2, "assistant", tool_calls=[_tc("a")]),
            _evt(3, "tool", tool_call_id="a", content="done"),
        ]
        events[2].data["provider_metadata"] = {"some": "thing"}
        msgs = build_messages(events, system_prompt=None).messages
        tool_msg = next(m for m in msgs if m.get("role") == "tool")
        assert "provider_metadata" not in tool_msg
        assert tool_msg["tool_call_id"] == "a"
        assert tool_msg["content"] == "done"

    def test_multiple_provider_fields_all_stripped(self) -> None:
        """All provider-specific fields are excluded, only spec fields remain."""
        events = [
            _evt(1, "user", content="think hard"),
            _evt(2, "assistant", content="here's my answer"),
        ]
        events[1].data.update(
            {
                "reasoning_content": "deep thoughts...",
                "reasoning": "step by step...",
                "reasoning_details": [{"type": "thinking", "content": "hmm"}],
                "reacting_to": 1,
                "provider_specific_id": "abc123",
            }
        )
        msgs = build_messages(events, system_prompt=None).messages
        assert set(msgs[1].keys()) == {"role", "content"}

    def test_system_prompt_clean(self) -> None:
        """System prompt message contains only spec fields."""
        events = [_evt(1, "user", content="hi")]
        msgs = build_messages(events, system_prompt="You are helpful.").messages
        assert msgs[0] == {"role": "system", "content": "You are helpful."}

    def test_stripping_does_not_mutate_event_data(self) -> None:
        """Stripping produces new dicts; original event data is unchanged."""
        events = [
            _evt(1, "user", content="hello"),
            _evt(2, "assistant", content="hi"),
        ]
        events[1].data["reasoning_content"] = "thoughts"
        build_messages(events, system_prompt=None)
        assert "reasoning_content" in events[1].data


class TestToolCallSanitization:
    """_strip_to_spec sanitizes the inner structure of tool_calls so
    malformed entries from one model don't break cross-model replay."""

    def test_valid_tool_calls_unchanged(self) -> None:
        """Well-formed tool_calls pass through identical."""
        events = [
            _evt(1, "user", content="go"),
            _evt(2, "assistant", tool_calls=[_tc("a", "bash")]),
            _evt(3, "tool", tool_call_id="a", content="done"),
        ]
        msgs = build_messages(events, system_prompt=None).messages
        assert msgs[1]["tool_calls"] == [_tc("a", "bash")]

    def test_malformed_arguments_replaced(self) -> None:
        """Control characters in function.arguments get replaced with '{}'."""
        bad_tc = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"cmd": "echo\nhello"}'},
        }
        events = [
            _evt(1, "user", content="go"),
            _evt(2, "assistant", tool_calls=[bad_tc]),
            _evt(3, "tool", tool_call_id="call_1", content="done"),
        ]
        msgs = build_messages(events, system_prompt=None).messages
        tc = msgs[1]["tool_calls"][0]
        assert tc["function"]["arguments"] == "{}"
        assert tc["function"]["name"] == "bash"
        assert tc["id"] == "call_1"

    def test_missing_function_name_defaults_empty(self) -> None:
        """Missing function.name defaults to empty string."""
        bad_tc = {"id": "call_1", "type": "function", "function": {"arguments": "{}"}}
        events = [
            _evt(1, "user", content="go"),
            _evt(2, "assistant", tool_calls=[bad_tc]),
            _evt(3, "tool", tool_call_id="call_1", content="done"),
        ]
        msgs = build_messages(events, system_prompt=None).messages
        assert msgs[1]["tool_calls"][0]["function"]["name"] == ""

    def test_missing_id_defaults_empty(self) -> None:
        """Missing tool_call id defaults to empty string."""
        bad_tc = {"type": "function", "function": {"name": "bash", "arguments": "{}"}}
        events = [
            _evt(1, "user", content="go"),
            _evt(2, "assistant", tool_calls=[bad_tc]),
        ]
        msgs = build_messages(events, system_prompt=None).messages
        assert msgs[1]["tool_calls"][0]["id"] == ""

    def test_extra_fields_stripped_from_tool_call(self) -> None:
        """Provider-specific fields inside tool_call dicts are excluded."""
        tc_with_extras = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "bash", "arguments": "{}"},
            "index": 0,
            "provider_id": "xyz",
        }
        events = [
            _evt(1, "user", content="go"),
            _evt(2, "assistant", tool_calls=[tc_with_extras]),
            _evt(3, "tool", tool_call_id="call_1", content="done"),
        ]
        msgs = build_messages(events, system_prompt=None).messages
        tc = msgs[1]["tool_calls"][0]
        assert set(tc.keys()) == {"id", "type", "function"}
        assert set(tc["function"].keys()) == {"name", "arguments"}

    def test_dict_arguments_serialized(self) -> None:
        """Arguments as a dict (from some providers) are serialized to JSON string."""
        bad_tc = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "bash", "arguments": {"command": "ls"}},
        }
        events = [
            _evt(1, "user", content="go"),
            _evt(2, "assistant", tool_calls=[bad_tc]),
            _evt(3, "tool", tool_call_id="call_1", content="done"),
        ]
        msgs = build_messages(events, system_prompt=None).messages
        args = msgs[1]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(args, str)
        assert json.loads(args) == {"command": "ls"}

    def test_missing_function_dict(self) -> None:
        """tool_call with no function dict at all gets safe defaults."""
        bad_tc = {"id": "call_1", "type": "function"}
        events = [
            _evt(1, "user", content="go"),
            _evt(2, "assistant", tool_calls=[bad_tc]),
            _evt(3, "tool", tool_call_id="call_1", content="done"),
        ]
        msgs = build_messages(events, system_prompt=None).messages
        tc = msgs[1]["tool_calls"][0]
        assert tc["function"]["name"] == ""
        assert tc["function"]["arguments"] == "{}"

    def test_sanitization_does_not_mutate_event_data(self) -> None:
        """Sanitization produces new dicts; original event data is unchanged."""
        bad_args = '{"cmd": "echo\nhello"}'
        bad_tc = {
            "id": "c1",
            "type": "function",
            "function": {"name": "bash", "arguments": bad_args},
        }
        events = [
            _evt(1, "user", content="go"),
            _evt(2, "assistant", tool_calls=[bad_tc]),
            _evt(3, "tool", tool_call_id="c1", content="done"),
        ]
        build_messages(events, system_prompt=None)
        assert events[1].data["tool_calls"][0]["function"]["arguments"] == bad_args


# ─── focal-channel rendering ────────────────────────────────────────────────


class TestFocalRendering:
    """Slice 4 of the focal-channel redesign (issue #29).

    User events are rendered differently based on ``orig_channel`` vs
    ``focal_channel_at_arrival``.  Three branches:

    * both NULL (legacy / direct non-connector message) → Phase 2
      rendering unchanged.
    * ``orig == focal_at_arrival`` → full content with the #46 metadata
      header inlined.
    * ``orig != focal_at_arrival`` OR focal NULL → short notification
      marker.
    """

    _CHAN_A = "signal/bot/alice"
    _CHAN_B = "signal/bot/bob"

    def test_legacy_null_event_phase2_rendering(self) -> None:
        """orig_channel=None: no header, no marker — same as Phase 2."""
        events = [_evt(1, "user", content="hi")]
        msg = build_messages(events, system_prompt=None).messages[0]
        assert msg["content"] == "hi"
        assert "metadata" not in msg
        assert "🔔" not in msg["content"]

    def test_focal_match_renders_full_content_with_header(self) -> None:
        md = {
            "channel": self._CHAN_A,
            "sender_uuid": "peer-uuid",
            "timestamp_ms": 1776401210703,
        }
        events = [_evt(1, "user", content="hi", metadata=md, focal_channel_at_arrival=self._CHAN_A)]
        msg = build_messages(events, system_prompt=None).messages[0]
        assert msg["content"].startswith(f"[channel={self._CHAN_A}")
        assert "sender_uuid=peer-uuid" in msg["content"]
        assert msg["content"].endswith("\nhi")
        assert "metadata" not in msg

    def test_focal_match_header_includes_chat_type_name_and_sender(self) -> None:
        md = {
            "channel": self._CHAN_A,
            "chat_type": "group",
            "chat_name": "QA",
            "sender_uuid": "u1",
            "sender_name": "Tom",
            "timestamp_ms": 1776401210703,
        }
        events = [_evt(1, "user", content="yo", metadata=md, focal_channel_at_arrival=self._CHAN_A)]
        content = build_messages(events, system_prompt=None).messages[0]["content"]
        assert "chat_type=group" in content
        assert "chat_name='QA'" in content
        assert "from=Tom" in content

    def test_focal_match_reply_to_surfaced(self) -> None:
        md = {
            "channel": self._CHAN_A,
            "reply_to": {
                "author_uuid": "bot",
                "timestamp_ms": 1776400000000,
                "text": "what I said before",
            },
        }
        events = [
            _evt(1, "user", content="reacting", metadata=md, focal_channel_at_arrival=self._CHAN_A)
        ]
        content = build_messages(events, system_prompt=None).messages[0]["content"]
        assert "\n[reply_to: author_uuid=bot · timestamp_ms=1776400000000]" in content
        assert "> what I said before" in content

    def test_focal_match_reaction_surfaced(self) -> None:
        md = {
            "channel": self._CHAN_A,
            "reaction": {
                "emoji": "👍",
                "target_author_uuid": "bot",
                "target_timestamp_ms": 1776400000000,
            },
        }
        events = [_evt(1, "user", content="", metadata=md, focal_channel_at_arrival=self._CHAN_A)]
        content = build_messages(events, system_prompt=None).messages[0]["content"]
        assert "[reaction='👍'" in content
        assert "target_author_uuid=bot" in content

    def test_focal_match_iso_timestamp(self) -> None:
        md = {"channel": self._CHAN_A, "timestamp_ms": 1776401210703}
        events = [_evt(1, "user", content="hi", metadata=md, focal_channel_at_arrival=self._CHAN_A)]
        content = build_messages(events, system_prompt=None).messages[0]["content"]
        assert "timestamp_ms=1776401210703" in content
        assert "(2026-04-17T" in content

    def test_focal_match_metadata_stripped_from_wire_message(self) -> None:
        md = {"channel": self._CHAN_A, "sender_uuid": "u1"}
        events = [_evt(1, "user", content="hi", metadata=md, focal_channel_at_arrival=self._CHAN_A)]
        msg = build_messages(events, system_prompt=None).messages[0]
        assert "metadata" not in msg
        assert set(msg.keys()) <= {"role", "content", "name"}

    def test_notification_when_orig_differs_from_focal(self) -> None:
        md = {"channel": self._CHAN_B, "sender_name": "Bob"}
        events = [
            _evt(
                1,
                "user",
                content="hey there",
                metadata=md,
                focal_channel_at_arrival=self._CHAN_A,
            )
        ]
        content = build_messages(events, system_prompt=None).messages[0]["content"]
        assert content.startswith(f"🔔 channel_id={self._CHAN_B}")
        assert "from=Bob" in content
        assert "hey there" in content

    def test_notification_when_focal_null(self) -> None:
        """Phone-down state: all inbound renders as notifications."""
        md = {"channel": self._CHAN_B, "sender_name": "Bob"}
        events = [_evt(1, "user", content="hey", metadata=md, focal_channel_at_arrival=None)]
        content = build_messages(events, system_prompt=None).messages[0]["content"]
        assert content.startswith(f"🔔 channel_id={self._CHAN_B}")

    def test_notification_omits_sender_when_absent(self) -> None:
        """No sender_name → no ``from=`` clause, just channel + preview
        on the header line.  Hint line follows unconditionally.
        """
        md = {"channel": self._CHAN_B}
        events = [
            _evt(1, "user", content="hey", metadata=md, focal_channel_at_arrival=self._CHAN_A)
        ]
        content = build_messages(events, system_prompt=None).messages[0]["content"]
        assert "from=" not in content
        header, hint = content.split("\n", 1)
        assert header == f"🔔 channel_id={self._CHAN_B} · hey"
        assert hint.startswith("(to respond, call switch_channel(channel_id=")

    def test_notification_hint_names_the_channel_id(self) -> None:
        """The ``to respond...`` hint line includes the same channel_id
        as the marker, so weaker models can copy-paste it directly into
        a ``switch_channel`` call.
        """
        md = {"channel": self._CHAN_B, "sender_name": "Bob"}
        events = [
            _evt(1, "user", content="hey", metadata=md, focal_channel_at_arrival=self._CHAN_A)
        ]
        content = build_messages(events, system_prompt=None).messages[0]["content"]
        assert f"switch_channel(channel_id='{self._CHAN_B}')" in content

    def test_notification_truncation_at_80_chars(self) -> None:
        long = "x" * 200
        md = {"channel": self._CHAN_B, "sender_name": "Bob"}
        events = [_evt(1, "user", content=long, metadata=md, focal_channel_at_arrival=self._CHAN_A)]
        content = build_messages(events, system_prompt=None).messages[0]["content"]
        # Exactly 80 x's followed by an ellipsis.
        assert "x" * 80 + "…" in content
        assert "x" * 81 not in content  # never more than 80 raw chars

    def test_notification_reaction_fallback_when_content_empty(self) -> None:
        md = {
            "channel": self._CHAN_B,
            "sender_name": "Bob",
            "reaction": {"emoji": "👍"},
        }
        events = [_evt(1, "user", content="", metadata=md, focal_channel_at_arrival=self._CHAN_A)]
        content = build_messages(events, system_prompt=None).messages[0]["content"]
        assert "reacted 👍" in content

    def test_notification_metadata_stripped_from_wire_message(self) -> None:
        md = {"channel": self._CHAN_B, "sender_name": "Bob"}
        events = [_evt(1, "user", content="hi", metadata=md, focal_channel_at_arrival=self._CHAN_A)]
        msg = build_messages(events, system_prompt=None).messages[0]
        assert "metadata" not in msg
        assert set(msg.keys()) <= {"role", "content", "name"}

    def test_switch_channel_does_not_rewrite_past_events(self) -> None:
        """Monotonicity invariant: past events' rendering is pinned by
        their own stamped fields, regardless of subsequent focal changes.
        """
        md_b = {"channel": self._CHAN_B, "sender_name": "Bob"}
        # Event arrives on B while focal=A → notification at append time.
        ev_early = _evt(
            1, "user", content="early", metadata=md_b, focal_channel_at_arrival=self._CHAN_A
        )
        # Later event on A while focal=A (post-switch simulation) → full.
        ev_late = _evt(
            2,
            "user",
            content="later",
            metadata={"channel": self._CHAN_A, "sender_name": "Alice"},
            focal_channel_at_arrival=self._CHAN_A,
        )
        msgs = build_messages([ev_early, ev_late], system_prompt=None).messages
        assert msgs[0]["content"].startswith(f"🔔 channel_id={self._CHAN_B}")
        assert msgs[1]["content"].startswith(f"[channel={self._CHAN_A}")


class TestSeparateAdjacentUserMessages:
    def test_inserts_empty_assistant_between_two_users(self) -> None:
        msgs = [
            {"role": "user", "content": "one"},
            {"role": "user", "content": "two"},
        ]
        assert separate_adjacent_user_messages(msgs) == [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "two"},
        ]

    def test_preserves_existing_alternation(self) -> None:
        msgs = [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "two"},
        ]
        assert separate_adjacent_user_messages(msgs) == msgs

    def test_tool_result_between_users_is_not_separated(self) -> None:
        """Adjacent means *consecutive same-role*. Tool results don't trigger."""
        msgs = [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "a"}]},
            {"role": "tool", "tool_call_id": "a", "content": "r"},
            {"role": "user", "content": "two"},
        ]
        assert separate_adjacent_user_messages(msgs) == msgs

    def test_three_consecutive_users_get_two_separators(self) -> None:
        msgs = [
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
            {"role": "user", "content": "c"},
        ]
        assert separate_adjacent_user_messages(msgs) == [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "b"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "c"},
        ]

    def test_empty_input_returns_empty(self) -> None:
        assert separate_adjacent_user_messages([]) == []

    def test_system_then_user_then_user_separates_only_users(self) -> None:
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "one"},
            {"role": "user", "content": "two"},
        ]
        assert separate_adjacent_user_messages(msgs) == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "two"},
        ]


class TestSeparateAdjacentUserMessagesPipeline:
    """Exercise the separator against realistic ``build_messages`` output
    (rather than synthetic dicts) so a refactor that changes the output's
    role sequence can't silently break the fix."""

    def test_inbound_then_tail_block_gets_separator(self) -> None:
        events = [_evt(1, "user", content="hello")]
        msgs = _full_pipeline(events, [_binding("signal/test/1")])

        assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
        assert msgs[1] == {"role": "assistant", "content": ""}
        assert msgs[2]["content"].startswith("━━━ Channels ━━━")

    def test_blind_spot_injection_adjacent_user_gets_separator(self) -> None:
        """``build_messages`` inlines a blind-spot tool result as a
        synthetic user message right after the horizon-setter.  When
        a real user event follows, the two land back-to-back and need
        separating."""
        events = [
            _evt(1, "user", content="run it"),
            _evt(2, "assistant", tool_calls=[_tc("t1")]),
            _evt(3, "tool", tool_call_id="t1", content="RESULT"),
            _evt(4, "assistant", content="checking..."),
            _evt(5, "user", content="anything else?"),
            _evt(6, "assistant", content="nope"),
        ]
        events[1].data["reacting_to"] = 1
        events[3].data["reacting_to"] = 1  # blind to tool at seq=3
        events[5].data["reacting_to"] = 5

        msgs = _full_pipeline(events, bindings=[])

        injection_idx = next(
            i
            for i, m in enumerate(msgs)
            if m["role"] == "user" and "RESULT" in str(m.get("content", ""))
        )
        assert msgs[injection_idx + 1] == {"role": "assistant", "content": ""}
        assert msgs[injection_idx + 2]["role"] == "user"
        assert msgs[injection_idx + 2]["content"] == "anything else?"

    def test_alternating_events_no_tail_block_no_separator(self) -> None:
        """Guards against a future change that inserts a separator when
        no adjacency exists (empty bindings → tail block is ``None``)."""
        events = [
            _evt(1, "user", content="hi"),
            _evt(2, "assistant", content="hello"),
            _evt(3, "user", content="bye"),
            _evt(4, "assistant", content="later"),
        ]
        msgs = _full_pipeline(events, bindings=[])

        assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
        assert not any(m == {"role": "assistant", "content": ""} for m in msgs)
