"""Unit tests for the switch_channel recap renderer (issue #52 fixes).

The recap builder is exposed as
:func:`aios.tools.switch_channel.render_reorient_block` — a pure
function over the session's message events.  These tests construct
synthetic :class:`~aios.models.events.Event` lists and exercise the
rendering behavior without touching the database.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from aios.harness.channels import MONOLOGUE_PREFIX, SWITCH_CHANNEL_METADATA_KEY
from aios.models.events import Event
from aios.tools.switch_channel import (
    RE_ORIENT_FLOOR_N,
    render_reorient_block,
)

CHAN_A = "signal/acct/chat-a"
CHAN_B = "signal/acct/chat-b"


def _user(
    seq: int,
    *,
    channel: str,
    content: str = "hi",
    sender: str = "Peer",
) -> Event:
    """Inbound user event with metadata header fields populated."""
    return Event(
        id=f"evt_{seq:04d}",
        session_id="sess_test",
        seq=seq,
        kind="message",
        data={
            "role": "user",
            "content": content,
            "metadata": {
                "channel": channel,
                "sender_name": sender,
                "timestamp_ms": 1000 + seq,
            },
        },
        cumulative_tokens=None,
        created_at=datetime(2026, 4, 20, tzinfo=UTC),
        orig_channel=channel,
        focal_channel_at_arrival=channel,  # full-content render path
        channel=channel,
    )


def _assistant(
    seq: int,
    *,
    focal: str | None,
    text: str = "ok",
    with_monologue_prefix: bool = True,
    tool_calls: list[dict[str, Any]] | None = None,
) -> Event:
    """Assistant event stamped with focal_channel_at_arrival = focal.

    ``channel`` equals ``focal`` because that's how append_event stamps
    assistant rows (their "belongs to" is the live focal at stamp time).
    A NULL focal produces channel=NULL — "phone down" assistant output
    that belongs to no channel.
    """
    content = (MONOLOGUE_PREFIX if with_monologue_prefix else "") + text
    data: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        data["tool_calls"] = tool_calls
    return Event(
        id=f"evt_{seq:04d}",
        session_id="sess_test",
        seq=seq,
        kind="message",
        data=data,
        cumulative_tokens=None,
        created_at=datetime(2026, 4, 20, tzinfo=UTC),
        orig_channel=None,
        focal_channel_at_arrival=focal,
        channel=focal,
    )


def _tool(
    seq: int,
    *,
    tool_call_id: str,
    channel: str | None,
    content: str = "result",
    name: str | None = None,
    marker_target: str | None = "__omit__",
    marker_success: bool = True,
) -> Event:
    """Tool result event stamped with channel = parent's focal.

    ``marker_target="__omit__"`` (default) means no switch_channel
    metadata marker is attached.  Pass a real target (or ``None``) to
    simulate a switch_channel tool result.
    """
    data: dict[str, Any] = {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": content,
    }
    if name is not None:
        data["name"] = name
    if marker_target != "__omit__":
        data["metadata"] = {
            SWITCH_CHANNEL_METADATA_KEY: {
                "target": marker_target,
                "success": marker_success,
            }
        }
    return Event(
        id=f"evt_{seq:04d}",
        session_id="sess_test",
        seq=seq,
        kind="message",
        data=data,
        cumulative_tokens=None,
        created_at=datetime(2026, 4, 20, tzinfo=UTC),
        orig_channel=None,
        focal_channel_at_arrival=None,
        channel=channel,
    )


def _assistant_with_tool_call(
    seq: int,
    *,
    focal: str | None,
    tool_call_id: str,
    name: str,
    arguments: dict[str, Any] | None = None,
) -> Event:
    """Assistant event carrying exactly one tool_call."""
    import json as _json

    return _assistant(
        seq,
        focal=focal,
        text="",
        with_monologue_prefix=False,
        tool_calls=[
            {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": _json.dumps(arguments or {}),
                },
            }
        ],
    )


class TestRecapFiltering:
    def test_empty_channel_returns_fallback(self) -> None:
        events: list[Event] = [_user(1, channel=CHAN_B, content="other")]
        out = render_reorient_block(events, CHAN_A)
        assert "no prior messages on this channel" in out
        assert CHAN_A in out

    def test_includes_peer_user_events(self) -> None:
        events = [_user(1, channel=CHAN_A, content="hello", sender="Alice")]
        out = render_reorient_block(events, CHAN_A)
        assert "hello" in out
        assert "Alice" in out

    def test_includes_agent_tool_calls(self) -> None:
        """An assistant turn on the target channel surfaces via its
        tool_calls — that's where the load-bearing content (what got
        sent to the peer) lives.  Bare assistant text is monologue and
        is dropped.
        """
        events: list[Event] = [
            _user(1, channel=CHAN_A, content="peer message"),
            _assistant_with_tool_call(
                2,
                focal=CHAN_A,
                tool_call_id="call_send",
                name="signal_send",
                arguments={"text": "on it"},
            ),
        ]
        out = render_reorient_block(events, CHAN_A)
        assert "peer message" in out
        assert "signal_send" in out
        assert '"text": "on it"' in out

    def test_excludes_other_channel_tool_calls(self) -> None:
        """Assistant events emitted while focal=B must NOT appear in A's
        recap — cross-channel-leakage guard.
        """
        events: list[Event] = [
            _user(1, channel=CHAN_A, content="peer on A"),
            _assistant_with_tool_call(
                2,
                focal=CHAN_B,
                tool_call_id="call_other",
                name="signal_send",
                arguments={"text": "reply on B"},
            ),
        ]
        out = render_reorient_block(events, CHAN_A)
        assert "peer on A" in out
        assert "reply on B" not in out

    def test_excludes_phone_down_assistant_events(self) -> None:
        """Assistant events emitted while focal=None (channel=None) have
        no channel membership and are excluded from every recap.
        """
        events: list[Event] = [
            _user(1, channel=CHAN_A, content="peer"),
            _assistant_with_tool_call(
                2,
                focal=None,
                tool_call_id="call_pd",
                name="signal_send",
                arguments={"text": "phone-down send"},
            ),
        ]
        out = render_reorient_block(events, CHAN_A)
        assert "peer" in out
        assert "phone-down send" not in out

    def test_includes_tool_results_via_parent_focal(self) -> None:
        """A tool result whose parent assistant had focal=target belongs
        to target even if the live focal has since changed.  The recap
        filter keys off the stamped channel (parent's focal), not the
        tool result's live focal_at_arrival.
        """
        events: list[Event] = [
            _user(1, channel=CHAN_A, content="peer"),
            _assistant_with_tool_call(2, focal=CHAN_A, tool_call_id="call_1", name="bash"),
            # Parent assistant stamped focal=A; tool result is marked
            # channel=A by the append-time derivation even if the live
            # focal has moved by the time the result arrives.
            _tool(3, tool_call_id="call_1", channel=CHAN_A, content="bash output"),
        ]
        out = render_reorient_block(events, CHAN_A)
        assert "bash output" in out

    def test_excludes_switch_channel_tool_results(self) -> None:
        """Prior switch_channel tool results must NOT be pulled into a
        later recap — that was the recursive-embedding bug.
        """
        events: list[Event] = [
            _user(1, channel=CHAN_A, content="peer"),
            _assistant_with_tool_call(
                2,
                focal=CHAN_A,
                tool_call_id="call_sw",
                name="switch_channel",
                arguments={"target": CHAN_A},
            ),
            _tool(
                3,
                tool_call_id="call_sw",
                channel=CHAN_A,
                content="(prior recap content we must not re-embed)",
                marker_target=CHAN_A,
            ),
        ]
        out = render_reorient_block(events, CHAN_A)
        assert "prior recap content we must not re-embed" not in out
        assert "peer" in out


class TestRecapRendering:
    def test_drops_pure_text_assistant_events(self) -> None:
        """Assistant events with no tool_calls are dropped entirely —
        they're pure internal monologue that the peer never saw.  When
        no events carry channel-bearing content, the recap falls back
        to the empty-channel line.
        """
        events = [_assistant(1, focal=CHAN_A, text="hello world")]
        out = render_reorient_block(events, CHAN_A)
        assert MONOLOGUE_PREFIX not in out
        assert "hello world" not in out
        assert "no prior messages on this channel" in out

    def test_drops_assistant_text_even_when_tool_calls_present(self) -> None:
        """An assistant turn that both monologues AND invokes tool_calls
        renders only the tool_calls.  The text is monologue regardless
        of whether a send also happened.
        """
        events: list[Event] = [
            _user(1, channel=CHAN_A, content="peer"),
            _assistant_with_tool_call(
                2,
                focal=CHAN_A,
                tool_call_id="call_send",
                name="signal_send",
                arguments={"text": "real reply"},
            ),
        ]
        # Splice in text alongside the tool_calls to simulate a turn
        # that both monologues and sends.
        events[1].data["content"] = MONOLOGUE_PREFIX + "thinking out loud"
        out = render_reorient_block(events, CHAN_A)
        assert "thinking out loud" not in out
        assert MONOLOGUE_PREFIX not in out
        assert '"text": "real reply"' in out

    def test_tool_call_arguments_are_rendered_verbatim(self) -> None:
        """The tool_call's ``function.arguments`` JSON string is emitted
        as-is so the agent can read any sent content (e.g. signal_send's
        ``text`` arg) directly without per-tool classification.  Uses
        second-person framing (``[you called: ...]``) because the recap
        is rendered back to the agent that made the calls.
        """
        events: list[Event] = [
            _assistant_with_tool_call(
                1,
                focal=CHAN_A,
                tool_call_id="call_1",
                name="signal_send",
                arguments={"text": "hello peer", "quote_id": "abc"},
            ),
        ]
        out = render_reorient_block(events, CHAN_A)
        assert "[you called: signal_send(" in out
        assert '"text": "hello peer"' in out
        assert '"quote_id": "abc"' in out

    def test_multiple_tool_calls_in_one_turn_are_joined(self) -> None:
        """A single assistant event carrying multiple tool_calls renders
        them comma-separated inside a single ``[you called: ...]`` line.
        """
        import json as _json

        asst = _assistant(
            1,
            focal=CHAN_A,
            text="",
            with_monologue_prefix=False,
            tool_calls=[
                {
                    "id": "call_a",
                    "type": "function",
                    "function": {
                        "name": "signal_send",
                        "arguments": _json.dumps({"text": "first"}),
                    },
                },
                {
                    "id": "call_b",
                    "type": "function",
                    "function": {
                        "name": "signal_send",
                        "arguments": _json.dumps({"text": "second"}),
                    },
                },
                {
                    "id": "call_c",
                    "type": "function",
                    "function": {"name": "bash", "arguments": _json.dumps({"command": "ls"})},
                },
            ],
        )
        out = render_reorient_block([asst], CHAN_A)
        # All three calls appear in the recap.
        assert '"text": "first"' in out
        assert '"text": "second"' in out
        assert '"command": "ls"' in out
        # They share a single [you called: ...] wrapper, comma-joined.
        assert out.count("[you called:") == 1
        assert "signal_send(" in out
        assert "bash(" in out

    def test_fences_top_and_bottom(self) -> None:
        events = [_user(1, channel=CHAN_A, content="msg")]
        out = render_reorient_block(events, CHAN_A)
        assert out.startswith(f"━━━ Recap: recent messages on {CHAN_A} ━━━")
        assert out.rstrip().endswith("━━━ End recap ━━━")

    def test_body_lines_are_blockquoted(self) -> None:
        events = [_user(1, channel=CHAN_A, content="msg")]
        out = render_reorient_block(events, CHAN_A)
        # Body lines within the fences all start with "> " (or are a
        # bare ">" for blank spacing lines between blocks).
        for line in out.split("\n"):
            if line.startswith("━━━"):
                continue
            assert line == "" or line.startswith(">"), f"unquoted line: {line!r}"

    def test_escapes_leading_gt_in_quoted_lines(self) -> None:
        """A peer message whose body starts with ``>`` renders as
        ``> \\>...`` in the recap, never ``> >...`` (which would parse
        as nested markdown quote).
        """
        events = [_user(1, channel=CHAN_A, content="> quoted reply")]
        out = render_reorient_block(events, CHAN_A)
        assert "> \\> quoted reply" in out
        assert "> > quoted reply" not in out

    def test_respects_floor_on_quiet_channel(self) -> None:
        """Fewer than FLOOR_N messages → show them all, no padding."""
        events = [_user(i, channel=CHAN_A, content=f"m{i}") for i in range(1, 3)]
        out = render_reorient_block(events, CHAN_A)
        assert "m1" in out
        assert "m2" in out

    def test_includes_all_unread_when_over_floor(self) -> None:
        """Unread > FLOOR_N → all unread included, not clamped to FLOOR_N."""
        n = RE_ORIENT_FLOOR_N + 5
        # Seqs 1..n; last_seen starts at 0, so every event has seq >
        # last_seen and counts as unread.  Also clear focal_at_arrival
        # so the derivation doesn't anchor the watermark mid-stream.
        events: list[Event] = [
            _user(i, channel=CHAN_A, content=f"m{i:02d}") for i in range(1, n + 1)
        ]
        for e in events:
            e.focal_channel_at_arrival = None
        out = render_reorient_block(events, CHAN_A)
        for i in range(1, n + 1):
            assert f"m{i:02d}" in out
