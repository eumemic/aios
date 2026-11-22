"""Reply-or-decline obligation, computed from a step's slate (#2092).

Mutation coverage in both directions: an agent that writes its reply as bare
text owes one; an agent that sends, reacts or calls ``no_response`` owes
nothing; and stimuli that aren't inbound channel messages open nothing.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from aios.harness.reminders import reminder_event_data
from aios.harness.reply_obligation import (
    NO_RESPONSE_TOOL_NAME,
    REPLY_NUDGE_METADATA_KEY,
    ReplyDue,
    find_reply_due,
    reply_nudge_content,
)
from aios.models.events import Event

_T0 = datetime(2026, 11, 1, tzinfo=UTC)
CHAN_A = "signal/bot/chat-a"
CHAN_B = "signal/bot/chat-b"
CONNECTOR_TOOLS = frozenset({"signal_send", "signal_react"})


def _event(seq: int, data: dict[str, Any], **extra: Any) -> Event:
    return Event(
        id=f"evt_{seq:03d}",
        session_id="sess_reply",
        seq=seq,
        kind="message",
        data=data,
        created_at=_T0 + timedelta(seconds=seq),
        **extra,
    )


def _inbound(seq: int, channel: str = CHAN_A, text: str = "hello?") -> Event:
    return _event(
        seq,
        {"role": "user", "content": text, "metadata": {"channel": channel}},
        orig_channel=channel,
        focal_channel_at_arrival=CHAN_A,
        channel=channel,
    )


def _turn(seq: int, *calls: str, reacting_to: int, focal: str | None = CHAN_A) -> Event:
    data: dict[str, Any] = {"role": "assistant", "content": "", "reacting_to": reacting_to}
    if calls:
        data["tool_calls"] = [
            {
                "id": f"c{seq}_{i}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps({})},
            }
            for i, name in enumerate(calls)
        ]
    return _event(seq, data, focal_channel_at_arrival=focal, channel=focal)


def _due(events: list[Event], reacting_to: int) -> ReplyDue | None:
    return find_reply_due(events, reacting_to=reacting_to, connector_tool_names=CONNECTOR_TOOLS)


class TestOpens:
    def test_bare_text_reply_leaves_the_inbound_owed(self) -> None:
        assert _due([_inbound(1)], reacting_to=1) == ReplyDue(
            through_seq=1, channels=(CHAN_A,), nudged=False
        )

    def test_a_non_reply_tool_does_not_discharge(self) -> None:
        events = [_inbound(1), _turn(2, "bash", reacting_to=1)]
        assert _due(events, reacting_to=3) == ReplyDue(1, (CHAN_A,), False)

    def test_channels_listed_oldest_first_once_each(self) -> None:
        events = [_inbound(1, CHAN_B), _inbound(2), _inbound(3, CHAN_B)]
        due = _due(events, reacting_to=3)
        assert due is not None
        assert due.channels == (CHAN_B, CHAN_A)
        assert due.through_seq == 3


class TestDischarges:
    def test_a_send_on_the_channel(self) -> None:
        events = [_inbound(1), _turn(2, "signal_send", reacting_to=1)]
        assert _due(events, reacting_to=3) is None

    def test_a_reaction_counts_as_a_reply(self) -> None:
        events = [_inbound(1), _turn(2, "signal_react", reacting_to=1)]
        assert _due(events, reacting_to=3) is None

    def test_no_response_declines_every_channel(self) -> None:
        events = [
            _inbound(1, CHAN_B),
            _inbound(2),
            _turn(3, NO_RESPONSE_TOOL_NAME, reacting_to=2, focal=None),
        ]
        assert _due(events, reacting_to=4) is None

    def test_one_reply_covers_a_burst(self) -> None:
        events = [_inbound(1), _inbound(2), _inbound(3), _turn(4, "signal_send", reacting_to=3)]
        assert _due(events, reacting_to=5) is None

    def test_a_send_on_one_channel_leaves_another_owed(self) -> None:
        events = [_inbound(1, CHAN_B), _inbound(2), _turn(3, "signal_send", reacting_to=2)]
        assert _due(events, reacting_to=4) == ReplyDue(1, (CHAN_B,), False)

    def test_an_inbound_the_reply_did_not_see_stays_owed(self) -> None:
        # The send reacted to seq 1; seq 2 landed during its inference.
        events = [_inbound(1), _inbound(2), _turn(3, "signal_send", reacting_to=1)]
        assert _due(events, reacting_to=4) == ReplyDue(2, (CHAN_A,), False)


class TestLegacyHistory:
    """Before #2092 a step declined by closing on stamped text. That turn still
    reads as a decline, so a session's pre-change backlog owes nothing at deploy."""

    def _stamped(self, seq: int, *, reacting_to: int, tools: bool = False) -> Event:
        data: dict[str, Any] = {
            "role": "assistant",
            "content": "INTERNAL_MONOLOGUE_NOT_SEEN_BY_USER: not addressed to me",
            "reacting_to": reacting_to,
        }
        if tools:
            data["tool_calls"] = [
                {"id": "c", "type": "function", "function": {"name": "bash", "arguments": "{}"}}
            ]
        return _event(seq, data, focal_channel_at_arrival=CHAN_A, channel=CHAN_A)

    def test_a_stamped_closing_turn_declines_what_it_saw(self) -> None:
        events = [_inbound(1, CHAN_B), _inbound(2), self._stamped(3, reacting_to=2)]
        assert _due(events, reacting_to=3) is None

    def test_a_stamped_tool_call_turn_is_not_a_decline(self) -> None:
        """The old harness stamped every text, tool-call turns included; only a
        step's closing line (no tool calls) meant "no reply"."""
        events = [_inbound(1), self._stamped(2, reacting_to=1, tools=True)]
        assert _due(events, reacting_to=3) == ReplyDue(1, (CHAN_A,), False)

    def test_an_inbound_after_the_legacy_decline_is_owed(self) -> None:
        events = [_inbound(1), self._stamped(2, reacting_to=1), _inbound(3)]
        assert _due(events, reacting_to=3) == ReplyDue(3, (CHAN_A,), False)


class TestOpensNothing:
    def test_operator_message_without_a_channel(self) -> None:
        assert _due([_event(1, {"role": "user", "content": "wake up"})], reacting_to=1) is None

    def test_reminder_rows_and_tool_results(self) -> None:
        events = [
            _event(1, reminder_event_data("channels", "━━━ Channels ━━━")),
            _turn(2, "bash", reacting_to=0),
            _event(3, {"role": "tool", "tool_call_id": "c2_0", "content": "ok"}),
        ]
        assert _due(events, reacting_to=3) is None

    def test_inbound_past_the_turns_watermark(self) -> None:
        """The harvest path's turn reacts to the park-time watermark; an inbound
        after it is the next step's business, not this turn's."""
        assert _due([_inbound(5)], reacting_to=4) is None


class TestNudgeBudget:
    def _nudge(self, seq: int, through_seq: int) -> Event:
        return _event(
            seq,
            {
                "role": "user",
                "content": "nudge",
                "metadata": {
                    REPLY_NUDGE_METADATA_KEY: {"through_seq": through_seq, "channels": [CHAN_A]}
                },
            },
        )

    def test_an_inbound_already_nudged_lapses(self) -> None:
        events = [_inbound(1), _turn(2, reacting_to=1), self._nudge(3, 1)]
        assert _due(events, reacting_to=3) == ReplyDue(1, (CHAN_A,), True)

    def test_a_newer_inbound_gets_its_own_nudge(self) -> None:
        events = [_inbound(1), _turn(2, reacting_to=1), self._nudge(3, 1), _inbound(4)]
        assert _due(events, reacting_to=4) == ReplyDue(4, (CHAN_A,), False)

    def test_the_nudge_itself_opens_nothing(self) -> None:
        events = [_inbound(1), _turn(2, "signal_send", reacting_to=1), self._nudge(3, 1)]
        assert _due(events, reacting_to=3) is None


def test_nudge_content_names_the_channels_and_both_exits() -> None:
    text = reply_nudge_content(ReplyDue(7, (CHAN_A, CHAN_B), False))
    assert f"channel_id={CHAN_A}" in text
    assert f"channel_id={CHAN_B}" in text
    assert "not sent" in text
    assert "no_response(reason)" in text
