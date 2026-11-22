"""Reply-or-decline obligation, computed from a step's slate (#2092).

Mutation coverage in both directions: an agent that writes its reply as bare
text owes one; an agent that sends, reacts or calls ``no_response`` owes
nothing; stimuli that aren't inbound channel messages open nothing. And each
inbound is judged once: one nudge, one lapse, then closed — so history from
before the protocol (steps that closed on a line of text) owes nothing.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from aios.harness.channels import SWITCH_CHANNEL_METADATA_KEY
from aios.harness.reminders import reminder_event_data
from aios.harness.reply_obligation import (
    CONNECTOR_CALLS_KEY,
    NO_RESPONSE_TOOL_NAME,
    REPLY_NUDGE_METADATA_KEY,
    ReplyDue,
    ReplyOwed,
    find_reply_due,
    reply_nudge_content,
    tag_connector_calls,
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


def _inbound(seq: int, channel: str = CHAN_A) -> Event:
    return _event(
        seq,
        {"role": "user", "content": "hello?", "metadata": {"channel": channel}},
        orig_channel=channel,
        focal_channel_at_arrival=CHAN_A,
        channel=channel,
    )


def _closing(seq: int, *, reacting_to: int, text: str = "bare text") -> Event:
    """A turn without tool calls: it ends a step."""
    return _event(
        seq,
        {"role": "assistant", "content": text, "reacting_to": reacting_to},
        focal_channel_at_arrival=CHAN_A,
        channel=CHAN_A,
    )


def _calls(
    seq: int, *names: str, reacting_to: int, focal: str | None = CHAN_A
) -> tuple[Event, list[str]]:
    """A tool-calling turn, and its call ids (results are added separately)."""
    ids = [f"c{seq}_{i}" for i in range(len(names))]
    data = {
        "role": "assistant",
        "content": "",
        "reacting_to": reacting_to,
        "tool_calls": [
            {"id": tcid, "type": "function", "function": {"name": name, "arguments": "{}"}}
            for tcid, name in zip(ids, names, strict=True)
        ],
    }
    return _event(seq, data, focal_channel_at_arrival=focal, channel=focal), ids


def _result(
    seq: int, tcid: str, *, error: bool = False, metadata: dict[str, Any] | None = None
) -> Event:
    data: dict[str, Any] = {"role": "tool", "tool_call_id": tcid, "content": "ok"}
    if error:
        data["is_error"] = True
    if metadata:
        data["metadata"] = metadata
    return _event(seq, data)


def _turn(seq: int, *names: str, reacting_to: int, focal: str | None = CHAN_A) -> list[Event]:
    """A tool-calling turn plus a successful result for each call."""
    turn, ids = _calls(seq, *names, reacting_to=reacting_to, focal=focal)
    return [turn, *(_result(seq + 1 + i, tcid) for i, tcid in enumerate(ids))]


def _nudge(seq: int, through_seq: int, channel: str = CHAN_A) -> Event:
    return _event(
        seq,
        {
            "role": "user",
            "content": "nudge",
            "metadata": {
                REPLY_NUDGE_METADATA_KEY: {"through_seq": through_seq, "channels": [channel]}
            },
        },
    )


def _due(events: list[Event], reacting_to: int) -> ReplyDue | None:
    return find_reply_due(events, reacting_to=reacting_to, connector_tool_names=CONNECTOR_TOOLS)


def _nudge_due(through_seq: int, *channels: str) -> ReplyDue:
    return ReplyDue(nudge=ReplyOwed(through_seq, channels or (CHAN_A,)), lapse=None)


def _lapse_due(through_seq: int, *channels: str) -> ReplyDue:
    return ReplyDue(nudge=None, lapse=ReplyOwed(through_seq, channels or (CHAN_A,)))


class TestOpens:
    def test_bare_text_reply_leaves_the_inbound_owed(self) -> None:
        assert _due([_inbound(1)], reacting_to=1) == _nudge_due(1)

    def test_a_non_reply_tool_does_not_discharge(self) -> None:
        assert _due([_inbound(1), *_turn(2, "bash", reacting_to=1)], reacting_to=3) == _nudge_due(1)

    def test_channels_listed_oldest_first_once_each(self) -> None:
        events = [_inbound(1, CHAN_B), _inbound(2), _inbound(3, CHAN_B)]
        assert _due(events, reacting_to=3) == _nudge_due(3, CHAN_B, CHAN_A)


class TestDischarges:
    def test_a_send_on_the_channel(self) -> None:
        assert _due([_inbound(1), *_turn(2, "signal_send", reacting_to=1)], reacting_to=3) is None

    def test_a_reaction_counts_as_a_reply(self) -> None:
        assert _due([_inbound(1), *_turn(2, "signal_react", reacting_to=1)], reacting_to=3) is None

    def test_no_response_declines_every_channel(self) -> None:
        events = [
            _inbound(1, CHAN_B),
            _inbound(2),
            *_turn(3, NO_RESPONSE_TOOL_NAME, reacting_to=2, focal=None),
        ]
        assert _due(events, reacting_to=4) is None

    def test_one_reply_covers_a_burst(self) -> None:
        events = [_inbound(1), _inbound(2), _inbound(3), *_turn(4, "signal_send", reacting_to=3)]
        assert _due(events, reacting_to=5) is None

    def test_a_send_on_one_channel_leaves_another_owed(self) -> None:
        events = [_inbound(1, CHAN_B), _inbound(2), *_turn(3, "signal_send", reacting_to=2)]
        assert _due(events, reacting_to=4) == _nudge_due(1, CHAN_B)

    def test_an_inbound_the_reply_did_not_see_stays_owed(self) -> None:
        # The send reacted to seq 1; seq 2 landed during its inference.
        events = [_inbound(1), _inbound(2), *_turn(3, "signal_send", reacting_to=1)]
        assert _due(events, reacting_to=4) == _nudge_due(2)

    def test_a_failed_send_delivered_nothing(self) -> None:
        turn, (tcid,) = _calls(2, "signal_send", reacting_to=1)
        events = [_inbound(1), turn, _result(3, tcid, error=True)]
        assert _due(events, reacting_to=3) == _nudge_due(1)

    def test_a_send_whose_result_is_not_back_does_not_count_yet(self) -> None:
        turn, _ids = _calls(2, "signal_send", reacting_to=1)
        assert _due([_inbound(1), turn], reacting_to=1) == _nudge_due(1)

    def test_a_send_beside_a_switch_credits_the_switch_target(self) -> None:
        """Focal is read when the connector picks the call up, so a send emitted
        with a switch may land on either channel: both count as replied."""
        turn, (switch_id, send_id) = _calls(3, "switch_channel", "signal_send", reacting_to=2)
        events = [
            _inbound(1, CHAN_B),
            _inbound(2),
            turn,
            _result(
                4,
                switch_id,
                metadata={SWITCH_CHANNEL_METADATA_KEY: {"target": CHAN_B, "success": True}},
            ),
            _result(5, send_id),
        ]
        assert _due(events, reacting_to=5) is None


class TestConnectorCallsAreClassifiedWhenMade:
    """A turn records which of its calls went to a connector, so a reply made
    while ``signal_send`` was offered still counts after the connector set changes
    (a connection detached, swapped, or attenuated before the judgment)."""

    def _tagged_send(self, seq: int, *, reacting_to: int, tagged: list[str]) -> list[Event]:
        turn, (tcid,) = _calls(seq, "signal_send", reacting_to=reacting_to)
        turn.data[CONNECTOR_CALLS_KEY] = [tcid] if tagged == ["self"] else tagged
        return [turn, _result(seq + 1, tcid)]

    def test_a_tagged_reply_counts_whatever_is_offered_now(self) -> None:
        events = [_inbound(1), *self._tagged_send(2, reacting_to=1, tagged=["self"])]
        assert find_reply_due(events, reacting_to=3, connector_tool_names=frozenset()) is None

    def test_the_tag_is_authoritative_over_this_steps_tool_set(self) -> None:
        # Not a connector call when it was made, whatever the name means now.
        events = [_inbound(1), *self._tagged_send(2, reacting_to=1, tagged=[])]
        assert _due(events, reacting_to=3) == _nudge_due(1)

    def test_tagging_marks_only_connector_calls(self) -> None:
        turn, ids = _calls(1, "bash", "signal_send", NO_RESPONSE_TOOL_NAME, reacting_to=0)
        tag_connector_calls(turn.data, CONNECTOR_TOOLS)
        assert turn.data[CONNECTOR_CALLS_KEY] == [ids[1]]

    def test_a_turn_without_connector_calls_gets_no_tag(self) -> None:
        turn, _ids = _calls(1, "bash", reacting_to=0)
        tag_connector_calls(turn.data, CONNECTOR_TOOLS)
        assert CONNECTOR_CALLS_KEY not in turn.data


class TestOpensNothing:
    def test_operator_message_without_a_channel(self) -> None:
        assert _due([_event(1, {"role": "user", "content": "wake up"})], reacting_to=1) is None

    def test_reminder_rows_and_tool_results(self) -> None:
        events = [
            _event(1, reminder_event_data("channels", "━━━ Channels ━━━")),
            *_turn(2, "bash", reacting_to=0),
        ]
        assert _due(events, reacting_to=3) is None

    def test_inbound_past_the_turns_watermark(self) -> None:
        """The harvest path's turn reacts to the park-time watermark; an inbound
        after it is the next step's business, not this turn's."""
        assert _due([_inbound(5)], reacting_to=4) is None


class TestJudgedOnce:
    def test_nudge_then_lapse_then_closed(self) -> None:
        log = [_inbound(1), _closing(2, reacting_to=1), _nudge(3, 1)]
        assert _due(log, reacting_to=3) == _lapse_due(1)
        log += [_closing(4, reacting_to=3)]  # the lapse was recorded at this turn
        assert _due(log, reacting_to=4) is None
        log += [_event(5, {"role": "user", "content": "cron tick"}), _closing(6, reacting_to=5)]
        assert _due(log, reacting_to=5) is None

    def test_a_newer_inbound_gets_its_own_nudge_while_the_older_lapses(self) -> None:
        log = [_inbound(1), _closing(2, reacting_to=1), _nudge(3, 1), _inbound(4)]
        assert _due(log, reacting_to=4) == ReplyDue(
            nudge=ReplyOwed(4, (CHAN_A,)), lapse=ReplyOwed(1, (CHAN_A,))
        )

    def test_a_closing_turn_that_drew_no_nudge_closes_the_inbound(self) -> None:
        """A closing turn that couldn't idle the session (a stimulus arrived during
        its inference) writes no nudge; the inbound closes rather than being
        chased later — the net errs toward silence, never a second reply."""
        log = [_inbound(1), _closing(2, reacting_to=1), _event(3, {"role": "user", "content": "x"})]
        assert _due(log, reacting_to=3) is None

    def test_a_closing_turn_that_never_saw_the_inbound_is_not_its_judgment(self) -> None:
        """Two quick messages: the second lands while the closing turn answering the
        first is in inference (``reacting_to`` 1). That turn never saw it, so the
        next closing turn — the bare-text answer to it — is its judgment point."""
        log = [
            _inbound(1),
            *_turn(2, "signal_send", reacting_to=1),
            _inbound(4),
            _closing(5, reacting_to=3),
        ]
        assert _due(log, reacting_to=4) == _nudge_due(4)

    def test_a_later_inbounds_nudge_is_not_mistaken_for_an_earlier_ones(self) -> None:
        log = [
            _inbound(1),
            _closing(2, reacting_to=1),  # judged here without a nudge: closed
            _inbound(3),
            _closing(4, reacting_to=3),
            _nudge(5, 3),  # covers seq 1 by through_seq, but was written for seq 3
        ]
        assert _due(log, reacting_to=5) == _lapse_due(3)

    def test_the_nudge_itself_opens_nothing(self) -> None:
        log = [_inbound(1), *_turn(2, "signal_send", reacting_to=1), _nudge(4, 1)]
        assert _due(log, reacting_to=4) is None


class TestHistoryFromBeforeTheProtocol:
    """Before #2092 every step closed on a line of text — stamped, or empty when
    the model wrote nothing. Either way that closing turn drew no nudge, so a
    session's backlog owes nothing at deploy."""

    def test_a_stamped_closing_line(self) -> None:
        log = [
            _inbound(1, CHAN_B),
            _inbound(2),
            _closing(3, reacting_to=2, text="INTERNAL_MONOLOGUE_NOT_SEEN_BY_USER: not for me"),
        ]
        assert _due(log, reacting_to=4) is None

    def test_an_empty_closing_turn(self) -> None:
        assert _due([_inbound(1), _closing(2, reacting_to=1, text="")], reacting_to=3) is None


def test_nudge_content_names_the_channels_and_both_exits() -> None:
    text = reply_nudge_content(ReplyOwed(7, (CHAN_A, CHAN_B)))
    assert f"channel_id={CHAN_A}" in text
    assert f"channel_id={CHAN_B}" in text
    assert "not sent" in text
    assert "no_response(reason)" in text
