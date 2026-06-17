"""Table-driven tests for the slice-2 Slack decision layer (no network).

Covers design §3.4 (``chat_id`` / ``event_id`` derivation, threads share
the channel session) and §3.6 (the four connector-side gates):

* self / bot-loop filter, including the **nested** ``message_changed``
  author read (a bot-authored edit is dropped);
* cross-app / cross-team filter;
* subtype filter;
* the mention-gate as a ``chat_kind`` discriminant, with the
  ``bot_thread_participant`` implicit-mention bypass; and
* the deterministic ``event_id`` (edit distinct from original).
"""

from __future__ import annotations

from typing import Any

import pytest

from aios_slack.parse import (
    GateOutcome,
    InboundMessage,
    build_inbound,
    chat_kind_of,
    extract_mentions,
    gate,
    sanitize_display_name,
)

BOT = "UBOT00000"
TEAM = "T0AAAAAAA"
APP = "A0APPAAAA"
HUMAN = "UHUMAN111"
OTHER_BOT = "B0BOTXXXX"


def _msg(
    *,
    channel: str = "C0CHAN111",
    user: str = HUMAN,
    text: str = "hello",
    ts: str = "1700000000.000100",
    channel_type: str = "channel",
    subtype: str | None = None,
    thread_ts: str | None = None,
    bot_id: str | None = None,
    team: str | None = TEAM,
    api_app_id: str | None = APP,
    parent_user_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a synthetic Slack ``message`` event payload."""
    event: dict[str, Any] = {
        "type": "message",
        "channel": channel,
        "channel_type": channel_type,
        "user": user,
        "text": text,
        "ts": ts,
    }
    if subtype is not None:
        event["subtype"] = subtype
    if thread_ts is not None:
        event["thread_ts"] = thread_ts
    if bot_id is not None:
        event["bot_id"] = bot_id
    if team is not None:
        event["team"] = team
    if api_app_id is not None:
        event["api_app_id"] = api_app_id
    if parent_user_id is not None:
        event["parent_user_id"] = parent_user_id
    if extra:
        event.update(extra)
    return event


def _message_changed(
    *,
    channel: str = "C0CHAN111",
    user: str = HUMAN,
    text: str = "edited text",
    ts: str = "1700000000.000100",
    edit_ts: str = "1700000999.000000",
    bot_id: str | None = None,
    channel_type: str = "channel",
    team: str | None = TEAM,
    api_app_id: str | None = APP,
) -> dict[str, Any]:
    """Build a synthetic ``message_changed`` event with a NESTED author."""
    nested: dict[str, Any] = {
        "type": "message",
        "user": user,
        "text": text,
        "ts": ts,
        "edited": {"user": user, "ts": edit_ts},
    }
    if bot_id is not None:
        nested["bot_id"] = bot_id
    event: dict[str, Any] = {
        "type": "message",
        "subtype": "message_changed",
        "channel": channel,
        "channel_type": channel_type,
        "message": nested,
        "previous_message": {"user": user, "text": "old text", "ts": ts},
    }
    if team is not None:
        event["team"] = team
    if api_app_id is not None:
        event["api_app_id"] = api_app_id
    return event


def _mention(uid: str) -> str:
    return f"<@{uid}>"


# ── the gate outcome table ────────────────────────────────────────────

# Each row: (id, event, kwargs-overrides, expected_outcome, expected_reason)
GATE_CASES: list[tuple[str, dict[str, Any], dict[str, Any], GateOutcome, str | None]] = [
    # ── self / bot-loop (gate 1) ──
    (
        "self_message_dropped",
        _msg(user=BOT, channel_type="im", channel="D0DMDMDM1"),
        {},
        GateOutcome.DROP,
        "self",
    ),
    (
        "bot_authored_message_dropped",
        _msg(user="USOMEBOT", bot_id=OTHER_BOT),
        {},
        GateOutcome.DROP,
        "bot",
    ),
    (
        "bot_authored_message_changed_dropped",
        _message_changed(user="USOMEBOT", bot_id=OTHER_BOT),
        {},
        GateOutcome.DROP,
        "bot",
    ),
    (
        "self_authored_message_changed_dropped",
        _message_changed(user=BOT),
        {},
        GateOutcome.DROP,
        "self",
    ),
    # ── cross-app / cross-team (gate 2) ──
    (
        "cross_team_dropped",
        _msg(team="T0WRONGGG", channel_type="im", channel="D0DMDMDM1"),
        {},
        GateOutcome.DROP,
        "cross_team",
    ),
    (
        "cross_app_dropped",
        _msg(api_app_id="A0WRONGGG", channel_type="im", channel="D0DMDMDM1"),
        {},
        GateOutcome.DROP,
        "cross_app",
    ),
    # ── message_changed diversion (human edit → non-emitting path) ──
    (
        "human_message_changed_diverted",
        _message_changed(user=HUMAN),
        {},
        GateOutcome.DIVERT_EDIT,
        "message_changed",
    ),
    # ── subtype filter (gate 3) ──
    (
        "channel_join_subtype_dropped",
        _msg(subtype="channel_join", channel_type="im", channel="D0DMDMDM1"),
        {},
        GateOutcome.DROP,
        "subtype",
    ),
    # ── mention-gate (gate 4) ──
    (
        "dm_always_passes",
        _msg(channel_type="im", channel="D0DMDMDM1", text="no mention needed"),
        {},
        GateOutcome.FORWARD,
        None,
    ),
    (
        "channel_non_mention_dropped",
        _msg(text="just chatting"),
        {},
        GateOutcome.DROP,
        "mention_required",
    ),
    (
        "channel_mention_passes",
        _msg(text=f"hey {_mention(BOT)} look"),
        {},
        GateOutcome.FORWARD,
        None,
    ),
    (
        "mpim_non_mention_dropped",
        _msg(channel_type="mpim", channel="G0MPIM111", text="group chatter"),
        {},
        GateOutcome.DROP,
        "mention_required",
    ),
    (
        "mpim_mention_passes",
        _msg(channel_type="mpim", channel="G0MPIM111", text=f"{_mention(BOT)} hi"),
        {},
        GateOutcome.FORWARD,
        None,
    ),
    (
        "thread_continuation_bypass_passes",
        _msg(text="follow-up with no mention", thread_ts="1700000000.000001"),
        {"bot_thread_ts": frozenset({"1700000000.000001"})},
        GateOutcome.FORWARD,
        None,
    ),
    (
        "reply_to_bot_bypass_passes",
        _msg(
            text="answering the bot",
            thread_ts="1700000000.000001",
            parent_user_id=BOT,
        ),
        {},
        GateOutcome.FORWARD,
        None,
    ),
    (
        "thread_reply_unrelated_thread_still_gated",
        _msg(text="unrelated thread, no mention", thread_ts="1700000000.000222"),
        {"bot_thread_ts": frozenset({"1700000000.000001"})},
        GateOutcome.DROP,
        "mention_required",
    ),
]


@pytest.mark.parametrize(
    ("event", "overrides", "expected_outcome", "expected_reason"),
    [pytest.param(ev, ov, out, rs, id=cid) for cid, ev, ov, out, rs in GATE_CASES],
)
def test_gate_outcomes(
    event: dict[str, Any],
    overrides: dict[str, Any],
    expected_outcome: GateOutcome,
    expected_reason: str | None,
) -> None:
    kwargs: dict[str, Any] = {
        "bot_user_id": BOT,
        "team_id": TEAM,
        "api_app_id": APP,
        **overrides,
    }
    decision = gate(event, **kwargs)
    assert decision.outcome is expected_outcome
    assert decision.reason == expected_reason
    if expected_outcome is GateOutcome.FORWARD:
        assert decision.message is not None
    else:
        assert decision.message is None


def test_forward_builds_normalized_inbound() -> None:
    event = _msg(text=f"hey {_mention(BOT)} please look", thread_ts="1700000000.000001")
    decision = gate(event, bot_user_id=BOT, team_id=TEAM, api_app_id=APP)
    assert decision.outcome is GateOutcome.FORWARD
    msg = decision.message
    assert msg is not None
    assert msg.chat_kind == "channel"
    assert msg.channel_id == "C0CHAN111"
    assert msg.chat_id == "C0CHAN111"  # bare channel id, not channel:thread
    assert msg.thread_ts == "1700000000.000001"
    assert msg.sender_id == HUMAN
    assert msg.mentions == (BOT,)
    assert msg.edited is False
    assert msg.edit_ts is None


def test_mention_gate_drop_records_pending() -> None:
    decision = gate(_msg(text="no mention"), bot_user_id=BOT, team_id=TEAM, api_app_id=APP)
    assert decision.outcome is GateOutcome.DROP
    assert decision.reason == "mention_required"
    assert decision.record_pending is True


# ── chat_id / event_id derivation (§3.4) ──────────────────────────────


def test_event_id_message_is_channel_and_ts() -> None:
    msg = build_inbound(_msg(ts="1700000000.000100"))
    assert msg.event_id == "slack-C0CHAN111-1700000000.000100"


def test_event_id_edit_distinct_from_original() -> None:
    original = build_inbound(_msg(ts="1700000000.000100"))
    edit_event = _message_changed(ts="1700000000.000100", edit_ts="1700000999.000000")
    edited = build_inbound(edit_event)
    assert edited.edited is True
    assert edited.edit_ts == "1700000999.000000"
    assert edited.event_id == "slack-C0CHAN111-1700000000.000100-e1700000999.000000"
    # The load-bearing property: an edit does NOT dedup-collide with the
    # original on the connector_inbound_acks PK.
    assert edited.event_id != original.event_id


def test_thread_and_toplevel_share_chat_id() -> None:
    top = build_inbound(_msg(channel="C0BUSY999", ts="1700000000.000100"))
    reply = build_inbound(
        _msg(channel="C0BUSY999", ts="1700000000.000200", thread_ts="1700000000.000100")
    )
    # Threads share the channel session: bare channel id for both.
    assert top.chat_id == reply.chat_id == "C0BUSY999"
    assert top.thread_ts is None
    assert reply.thread_ts == "1700000000.000100"


def test_thread_root_surfaces_as_toplevel() -> None:
    # thread_ts == own ts is a thread root, not a reply.
    msg = build_inbound(_msg(ts="1700000000.000100", thread_ts="1700000000.000100"))
    assert msg.thread_ts is None


# ── helpers ───────────────────────────────────────────────────────────


def test_chat_kind_from_channel_type() -> None:
    assert chat_kind_of(_msg(channel_type="im", channel="D0X")) == "im"
    assert chat_kind_of(_msg(channel_type="mpim", channel="G0X")) == "mpim"
    assert chat_kind_of(_msg(channel_type="channel", channel="C0X")) == "channel"
    assert chat_kind_of(_msg(channel_type="group", channel="G0X")) == "group"


def test_chat_kind_falls_back_to_id_prefix() -> None:
    no_type = {"type": "message", "channel": "D0DIRECT1", "user": HUMAN, "text": "hi", "ts": "1"}
    assert chat_kind_of(no_type) == "im"


def test_extract_mentions_handles_labels_and_order() -> None:
    text = f"{_mention(HUMAN)} and {_mention(BOT)}|botlabel removed"
    assert extract_mentions(f"{_mention(HUMAN)} and <@{BOT}|botlabel> too") == (HUMAN, BOT)
    assert extract_mentions(text)[0] == HUMAN


@pytest.mark.parametrize(
    ("raw", "fallback", "expected"),
    [
        ("Alice", "UID", "Alice"),
        ("multi\nline\ninjected", "UID", "multi line injected"),
        ("  spaced  out \t name ", "UID", "spaced out name"),
        ("", "UID", "UID"),
        (None, "UID", "UID"),
        ("\n\n\n", "UID", "UID"),
        ("x" * 500, "UID", "x" * 256),
    ],
)
def test_sanitize_display_name(raw: str | None, fallback: str, expected: str) -> None:
    assert sanitize_display_name(raw, fallback) == expected


def test_inbound_message_is_frozen() -> None:
    msg = build_inbound(_msg())
    assert isinstance(msg, InboundMessage)
    with pytest.raises((AttributeError, Exception)):
        msg.text = "mutated"  # type: ignore[misc]
