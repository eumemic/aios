"""Tests for parse.py — daemon ``message`` notification → InboundMessage."""

from __future__ import annotations

from aios_whatsapp.parse import (
    InboundAttachment,
    InboundMessage,
    InboundReaction,
    parse_message,
)

from .conftest import GROUP_JID, PEER_JID, dm_payload, group_payload


def test_parse_message_dm_round_trip() -> None:
    msg = parse_message(dm_payload())
    assert msg == InboundMessage(
        chat_type="dm",
        chat_jid=PEER_JID,
        chat_name=None,
        sender_jid=PEER_JID,
        sender_name="Alice",
        message_id="3EB0BB36C97D4F8C29A4",
        timestamp_ms=1700000000000,
        text="hello bot",
    )


def test_parse_message_group_round_trip() -> None:
    msg = parse_message(group_payload())
    assert msg is not None
    assert msg.chat_type == "group"
    assert msg.chat_jid == GROUP_JID
    assert msg.chat_name == "Test Group"
    assert msg.text == "group hello"


def test_parse_message_drops_self_echo() -> None:
    assert parse_message(dm_payload(is_self=True)) is None


def test_parse_message_drops_empty_text() -> None:
    assert parse_message(dm_payload(text="")) is None
    assert parse_message(dm_payload(text=None)) is None


def test_parse_message_drops_broadcast() -> None:
    assert parse_message(dm_payload(chat_type="broadcast", chat_jid="12345@broadcast")) is None


def test_parse_message_drops_newsletter() -> None:
    assert parse_message(dm_payload(chat_type="newsletter", chat_jid="99999@newsletter")) is None


def test_parse_message_drops_missing_required_field() -> None:
    p = dm_payload()
    del p["id"]
    assert parse_message(p) is None
    p = dm_payload()
    del p["timestamp_ms"]
    assert parse_message(p) is None


def test_parse_message_tolerates_missing_chat_name_for_dm() -> None:
    p = dm_payload()
    del p["chat_name"]
    msg = parse_message(p)
    assert msg is not None
    assert msg.chat_name is None


def test_parse_message_tolerates_missing_push_name() -> None:
    p = dm_payload()
    del p["from_push_name"]
    msg = parse_message(p)
    assert msg is not None
    assert msg.sender_name is None


def test_parse_message_carries_attachments() -> None:
    p = dm_payload(text="check this out")
    p["attachments"] = [
        {"path": "/tmp/media/3EB0X_photo.jpg", "mimetype": "image/jpeg", "filename": "photo.jpg"}
    ]
    msg = parse_message(p)
    assert msg is not None
    assert msg.attachments == (
        InboundAttachment(
            host_path="/tmp/media/3EB0X_photo.jpg",
            filename="photo.jpg",
            content_type="image/jpeg",
        ),
    )
    assert msg.text == "check this out"


def test_parse_message_keeps_attachment_only_message() -> None:
    # Previously, an empty-text payload was dropped wholesale.  PR 5
    # inverts that: an image-only message is signal worth surfacing —
    # the harness can still render "received an image" via the
    # attachment metadata.
    p = dm_payload(text="")
    p["attachments"] = [
        {"path": "/tmp/m/3EB0Y.jpg", "mimetype": "image/jpeg", "filename": "photo.jpg"}
    ]
    msg = parse_message(p)
    assert msg is not None
    assert msg.text == ""
    assert len(msg.attachments) == 1


def test_parse_message_drops_malformed_attachment_entries() -> None:
    # The daemon emits only complete entries on success, but defending
    # against partial dicts protects the connector from a future
    # daemon-side regression silently dropping a path field.
    p = dm_payload(text="caption")
    p["attachments"] = [
        {"path": "/tmp/m/a.jpg", "mimetype": "image/jpeg", "filename": "a.jpg"},
        {"path": "", "mimetype": "image/jpeg", "filename": "b.jpg"},  # bad path
        {"path": "/tmp/m/c.jpg", "mimetype": "", "filename": "c.jpg"},  # bad mime
        "not-a-dict",  # bad type entirely
    ]
    msg = parse_message(p)
    assert msg is not None
    assert len(msg.attachments) == 1
    assert msg.attachments[0].filename == "a.jpg"


def test_parse_message_sticker_emoji_kept_without_text() -> None:
    p = dm_payload(text="")
    p["sticker_emoji"] = "🎉"
    msg = parse_message(p)
    assert msg is not None
    assert msg.sticker_emoji == "🎉"
    assert msg.attachments == ()


def test_parse_message_drops_when_no_signal_at_all() -> None:
    # No text, no attachments, no sticker, no reaction — nothing for
    # the model to act on, so silently drop.
    p = dm_payload(text="")
    assert parse_message(p) is None


def test_parse_message_carries_reaction() -> None:
    p = dm_payload(text="")
    p["reaction"] = {"emoji": "👍", "target_message_id": "3EB0ORIGINAL"}
    msg = parse_message(p)
    assert msg is not None
    assert msg.reaction == InboundReaction(emoji="👍", target_message_id="3EB0ORIGINAL")
    assert msg.text == ""


def test_parse_message_reaction_removal_keeps_event() -> None:
    # Empty emoji = peer cleared their prior reaction.  Surface
    # explicitly so the model can update its mental model.
    p = dm_payload(text="")
    p["reaction"] = {"emoji": "", "target_message_id": "3EB0ORIGINAL"}
    msg = parse_message(p)
    assert msg is not None
    assert msg.reaction is not None
    assert msg.reaction.emoji == ""


def test_parse_message_drops_reaction_without_target_id() -> None:
    # A reaction with no target_message_id can't be matched against
    # anything in the model's context — drop it rather than surface a
    # half-populated event that confuses the model.
    p = dm_payload(text="")
    p["reaction"] = {"emoji": "👍"}
    assert parse_message(p) is None
