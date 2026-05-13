"""Cover ``build_metadata`` — the connector-side helper that turns an
:class:`InboundMessage` into the structured metadata dict aios stamps
on every inbound event.

Most of the function is mechanical field mirroring; the interesting
bits are the optional fields (mentions / reply / reaction) and the
``self_mentioned`` derivation, all of which are reflective of platform
quirks the model needs explicit signal for.
"""

from __future__ import annotations

from aios_signal.connector import build_metadata
from aios_signal.parse import InboundMessage, Mention


def _msg(**overrides: object) -> InboundMessage:
    defaults: dict[str, object] = dict(
        chat_type="dm",
        raw_chat_id="alice-uuid",
        sender_uuid="alice-uuid",
        sender_name="Alice",
        chat_name=None,
        timestamp_ms=1700000000000,
        text="hi",
        attachments=(),
        mentions=(),
        reply=None,
        reaction=None,
    )
    defaults.update(overrides)
    return InboundMessage(**defaults)  # type: ignore[arg-type]


BOT_UUID = "99999999-8888-7777-6666-555555555555"
ALICE_UUID = "11111111-aaaa-aaaa-aaaa-aaaaaaaaaaaa"


def test_no_mentions_omitted() -> None:
    """Empty mentions → no ``mentions`` / ``self_mentioned`` keys at all."""
    meta = build_metadata(_msg(), chat_id="alice-uuid", bot_uuid=BOT_UUID)
    assert "mentions" not in meta
    assert "self_mentioned" not in meta


def test_mentions_surfaced_with_self_mentioned_true() -> None:
    """When the bot's UUID is mentioned, ``self_mentioned`` is True so the
    agent can react/respond differently from incidental name usage."""
    msg = _msg(mentions=(Mention(uuid=BOT_UUID, name="SmokeBot"),))
    meta = build_metadata(msg, chat_id="group-id", bot_uuid=BOT_UUID)
    assert meta["mentions"] == [{"uuid": BOT_UUID, "name": "SmokeBot"}]
    assert meta["self_mentioned"] is True


def test_mentions_surfaced_with_self_mentioned_false() -> None:
    """Mentions of someone other than the bot still ride along, but
    ``self_mentioned`` is False — useful context without summoning."""
    msg = _msg(mentions=(Mention(uuid=ALICE_UUID, name="Alice"),))
    meta = build_metadata(msg, chat_id="group-id", bot_uuid=BOT_UUID)
    assert meta["mentions"] == [{"uuid": ALICE_UUID, "name": "Alice"}]
    assert meta["self_mentioned"] is False


def test_multiple_mentions_preserved_in_order() -> None:
    """Order is preserved so callers can match offsets back to the text."""
    msg = _msg(
        mentions=(
            Mention(uuid=ALICE_UUID, name="Alice"),
            Mention(uuid=BOT_UUID, name=None),
        )
    )
    meta = build_metadata(msg, chat_id="group-id", bot_uuid=BOT_UUID)
    assert meta["mentions"] == [
        {"uuid": ALICE_UUID, "name": "Alice"},
        {"uuid": BOT_UUID, "name": None},
    ]
    assert meta["self_mentioned"] is True
