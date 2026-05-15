"""An edited Telegram message must NOT collide with the original's event_id.

``_emit_message`` (``connector.py:235``) built
``event_id=f"telegram-{chat_id}-{message_id}"``. Telegram preserves
``message_id`` across edits (it delivers an ``edited_message`` update
with the same id), so an edit produced the SAME ``event_id`` as the
original. aios's ``inbound_acks`` table dedups on
``(connector, account, event_id)`` → the edit is rejected as a replay
and its content is silently dropped.

This directly contradicts the connector's own system prompt
(``prompts.py:149``): "Edits arrive as a fresh inbound with
``metadata.edited == True`` — the ``message_id`` is the same as the
original message ... and the body is the new (post-edit) text." The
model is told to expect edits; the dedup layer silently eats them.

The fix threads the edit timestamp into the event_id so each edit
revision is a distinct inbound, while a non-edited message keeps its
original event_id (preserving the redelivery-dedup-after-restart
property the original comment describes).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from telegram import Bot, Message

from aios_telegram.parse import parse_message
from tests.conftest import CONNECTION_ID


@pytest.fixture
def bot() -> Any:
    """Minimal mocked PTB Bot — the ``connector`` fixture wires this in.

    ``_emit_message`` for a text message (no attachments) never touches
    the bot, so a bare MagicMock suffices."""
    return MagicMock()


def _msg(ptb_bot: Bot, *, edit_date: int | None) -> Message:
    """A DM text message with a fixed message_id; ``edit_date`` toggles
    whether it's the original or an edit of that same message."""
    data: dict[str, Any] = {
        "message_id": 730,
        "date": 1700000030,
        "chat": {"id": 123456789, "type": "private"},
        "from": {"id": 123456789, "is_bot": False, "first_name": "Alice"},
        "text": "fixed typo" if edit_date else "fixd typo",
    }
    if edit_date is not None:
        data["edit_date"] = edit_date
    parsed = Message.de_json(data, ptb_bot)
    assert parsed is not None
    return parsed


async def test_edit_does_not_collide_with_original_event_id(
    connector: Any, ptb_bot: Bot, bot_id: int
) -> None:
    """Original + edit of the same message_id must emit distinct event_ids."""
    original = parse_message(_msg(ptb_bot, edit_date=None), bot_id=bot_id)
    edited = parse_message(_msg(ptb_bot, edit_date=1700000040), bot_id=bot_id)
    assert original is not None and edited is not None
    # Telegram preserves message_id across edits — this is exactly why
    # the naive (chat_id, message_id) event_id collides.
    assert original.message_id == edited.message_id

    captured: list[str | None] = []

    async def _fake_emit(**kwargs: Any) -> None:
        captured.append(kwargs.get("event_id"))
        return None

    state = connector.state[CONNECTION_ID]
    with patch.object(connector, "emit_inbound", side_effect=_fake_emit):
        await connector._emit_message(CONNECTION_ID, state, original)
        await connector._emit_message(CONNECTION_ID, state, edited)

    assert captured[0] is not None and captured[1] is not None
    assert captured[0] != captured[1], (
        f"edit emitted the same event_id as the original "
        f"({captured[0]!r}); aios inbound_acks dedups on "
        f"(connector, account, event_id) so the edit is silently "
        f"dropped — contradicting the prompts.py:149 contract that the "
        f"model is told to rely on"
    )


async def test_two_successive_edits_get_distinct_event_ids(
    connector: Any, ptb_bot: Bot, bot_id: int
) -> None:
    """Edit #1 and edit #2 of the same message must also differ — a
    single ``edited: bool`` flag couldn't disambiguate them; the edit
    *timestamp* is what makes each revision a distinct inbound."""
    edit1 = parse_message(_msg(ptb_bot, edit_date=1700000040), bot_id=bot_id)
    edit2 = parse_message(_msg(ptb_bot, edit_date=1700000055), bot_id=bot_id)
    assert edit1 is not None and edit2 is not None

    captured: list[str | None] = []

    async def _fake_emit(**kwargs: Any) -> None:
        captured.append(kwargs.get("event_id"))
        return None

    state = connector.state[CONNECTION_ID]
    with patch.object(connector, "emit_inbound", side_effect=_fake_emit):
        await connector._emit_message(CONNECTION_ID, state, edit1)
        await connector._emit_message(CONNECTION_ID, state, edit2)

    assert captured[0] != captured[1], (
        f"two successive edits of one message emitted the same "
        f"event_id ({captured[0]!r}); the second edit is silently "
        f"dropped by inbound_acks dedup"
    )


async def test_non_edited_message_keeps_stable_event_id(
    connector: Any, ptb_bot: Bot, bot_id: int
) -> None:
    """Regression guard: a non-edited message's event_id must stay
    ``telegram-{chat_id}-{message_id}`` so a redelivered update after a
    runtime restart still dedups (the property the original comment at
    connector.py:230-234 documents)."""
    msg = parse_message(_msg(ptb_bot, edit_date=None), bot_id=bot_id)
    assert msg is not None

    captured: list[str | None] = []

    async def _fake_emit(**kwargs: Any) -> None:
        captured.append(kwargs.get("event_id"))
        return None

    state = connector.state[CONNECTION_ID]
    with patch.object(connector, "emit_inbound", side_effect=_fake_emit):
        await connector._emit_message(CONNECTION_ID, state, msg)

    assert captured[0] == f"telegram-{msg.chat_id}-{msg.message_id}", (
        f"non-edited message event_id changed to {captured[0]!r}; this "
        f"breaks redelivery dedup across runtime restarts for the "
        f"common (unedited) case"
    )
