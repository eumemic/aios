"""Parse python-telegram-bot ``Message`` objects into :class:`InboundMessage`.

v1 scope is text only. We return ``None`` for anything we don't forward to
aios — messages from the bot itself, bot-to-bot traffic, non-text payloads
(photos, stickers, voice, documents), and messages without a sender.

Edits (``update.edited_message``) and channel posts never reach this
function because the handler in :mod:`aios_telegram.bot` registers on
``update.message`` with a ``filters.TEXT`` gate — this module is only
responsible for the parse, not for the filter-at-dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from telegram import Message
from telegram.constants import ChatType

ChatKind = Literal["dm", "group", "supergroup", "channel"]


@dataclass(slots=True, frozen=True)
class Reply:
    message_id: int
    text: str | None


@dataclass(slots=True, frozen=True)
class InboundMessage:
    chat_kind: ChatKind
    chat_id: int
    chat_name: str | None
    sender_id: int
    sender_name: str | None
    message_id: int
    timestamp_ms: int
    text: str
    reply: Reply | None


def _chat_kind(chat_type: str) -> ChatKind:
    if chat_type == ChatType.PRIVATE:
        return "dm"
    if chat_type == ChatType.GROUP:
        return "group"
    if chat_type == ChatType.SUPERGROUP:
        return "supergroup"
    return "channel"


def parse_message(message: Message, *, bot_id: int) -> InboundMessage | None:
    sender = message.from_user
    if sender is None:
        # Channel posts and anonymous admins have no ``from_user``. Out of
        # scope for v1.
        return None
    if sender.id == bot_id:
        return None
    if sender.is_bot:
        return None

    text = message.text
    if not text:
        # v1 is text-only. Media, stickers, voice, etc. are dropped.
        return None

    chat = message.chat
    chat_kind = _chat_kind(chat.type)
    chat_name: str | None = chat.title if chat_kind != "dm" else None

    reply: Reply | None = None
    if message.reply_to_message is not None:
        reply = Reply(
            message_id=message.reply_to_message.message_id,
            text=message.reply_to_message.text or message.reply_to_message.caption,
        )

    return InboundMessage(
        chat_kind=chat_kind,
        chat_id=chat.id,
        chat_name=chat_name,
        sender_id=sender.id,
        sender_name=sender.full_name or None,
        message_id=message.message_id,
        timestamp_ms=int(message.date.timestamp() * 1000),
        text=text,
        reply=reply,
    )
