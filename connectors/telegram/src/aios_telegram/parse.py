"""Parse python-telegram-bot ``Message`` objects into :class:`InboundMessage`.

Returns ``None`` for messages from the bot itself, bot-to-bot
traffic, channel posts (no ``from_user``), and otherwise-empty
messages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from telegram import Message
from telegram.constants import ChatType

ChatKind = Literal["dm", "group", "supergroup", "channel"]


@dataclass(slots=True, frozen=True)
class Attachment:
    file_id: str
    content_type: str
    filename: str


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
    attachments: tuple[Attachment, ...]
    reply: Reply | None


def _chat_kind(chat_type: str) -> ChatKind:
    if chat_type == ChatType.PRIVATE:
        return "dm"
    if chat_type == ChatType.GROUP:
        return "group"
    if chat_type == ChatType.SUPERGROUP:
        return "supergroup"
    return "channel"


def _extract_attachments(message: Message) -> tuple[Attachment, ...]:
    """Pick out media we forward: photo / voice / document / video / audio."""
    out: list[Attachment] = []
    if message.photo:
        # ``photo`` is a tuple of progressively larger PhotoSizes; the
        # last one is the largest the bot has access to.
        largest = message.photo[-1]
        out.append(
            Attachment(
                file_id=largest.file_id,
                content_type="image/jpeg",
                filename=f"photo-{message.message_id}.jpg",
            )
        )
    if message.voice:
        out.append(
            Attachment(
                file_id=message.voice.file_id,
                content_type=message.voice.mime_type or "audio/ogg",
                filename=f"voice-{message.message_id}.ogg",
            )
        )
    if message.document:
        doc = message.document
        out.append(
            Attachment(
                file_id=doc.file_id,
                content_type=doc.mime_type or "application/octet-stream",
                filename=doc.file_name or f"document-{message.message_id}",
            )
        )
    if message.video:
        vid = message.video
        out.append(
            Attachment(
                file_id=vid.file_id,
                content_type=vid.mime_type or "video/mp4",
                filename=vid.file_name or f"video-{message.message_id}.mp4",
            )
        )
    if message.audio:
        aud = message.audio
        out.append(
            Attachment(
                file_id=aud.file_id,
                content_type=aud.mime_type or "audio/mpeg",
                filename=aud.file_name or f"audio-{message.message_id}.mp3",
            )
        )
    return tuple(out)


def parse_message(message: Message, *, bot_id: int) -> InboundMessage | None:
    sender = message.from_user
    if sender is None:
        # Channel posts and anonymous admins have no ``from_user``.
        return None
    if sender.id == bot_id:
        return None
    if sender.is_bot:
        return None

    text = message.text or message.caption or ""
    attachments = _extract_attachments(message)
    if not text and not attachments:
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
        attachments=attachments,
        reply=reply,
    )
