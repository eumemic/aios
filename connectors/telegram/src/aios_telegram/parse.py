"""Parse python-telegram-bot ``Message`` objects into :class:`InboundMessage`.

Returns ``None`` for messages from the bot itself, bot-to-bot
traffic, channel posts (no ``from_user``), and otherwise-empty
messages.

Also exposes :func:`parse_reaction` for ``MessageReactionUpdated``
updates — Telegram delivers reactions as a separate update type
(not embedded in the reacted-to message), so they need their own
parse path that emits a distinct :class:`InboundReaction` shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from telegram import Message, MessageReactionUpdated, ReactionTypeEmoji
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
    edited: bool = False
    # Emoji that came with a sticker, when the inbound was a sticker.
    # Sometimes the sticker file is non-vision-readable (animated/video),
    # and the emoji is the only text-side cue the model gets.
    sticker_emoji: str | None = None


@dataclass(slots=True, frozen=True)
class InboundReaction:
    """User-on-message reaction event.

    Telegram delivers reaction changes as ``MessageReactionUpdated``
    updates, which carry the *delta* between the prior reaction set
    and the new one — letting the model see additions, removals, or
    swaps explicitly rather than as a stale "all current reactions"
    snapshot.
    """

    chat_kind: ChatKind
    chat_id: int
    chat_name: str | None
    sender_id: int
    sender_name: str | None
    target_message_id: int
    timestamp_ms: int
    old_emojis: tuple[str, ...]
    new_emojis: tuple[str, ...]


def _chat_kind(chat_type: str) -> ChatKind:
    if chat_type == ChatType.PRIVATE:
        return "dm"
    if chat_type == ChatType.GROUP:
        return "group"
    if chat_type == ChatType.SUPERGROUP:
        return "supergroup"
    return "channel"


def _extract_attachments(message: Message) -> tuple[Attachment, ...]:
    """Pick out media we forward."""
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
    if message.sticker:
        st = message.sticker
        if st.is_animated:
            # Lottie JSON; vision pipeline can't read it but the emoji
            # is still surfaced via sticker_emoji metadata.
            content_type, ext = "application/x-tgsticker", ".tgs"
        elif st.is_video:
            content_type, ext = "video/webm", ".webm"
        else:
            content_type, ext = "image/webp", ".webp"
        out.append(
            Attachment(
                file_id=st.file_id,
                content_type=content_type,
                filename=f"sticker-{message.message_id}{ext}",
            )
        )
    if message.animation:
        # GIFs in Telegram are MP4-encoded animations.
        anim = message.animation
        out.append(
            Attachment(
                file_id=anim.file_id,
                content_type=anim.mime_type or "video/mp4",
                filename=anim.file_name or f"animation-{message.message_id}.mp4",
            )
        )
    if message.video_note:
        vn = message.video_note
        out.append(
            Attachment(
                file_id=vn.file_id,
                content_type="video/mp4",
                filename=f"video_note-{message.message_id}.mp4",
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

    sticker_emoji = message.sticker.emoji if message.sticker is not None else None

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
        edited=message.edit_date is not None,
        sticker_emoji=sticker_emoji,
    )


def parse_reaction(reaction: MessageReactionUpdated, *, bot_id: int) -> InboundReaction | None:
    """Parse a ``MessageReactionUpdated`` update into :class:`InboundReaction`.

    Returns ``None`` for the bot's own reactions, other-bot reactions,
    anonymous reactions (``actor_chat`` instead of ``user``), and
    no-op deltas (both lists empty after filtering custom emoji).

    Only ``ReactionTypeEmoji`` is surfaced; custom (premium) reactions
    are dropped because their stable identifier is a numeric id, not
    a glyph the model can read or echo back via ``telegram_react``.
    """
    user = reaction.user
    if user is None:
        # Anonymous supergroup reactions ride on ``actor_chat`` — punt
        # in v1, the model can't react back to them anyway.
        return None
    if user.id == bot_id or user.is_bot:
        return None
    new_emojis = tuple(
        r.emoji for r in (reaction.new_reaction or ()) if isinstance(r, ReactionTypeEmoji)
    )
    old_emojis = tuple(
        r.emoji for r in (reaction.old_reaction or ()) if isinstance(r, ReactionTypeEmoji)
    )
    if not new_emojis and not old_emojis:
        return None
    chat = reaction.chat
    chat_kind = _chat_kind(chat.type)
    return InboundReaction(
        chat_kind=chat_kind,
        chat_id=chat.id,
        chat_name=chat.title if chat_kind != "dm" else None,
        sender_id=user.id,
        sender_name=user.full_name or None,
        target_message_id=reaction.message_id,
        timestamp_ms=int(reaction.date.timestamp() * 1000),
        old_emojis=old_emojis,
        new_emojis=new_emojis,
    )
