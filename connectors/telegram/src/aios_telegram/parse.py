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
from telegram.constants import ChatType, MessageEntityType

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
class Mention:
    """One structured @-mention from a Telegram inbound.

    Surfaced for ``text_mention`` entities (which embed a full ``User``
    object with ``id``) and synthesized for plain ``@<bot_username>``
    matches against the running bot. Plain ``@<other_user>`` mentions
    carry no user_id on the wire and are intentionally not surfaced —
    the model can substring-match ``text`` if it needs to inspect them.
    """

    user_id: int
    name: str | None


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
    mentions: tuple[Mention, ...]
    reply: Reply | None
    edited: bool = False
    # Epoch-ms of the most recent edit, or None if never edited. Telegram
    # preserves message_id across edits, so this is the only field that
    # distinguishes one edit revision from another (and from the
    # original) — load-bearing for inbound event_id uniqueness.
    edit_date_ms: int | None = None
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


def _parse_mentions(
    message: Message,
    *,
    bot_id: int,
    bot_username: str | None,
    bot_display_name: str | None,
) -> tuple[Mention, ...]:
    """Extract structured mentions from Telegram entities.

    Surfaces:
    - ``text_mention`` entities directly — their embedded ``User`` carries
      a stable ``id`` (and usually ``first_name``).
    - Plain ``mention`` entities whose text equals ``@<bot_username>`` —
      Telegram doesn't ship user_id for ``@username`` mentions on the
      wire, but a self-tag is the high-value case the model needs to
      detect, so we synthesize a structured entry against the bot's own
      id when ``bot_username`` is supplied.

    Plain ``mention`` entries for *other* users carry no resolvable
    user_id and are intentionally not surfaced — substring-matching
    ``text`` is the model's fallback for those.

    Entity offsets/lengths are UTF-16 code units on the wire; PTB's
    ``parse_entity`` / ``parse_caption_entity`` handles the conversion
    to Python's code-point indexing, so we delegate to it rather than
    indexing ``message.text`` directly.
    """
    if message.text:
        entities = message.entities
        get_entity_text = message.parse_entity
    elif message.caption:
        entities = message.caption_entities
        get_entity_text = message.parse_caption_entity
    else:
        return ()
    if not entities:
        return ()
    self_tag = f"@{bot_username}".casefold() if bot_username else None
    out: list[Mention] = []
    for entity in entities:
        if entity.type == MessageEntityType.TEXT_MENTION and entity.user is not None:
            out.append(
                Mention(
                    user_id=entity.user.id,
                    name=entity.user.full_name or None,
                )
            )
        elif (
            entity.type == MessageEntityType.MENTION
            and self_tag is not None
            and get_entity_text(entity).casefold() == self_tag
        ):
            out.append(Mention(user_id=bot_id, name=bot_display_name or bot_username))
    return tuple(out)


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
    if message.document and not message.animation:
        # PTB exposes GIFs as both ``message.animation`` and
        # ``message.document`` with the same file_id + file_name; surfacing
        # both yields duplicate attachments the supervisor's staging layer
        # rejects.  The animation branch below carries the same file.
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


def parse_message(
    message: Message,
    *,
    bot_id: int,
    bot_username: str | None = None,
    bot_display_name: str | None = None,
) -> InboundMessage | None:
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

    mentions = _parse_mentions(
        message,
        bot_id=bot_id,
        bot_username=bot_username,
        bot_display_name=bot_display_name,
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
        mentions=mentions,
        reply=reply,
        edited=message.edit_date is not None,
        edit_date_ms=(
            int(message.edit_date.timestamp() * 1000) if message.edit_date is not None else None
        ),
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
