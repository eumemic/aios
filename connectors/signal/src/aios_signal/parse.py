"""Parse signal-cli envelopes into :class:`InboundMessage`.

signal-cli emits JSON-RPC notifications whose ``params.envelope`` holds the
inbound payload. This module returns ``None`` for events we drop
(self-messages, receipts, typing, sync, group updates, empty data messages).

Mention substitution is lifted from ``jarvis/receiver.py`` with the mention
argument flattened to ``list[dict[str, Any]]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .addressing import ChatType

# Unicode Object Replacement Character used by Signal for @-mentions.
MENTION_PLACEHOLDER = "\ufffc"


@dataclass(slots=True, frozen=True)
class Attachment:
    content_type: str
    filename: str | None


@dataclass(slots=True, frozen=True)
class Reaction:
    emoji: str
    target_author_uuid: str
    target_timestamp_ms: int


@dataclass(slots=True, frozen=True)
class Reply:
    author_uuid: str
    timestamp_ms: int
    text: str | None


@dataclass(slots=True, frozen=True)
class InboundMessage:
    chat_type: ChatType
    raw_chat_id: str
    sender_uuid: str
    sender_name: str | None
    chat_name: str | None
    timestamp_ms: int
    text: str
    attachments: tuple[Attachment, ...]
    reply: Reply | None
    reaction: Reaction | None


def _substitute_mentions(text: str, mentions: list[dict[str, Any]]) -> str:
    # Process back-to-front so earlier indices stay valid as we splice.
    if not mentions:
        return text

    sorted_mentions = sorted(mentions, key=lambda m: int(m.get("start", 0)), reverse=True)

    result = text
    for mention in sorted_mentions:
        start = int(mention.get("start", 0))
        length = int(mention.get("length", 0))
        if start >= len(result) or result[start] != MENTION_PLACEHOLDER:
            continue
        display_name = mention.get("name") or mention.get("number") or mention.get("uuid") or ""
        result = result[:start] + f"@{display_name}" + result[start + length :]

    return result


def parse_envelope(
    envelope: dict[str, Any],
    *,
    bot_account_uuid: str,
) -> InboundMessage | None:
    source_uuid = envelope.get("sourceUuid")
    if not isinstance(source_uuid, str) or not source_uuid:
        return None
    if source_uuid == bot_account_uuid:
        return None

    if envelope.get("receiptMessage") or envelope.get("typingMessage"):
        return None

    data_message = envelope.get("dataMessage")
    if not isinstance(data_message, dict):
        return None

    timestamp_ms = int(envelope.get("timestamp", 0))
    sender_name = envelope.get("sourceName") or None

    group_info = data_message.get("groupInfo")
    group_id: str | None = None
    group_name: str | None = None
    if isinstance(group_info, dict):
        group_id = group_info.get("groupId") or None
        group_name = group_info.get("groupName") or None

    chat_type: ChatType = "group" if group_id else "dm"
    raw_chat_id: str = group_id if group_id else source_uuid

    # Group metadata updates (membership/rename) — out of scope for v1.
    if (
        isinstance(group_info, dict)
        and group_info.get("type") == "UPDATE"
        and not data_message.get("message")
        and not data_message.get("reaction")
        and not data_message.get("attachments")
    ):
        return None

    reaction_raw = data_message.get("reaction")
    reaction: Reaction | None = None
    if isinstance(reaction_raw, dict):
        target_author = reaction_raw.get("targetAuthorUuid") or reaction_raw.get("targetAuthor")
        emoji = reaction_raw.get("emoji")
        target_ts = reaction_raw.get("targetSentTimestamp")
        if isinstance(target_author, str) and isinstance(emoji, str) and target_ts is not None:
            reaction = Reaction(
                emoji=emoji,
                target_author_uuid=target_author,
                target_timestamp_ms=int(target_ts),
            )

    quote_raw = data_message.get("quote")
    reply: Reply | None = None
    if isinstance(quote_raw, dict):
        quote_author = quote_raw.get("authorUuid") or quote_raw.get("author")
        quote_id = quote_raw.get("id")
        quote_text = quote_raw.get("text")
        if isinstance(quote_author, str) and quote_id is not None:
            # Quotes don't carry mention metadata — placeholders get a generic marker.
            if isinstance(quote_text, str):
                quote_text = quote_text.replace(MENTION_PLACEHOLDER, "@mention")
            else:
                quote_text = None
            reply = Reply(
                author_uuid=quote_author,
                timestamp_ms=int(quote_id),
                text=quote_text,
            )

    attachments_raw = data_message.get("attachments") or []
    attachments: tuple[Attachment, ...] = tuple(
        Attachment(
            content_type=a.get("contentType", "application/octet-stream"),
            filename=a.get("filename") if isinstance(a.get("filename"), str) else None,
        )
        for a in attachments_raw
        if isinstance(a, dict)
    )

    raw_text = data_message.get("message")
    mentions = data_message.get("mentions")
    if isinstance(raw_text, str):
        text = _substitute_mentions(raw_text, mentions) if isinstance(mentions, list) else raw_text
    else:
        text = ""

    if not text and not attachments and reaction is None:
        return None

    return InboundMessage(
        chat_type=chat_type,
        raw_chat_id=raw_chat_id,
        sender_uuid=source_uuid,
        sender_name=sender_name,
        chat_name=group_name,
        timestamp_ms=timestamp_ms,
        text=text,
        attachments=attachments,
        reply=reply,
        reaction=reaction,
    )


def build_content_text(msg: InboundMessage) -> str:
    parts: list[str] = []
    if msg.text:
        parts.append(msg.text)
    for a in msg.attachments:
        parts.append(f"[attachment: {a.filename or '(unnamed)'} ({a.content_type})]")
    return "\n".join(parts)
