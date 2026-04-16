"""Parse signal-cli envelopes into :class:`InboundMessage`.

signal-cli emits JSON-RPC notifications whose ``params.envelope`` holds the
inbound payload. This module consumes that inner envelope dict and either
returns a flat :class:`InboundMessage` or ``None`` for events we drop
(self-messages, receipts, typing indicators, sync messages, group updates,
unhandled types).

Lifts the mention-placeholder substitution from
``jarvis/receiver.py::_substitute_mentions`` with the mention param flattened
to ``list[dict[str, Any]]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ChatType = Literal["dm", "group"]

# Unicode Object Replacement Character used by Signal for @-mentions.
MENTION_PLACEHOLDER = "\ufffc"


@dataclass(slots=True, frozen=True)
class Attachment:
    content_type: str
    filename: str | None
    signal_file: str | None  # signal-cli's local path, when provided


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
    raw_chat_id: str  # counterparty UUID (dm) or raw (non-URL-safe) base64 group id
    sender_uuid: str
    sender_name: str | None
    chat_name: str | None  # group name if group, else None
    timestamp_ms: int
    text: str
    attachments: tuple[Attachment, ...]
    reply: Reply | None
    reaction: Reaction | None


def _substitute_mentions(text: str, mentions: list[dict[str, Any]]) -> str:
    """Replace U+FFFC placeholders with readable ``@Name`` form.

    Process mentions back-to-front so earlier indices stay valid. Each
    mention dict matches signal-cli's shape: ``{"name", "number", "uuid",
    "start", "length"}`` with ``number`` optional.
    """
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


def _source_uuid(envelope: dict[str, Any]) -> str | None:
    """Return the envelope's source ACI UUID, or ``None``."""
    src = envelope.get("sourceUuid")
    if isinstance(src, str) and src:
        return src
    return None


def parse_envelope(
    envelope: dict[str, Any],
    *,
    bot_account_uuid: str,
) -> InboundMessage | None:
    """Parse a signal-cli ``envelope`` dict into :class:`InboundMessage`.

    Returns ``None`` for:

    - Self-messages (``sourceUuid == bot_account_uuid``)
    - Receipt messages
    - Typing indicators
    - Sync messages (envelopes with ``syncMessage`` and no ``dataMessage``)
    - Group update events (membership/rename — out of scope for v1)
    - ``dataMessage`` without text, attachments, or reaction content
    """
    source_uuid = _source_uuid(envelope)
    if source_uuid is None:
        return None
    if source_uuid == bot_account_uuid:
        return None  # self

    if envelope.get("receiptMessage"):
        return None
    if envelope.get("typingMessage"):
        return None

    data_message = envelope.get("dataMessage")
    if not isinstance(data_message, dict):
        return None

    timestamp_ms = int(envelope.get("timestamp", 0))
    sender_name_raw = envelope.get("sourceName")
    sender_name = sender_name_raw if isinstance(sender_name_raw, str) and sender_name_raw else None

    # Chat identification — prefer groupInfo.groupId when present and non-empty.
    group_info = data_message.get("groupInfo")
    group_id: str | None = None
    group_name: str | None = None
    if isinstance(group_info, dict):
        gid = group_info.get("groupId")
        if isinstance(gid, str) and gid:
            group_id = gid
        gname = group_info.get("groupName")
        if isinstance(gname, str) and gname:
            group_name = gname

    chat_type: ChatType = "group" if group_id else "dm"
    raw_chat_id: str = group_id if group_id else source_uuid

    # Skip group metadata updates — out of scope for v1.
    if (
        isinstance(group_info, dict)
        and group_info.get("type") == "UPDATE"
        and not data_message.get("message")
        and not data_message.get("reaction")
        and not data_message.get("attachments")
    ):
        return None

    # Reaction message.
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

    # Quote (reply).
    quote_raw = data_message.get("quote")
    reply: Reply | None = None
    if isinstance(quote_raw, dict):
        quote_author = quote_raw.get("authorUuid") or quote_raw.get("author")
        quote_id = quote_raw.get("id")
        quote_text = quote_raw.get("text")
        if isinstance(quote_author, str) and quote_id is not None:
            # Quotes don't carry mention metadata — fall back to a generic marker.
            if isinstance(quote_text, str):
                quote_text = quote_text.replace(MENTION_PLACEHOLDER, "@mention")
            else:
                quote_text = None
            reply = Reply(
                author_uuid=quote_author,
                timestamp_ms=int(quote_id),
                text=quote_text,
            )

    # Attachments.
    attachments_raw = data_message.get("attachments") or []
    attachments: tuple[Attachment, ...] = tuple(
        Attachment(
            content_type=a.get("contentType", "application/octet-stream"),
            filename=a.get("filename") if isinstance(a.get("filename"), str) else None,
            signal_file=a.get("file") if isinstance(a.get("file"), str) else None,
        )
        for a in attachments_raw
        if isinstance(a, dict)
    )

    # Text with mention substitution.
    raw_text = data_message.get("message")
    mentions = data_message.get("mentions")
    if isinstance(raw_text, str):
        text = _substitute_mentions(raw_text, mentions) if isinstance(mentions, list) else raw_text
    else:
        text = ""

    # Drop truly empty envelopes (no text, no attachments, no reaction).
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
    """Render the text the agent will see.

    Plain message text, plus one ``[attachment: <name> (<mime>)]`` marker per
    attachment. If the message has no text body but has attachments, the
    result starts with the marker lines.
    """
    parts: list[str] = []
    if msg.text:
        parts.append(msg.text)
    for a in msg.attachments:
        name = a.filename or "(unnamed)"
        parts.append(f"[attachment: {name} ({a.content_type})]")
    return "\n".join(parts)
