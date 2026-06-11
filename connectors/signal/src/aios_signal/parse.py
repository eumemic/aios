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

import structlog

from .addressing import ChatType

log = structlog.get_logger(__name__)

# Unicode Object Replacement Character used by Signal for @-mentions.
MENTION_PLACEHOLDER = "\ufffc"


def is_group_update_envelope(envelope: dict[str, Any]) -> bool:
    data_message = envelope.get("dataMessage")
    if not isinstance(data_message, dict):
        return False
    group_info = data_message.get("groupInfo")
    return isinstance(group_info, dict) and group_info.get("type") == "UPDATE"


@dataclass(slots=True, frozen=True)
class Attachment:
    content_type: str
    filename: str | None
    host_path: str | None
    # signal-cli's storage id, used to compute the on-disk path when
    # the daemon's JSON-RPC envelope omits the ``file`` field (which it
    # does in JSON-RPC daemon mode 0.14.x — only the legacy CLI output
    # included it).  By signal-cli convention the file lives at
    # ``<config_dir>/attachments/<id>``.
    id: str | None


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
class Mention:
    """One structured @-mention from a Signal inbound.

    The placeholder-substituted text in ``InboundMessage.text`` is what
    the sender's UI rendered; the structured ``Mention`` list is what
    the platform actually encoded.  Agents that need to distinguish "the
    sender typed my name as text" from "the sender's client emitted a
    mention targeting my UUID" read ``mentions`` (and ``self_mentioned``
    on the metadata) rather than substring-searching ``text``.
    """

    uuid: str
    name: str | None


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
    mentions: tuple[Mention, ...]
    reply: Reply | None
    reaction: Reaction | None
    edited: bool = False
    edit_target_timestamp_ms: int | None = None


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
        log.warning(
            "signal.inbound.skipped", reason="source_less", timestamp=envelope.get("timestamp")
        )
        return None
    if source_uuid == bot_account_uuid:
        log.debug("signal.inbound.skipped", reason="self_message", source_uuid=source_uuid)
        return None

    if envelope.get("receiptMessage"):
        log.debug("signal.inbound.skipped", reason="receipt", source_uuid=source_uuid)
        return None
    if envelope.get("typingMessage"):
        log.debug("signal.inbound.skipped", reason="typing", source_uuid=source_uuid)
        return None

    # signal-cli emits two top-level shapes for inbound payloads:
    # ``dataMessage`` for plain sends and ``editMessage.dataMessage``
    # for edits.  An edit envelope carries the EDITED content's
    # ``timestamp`` at the envelope root and ``targetSentTimestamp``
    # inside ``editMessage`` pointing at the original.  Both shapes
    # feed the same downstream parsing — only the ``edited`` /
    # ``edit_target_timestamp_ms`` fields on the result differ.
    edited = False
    edit_target_ts: int | None = None
    data_message = envelope.get("dataMessage")
    if not isinstance(data_message, dict):
        edit_envelope = envelope.get("editMessage")
        if isinstance(edit_envelope, dict):
            nested = edit_envelope.get("dataMessage")
            if isinstance(nested, dict):
                data_message = nested
                edited = True
                target_raw = edit_envelope.get("targetSentTimestamp")
                if isinstance(target_raw, int):
                    edit_target_ts = target_raw
            else:
                log.warning(
                    "signal.inbound.skipped", reason="edit_no_data", source_uuid=source_uuid
                )
                return None
        else:
            log.warning("signal.inbound.skipped", reason="no_content", source_uuid=source_uuid)
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
        log.warning("signal.inbound.skipped", reason="group_update", source_uuid=source_uuid)
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
            host_path=a.get("file") if isinstance(a.get("file"), str) else None,
            id=a.get("id") if isinstance(a.get("id"), str) else None,
        )
        for a in attachments_raw
        if isinstance(a, dict)
    )

    raw_text = data_message.get("message")
    mentions_raw = data_message.get("mentions")
    if isinstance(raw_text, str):
        text = (
            _substitute_mentions(raw_text, mentions_raw)
            if isinstance(mentions_raw, list)
            else raw_text
        )
    else:
        text = ""

    parsed_mentions: tuple[Mention, ...] = ()
    if isinstance(mentions_raw, list):
        parsed_mentions = tuple(
            Mention(
                uuid=m["uuid"],
                name=m.get("name") if isinstance(m.get("name"), str) else None,
            )
            for m in mentions_raw
            if isinstance(m, dict) and isinstance(m.get("uuid"), str) and m.get("uuid")
        )

    if not text and not attachments and reaction is None:
        log.warning("signal.inbound.skipped", reason="no_content", source_uuid=source_uuid)
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
        mentions=parsed_mentions,
        reply=reply,
        reaction=reaction,
        edited=edited,
        edit_target_timestamp_ms=edit_target_ts,
    )


def build_content_text(msg: InboundMessage) -> str:
    return msg.text
