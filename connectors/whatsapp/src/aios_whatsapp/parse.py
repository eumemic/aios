"""Parse daemon ``message`` notifications into :class:`InboundMessage`.

Drops self-echoes, broadcasts, newsletters, and envelopes with
required fields missing or malformed.  Attachment-only and
sticker-only messages are KEPT (their text is empty) — the
connector renders them as zero-content events whose
``attachments``/``metadata.sticker_emoji`` carry the model-relevant
signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ChatType = Literal["dm", "group", "broadcast", "newsletter"]


@dataclass(slots=True, frozen=True)
class InboundAttachment:
    """Inbound media bundle the daemon wrote to disk for us.

    ``host_path`` is the daemon-side absolute path; the Python
    connector reads its bytes off-loop and forwards a
    ``(filename, bytes, content_type)`` tuple to the aios SDK.
    """

    host_path: str
    filename: str
    content_type: str


@dataclass(slots=True, frozen=True)
class InboundReaction:
    """Peer-side reaction to a message, surfaced as a metadata block.

    ``target_message_id`` identifies which message in our session
    history the reaction targets.  Empty ``emoji`` means the peer
    removed an earlier reaction — surfaced explicitly so the model
    can update any "they reacted with X" state.
    """

    emoji: str
    target_message_id: str


@dataclass(slots=True, frozen=True)
class InboundMessage:
    chat_type: ChatType
    chat_jid: str
    chat_name: str | None
    sender_jid: str
    sender_name: str | None
    message_id: str
    timestamp_ms: int
    text: str
    attachments: tuple[InboundAttachment, ...] = field(default_factory=tuple)
    # ``sticker_emoji`` is None when no sticker was attached.  An empty
    # string means "sticker received but the sender's WhatsApp client
    # didn't pick an emoji label" (custom stickers from the sticker
    # maker frequently land this way) — the connector surfaces this as
    # a metadata-only signal so the model still knows the peer sent
    # one.  Pre-fix both empty AND missing collapsed to None, so the
    # parser dropped no-emoji stickers entirely.
    sticker_emoji: str | None = None
    reaction: InboundReaction | None = None
    edit_target_message_id: str | None = None
    revoke_target_message_id: str | None = None
    # ``quoted_message_id`` is set when this inbound is a reply to one
    # of the bot's earlier messages (or to any prior message in the
    # chat the peer chose to quote).  The harness threads it into the
    # event metadata so the model can address the right thread instead
    # of inferring from message order.
    quoted_message_id: str | None = None
    mentioned_jids: tuple[str, ...] = field(default_factory=tuple)


def parse_message(params: dict[str, Any]) -> InboundMessage | None:
    """Convert a daemon ``message`` notification's ``params`` to an InboundMessage.

    Returns ``None`` to silently drop the envelope.  Drop rules:

    * ``is_self=True`` — echoes of our own sends.
    * ``chat_type`` not in ``{"dm", "group"}``.
    * Empty text AND empty attachments AND no sticker AND no other
      metadata signal — no signal to surface (e.g. an empty
      reaction-only message that slipped past whatsmeow's
      protocol-message filter).
    * Required fields (``id``, ``chat_jid``, ``from_jid``, ``timestamp_ms``)
      missing or wrong type.
    """
    if params.get("is_self"):
        return None

    chat_type = params.get("chat_type")
    if chat_type not in ("dm", "group"):
        return None

    message_id = params.get("id")
    chat_jid = params.get("chat_jid")
    sender_jid = params.get("from_jid")
    timestamp_ms = params.get("timestamp_ms")
    if not (
        isinstance(message_id, str)
        and isinstance(chat_jid, str)
        and isinstance(sender_jid, str)
        and isinstance(timestamp_ms, int)
    ):
        return None

    raw_text = params.get("text")
    text = raw_text if isinstance(raw_text, str) else ""
    # Whitespace-only text is truthy in Python but carries no signal
    # — peer mis-tapped or sent an accessibility-input artifact.
    # Treat as empty for the "no signal" drop check below so we don't
    # emit blank-bubble inbound events with nothing the model can act
    # on.
    text_is_signal = bool(text.strip())

    attachments = _parse_attachments(params.get("attachments"))

    # Sticker presence is the signal; the emoji label is informational
    # and may be empty for custom stickers without a sender-picked tag.
    # Accepting "" (not just truthy strings) lets the drop check below
    # keep no-emoji stickers, which pre-fix were silently swallowed.
    raw_sticker = params.get("sticker_emoji")
    sticker_emoji = raw_sticker if isinstance(raw_sticker, str) else None

    reaction = _parse_reaction(params.get("reaction"))
    edit_target_message_id = _parse_target(params.get("edit"))
    revoke_target_message_id = _parse_target(params.get("revoke"))
    raw_quoted = params.get("quoted_message_id")
    quoted_message_id = raw_quoted if isinstance(raw_quoted, str) and raw_quoted else None
    raw_mentions = params.get("mentioned_jids")
    mentioned_jids: tuple[str, ...] = (
        tuple(j for j in raw_mentions if isinstance(j, str) and j)
        if isinstance(raw_mentions, list)
        else ()
    )

    if (
        not text_is_signal
        and not attachments
        and sticker_emoji is None
        and reaction is None
        and edit_target_message_id is None
        and revoke_target_message_id is None
    ):
        return None

    raw_chat_name = params.get("chat_name")
    raw_push_name = params.get("from_push_name")

    return InboundMessage(
        chat_type=chat_type,
        chat_jid=chat_jid,
        chat_name=raw_chat_name if isinstance(raw_chat_name, str) else None,
        sender_jid=sender_jid,
        sender_name=raw_push_name if isinstance(raw_push_name, str) else None,
        message_id=message_id,
        timestamp_ms=timestamp_ms,
        text=text,
        attachments=attachments,
        sticker_emoji=sticker_emoji,
        reaction=reaction,
        edit_target_message_id=edit_target_message_id,
        revoke_target_message_id=revoke_target_message_id,
        quoted_message_id=quoted_message_id,
        mentioned_jids=mentioned_jids,
    )


def _parse_target(raw: Any) -> str | None:
    """Pull ``target_message_id`` out of the daemon's ``edit`` / ``revoke``
    block.  Returns None for missing blocks or missing ids — the model
    has nothing to act on without an id, so silently drop.
    """
    if not isinstance(raw, dict):
        return None
    target_id = raw.get("target_message_id")
    if not isinstance(target_id, str) or not target_id:
        return None
    return target_id


def _parse_reaction(raw: Any) -> InboundReaction | None:
    """Normalize the daemon's ``reaction`` payload.

    Requires a target_message_id (otherwise the model can't match it
    against anything in its context); emoji may be empty (peer
    removing a prior reaction).
    """
    if not isinstance(raw, dict):
        return None
    target_id = raw.get("target_message_id")
    if not isinstance(target_id, str) or not target_id:
        return None
    emoji = raw.get("emoji")
    if not isinstance(emoji, str):
        return None
    return InboundReaction(emoji=emoji, target_message_id=target_id)


def _parse_attachments(raw: Any) -> tuple[InboundAttachment, ...]:
    """Normalize the daemon's ``attachments`` list, dropping malformed entries.

    The daemon emits items of shape ``{"path", "mimetype", "filename"}``
    on a best-effort basis (download failures land NO entry rather than
    a half-populated one), so an empty list is the typical no-media
    case and a missing/None field is the malformed-entry signal.
    """
    if not isinstance(raw, list):
        return ()
    out: list[InboundAttachment] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        mimetype = item.get("mimetype")
        filename = item.get("filename")
        if not (isinstance(path, str) and path and isinstance(mimetype, str) and mimetype):
            continue
        out.append(
            InboundAttachment(
                host_path=path,
                filename=filename if isinstance(filename, str) and filename else "unnamed",
                content_type=mimetype,
            )
        )
    return tuple(out)
