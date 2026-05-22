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
    sticker_emoji: str | None = None


def parse_message(params: dict[str, Any]) -> InboundMessage | None:
    """Convert a daemon ``message`` notification's ``params`` to an InboundMessage.

    Returns ``None`` to silently drop the envelope.  Drop rules:

    * ``is_self=True`` — echoes of our own sends.
    * ``chat_type`` not in ``{"dm", "group"}``.
    * Empty text AND empty attachments AND no sticker — no signal to
      surface (e.g. an empty reaction-only message that slipped past
      whatsmeow's protocol-message filter).
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

    attachments = _parse_attachments(params.get("attachments"))

    raw_sticker = params.get("sticker_emoji")
    sticker_emoji = raw_sticker if isinstance(raw_sticker, str) and raw_sticker else None

    if not text and not attachments and sticker_emoji is None:
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
    )


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
