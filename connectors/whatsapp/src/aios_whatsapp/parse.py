"""Parse daemon ``message`` notifications into :class:`InboundMessage`.

Drops self-echoes, attachment-only / sticker-only messages, broadcasts,
newsletters, and envelopes with required fields missing or malformed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ChatType = Literal["dm", "group", "broadcast", "newsletter"]


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


def parse_message(params: dict[str, Any]) -> InboundMessage | None:
    """Convert a daemon ``message`` notification's ``params`` to an InboundMessage.

    Returns ``None`` to silently drop the envelope.  Drop rules:

    * ``is_self=True`` — echoes of our own sends.
    * Empty / non-string ``text``.
    * ``chat_type`` not in ``{"dm", "group"}``.
    * Required fields (``id``, ``chat_jid``, ``from_jid``, ``timestamp_ms``)
      missing or wrong type.
    """
    if params.get("is_self"):
        return None

    text = params.get("text")
    if not isinstance(text, str) or not text:
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
    )
