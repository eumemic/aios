"""Parse daemon ``message`` notifications into :class:`InboundMessage`.

PR 2 scope is text only: media, quoted replies, mentions, reactions,
edits, broadcasts and newsletters are dropped (later PRs reintroduce
them as they're implemented end-to-end).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .addressing import ChatType


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
    * Empty ``text`` — attachment-only / sticker-only messages (PR 5).
    * ``chat_type`` not in ``{"dm", "group"}`` — broadcasts and
      newsletters are out of scope for v1.
    * Required fields (``id``, ``chat_jid``, ``from_jid``, ``timestamp_ms``)
      missing or malformed — daemon contract violation, logged upstream
      via the listener's ``rpc.listener.bad_params`` path.
    """
    if params.get("is_self"):
        return None

    text = params.get("text") or ""
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

    chat_name = params.get("chat_name") if isinstance(params.get("chat_name"), str) else None
    sender_name = (
        params.get("from_push_name") if isinstance(params.get("from_push_name"), str) else None
    )

    return InboundMessage(
        chat_type=chat_type,
        chat_jid=chat_jid,
        chat_name=chat_name,
        sender_jid=sender_jid,
        sender_name=sender_name,
        message_id=message_id,
        timestamp_ms=timestamp_ms,
        text=text,
    )
