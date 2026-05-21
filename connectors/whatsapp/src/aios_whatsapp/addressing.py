"""WhatsApp JID parsing.

WhatsApp JIDs are already URL-safe (digits + ``@`` + ``s.whatsapp.net`` /
``g.us`` / ``broadcast`` / ``newsletter`` + optional ``:<device>`` on
LID-routed accounts).  The connector uses them as ``chat_id`` verbatim —
no encoding round-trip like Signal needs for its group IDs.
"""

from __future__ import annotations

import re
from typing import Literal

ChatType = Literal["dm", "group", "broadcast", "newsletter"]

_SERVER_TO_CHAT_TYPE: dict[str, ChatType] = {
    "s.whatsapp.net": "dm",
    "g.us": "group",
    "broadcast": "broadcast",
    "newsletter": "newsletter",
}

_JID_RE = re.compile(r"\A([0-9]+)(?::([0-9]+))?@(s\.whatsapp\.net|g\.us|broadcast|newsletter)\Z")


def is_valid_chat_id(chat_id: str) -> bool:
    """True if ``chat_id`` is a well-formed WhatsApp JID."""
    return _JID_RE.match(chat_id) is not None


def chat_type_of(chat_id: str) -> ChatType:
    """Return the chat type encoded in the JID's server suffix.

    Raises :class:`ValueError` on a malformed JID.
    """
    m = _JID_RE.match(chat_id)
    if m is None:
        raise ValueError(f"not a WhatsApp JID: {chat_id!r}")
    return _SERVER_TO_CHAT_TYPE[m.group(3)]


def jid_user_segment(chat_id: str) -> str:
    """Return the user portion (E.164 digits or group hash) sans server suffix."""
    m = _JID_RE.match(chat_id)
    if m is None:
        raise ValueError(f"not a WhatsApp JID: {chat_id!r}")
    return m.group(1)
