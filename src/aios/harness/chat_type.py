"""Connector-agnostic ``chat_type`` derivation from a channel address (#1613).

``chat_type`` (``dm`` vs ``group``) is a *pure function of the channel
address* — it needs no storage to filter on. A channel is addressed as
``<connector>/<account>/<chat_id>`` (e.g. ``signal/<bot_uuid>/<chat_id>``,
``telegram/<bot>/<chat_id>``). The trailing ``<chat_id>`` segment classifies
the conversation:

* **DM**: a 36-char UUID with dashes (signal: the counterparty's ACI UUID;
  telegram: a positive numeric peer id).
* **Group**: URL-safe base64 (signal group id) or a negative numeric id
  (telegram supergroup/group).

This mirrors signal ``addressing.decode_chat_id``'s UUID-vs-base64 test
without importing the connector package (which is a separate distribution).
The observer read-plane applies it as a *post-filter* on the already-stored
``channel`` string — zero migration. If profiling later shows the post-filter
is too coarse, ``chat_type`` can be promoted to a stamped column.
"""

from __future__ import annotations

import re
from typing import Literal

ChatType = Literal["dm", "group"]

# A canonical UUID with dashes — signal DMs use the counterparty's ACI UUID.
_DM_UUID_RE = re.compile(
    r"\A[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\Z"
)

# URL-safe base64 (signal group ids are mapped to this in the channel path).
_URLSAFE_B64_RE = re.compile(r"\A[A-Za-z0-9_-]+=*\Z")


def chat_type_of(channel: str | None) -> ChatType | None:
    """Classify a channel address as ``"dm"`` / ``"group"`` / ``None``.

    Accepts either a full channel path (``<connector>/<account>/<chat_id>``)
    or a bare trailing ``<chat_id>`` segment; the classification looks only at
    the trailing segment. Returns ``None`` for an empty/None input or a chat_id
    that matches neither shape (so an unrecognized address never falsely
    claims a chat_type).

    Connector-agnostic by design: it mirrors signal ``decode_chat_id``'s
    UUID-vs-base64 test and additionally handles telegram numeric ids
    (negative ⇒ group, non-negative ⇒ dm).
    """
    if not channel:
        return None
    chat_id = channel.rsplit("/", 1)[-1]
    if not chat_id:
        return None
    # Signal DM: ACI UUID.
    if _DM_UUID_RE.match(chat_id):
        return "dm"
    # Telegram numeric ids: negative ⇒ group/supergroup, non-negative ⇒ dm.
    if re.fullmatch(r"-?\d+", chat_id):
        return "group" if chat_id.startswith("-") else "dm"
    # Signal group: URL-safe base64.
    if _URLSAFE_B64_RE.match(chat_id):
        return "group"
    return None
