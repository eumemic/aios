"""Chat-ID encoding between signal-cli and aios channel addresses.

aios addresses a channel as ``signal/<bot_uuid>/<chat_id>``. signal-cli uses
UUIDs for DMs (URL-safe as-is) and standard base64 for group IDs (``+``,
``/``, ``=``). We map group IDs to URL-safe base64 (``-``, ``_``, keep ``=``)
so the ``/`` doesn't break the channel path, and reverse the mapping before
handing a ``groupId`` back to signal-cli.
"""

from __future__ import annotations

import re
from typing import Literal

ChatType = Literal["dm", "group"]

_DM_RE = re.compile(
    r"\A[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\Z"
)


def is_dm_chat_id(s: str) -> bool:
    return _DM_RE.match(s) is not None


def encode_chat_id(raw: str, chat_type: ChatType) -> str:
    if chat_type == "dm":
        if not is_dm_chat_id(raw):
            raise ValueError(f"not a canonical UUID: {raw!r}")
        return raw
    return raw.replace("+", "-").replace("/", "_")


def decode_chat_id(chat_id: str) -> tuple[ChatType, str]:
    if is_dm_chat_id(chat_id):
        return "dm", chat_id
    return "group", chat_id.replace("-", "+").replace("_", "/")
