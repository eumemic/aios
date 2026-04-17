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


# URL-safe base64 alphabet (includes '=' padding). Group IDs only.
_URLSAFE_B64_RE = re.compile(r"\A[A-Za-z0-9_-]+=*\Z")


def decode_chat_id(chat_id: str) -> tuple[ChatType, str]:
    """Validate and classify a chat_id, returning (chat_type, raw_for_cli).

    The only accepted forms are the bare chat_id from a bound channel's
    trailing path segment:

    * **DM**: a 36-char UUID with dashes (e.g. ``6c21718f-f095-483f-8cd6-610137d581aa``) —
      the counterparty's ACI UUID.
    * **Group**: URL-safe base64 (``A-Za-z0-9_-=``) — the last segment of
      the bound channel address.

    Do NOT pass the full channel path (``signal/<bot>/<chat_id>``) or any
    transformation of it. Pass only the trailing chat_id segment.

    Raises :class:`ValueError` with a self-describing message on invalid
    input, surfaced to the caller (and the model) as a tool error.
    """
    if is_dm_chat_id(chat_id):
        return "dm", chat_id
    if _URLSAFE_B64_RE.match(chat_id):
        return "group", chat_id.replace("-", "+").replace("_", "/")
    # Neither a UUID nor URL-safe base64. Explain both accepted shapes so
    # the model can correct without guessing.
    hint = ""
    if chat_id.startswith("signal/"):
        suffix = chat_id.rsplit("/", 1)[-1]
        hint = (
            f" Looks like you passed the full channel path — "
            f"use only the trailing segment ({suffix!r})."
        )
    raise ValueError(
        "chat_id must be either a 36-char UUID with dashes (for a DM — "
        "the counterparty's ACI UUID) or URL-safe base64 (for a group — "
        "alphabet A-Z a-z 0-9 '_' '-', with optional '=' padding). "
        f"Got {chat_id!r}.{hint}"
    )
