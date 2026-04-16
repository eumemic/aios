"""Chat-ID encoding between signal-cli and aios channel addresses.

aios addresses a channel as ``signal/<bot_uuid>/<chat_id>`` where ``chat_id``
must be URL-path-safe. signal-cli uses:

- **DM**: the counterparty's ACI UUID (e.g. ``aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee``).
  Already URL-safe.
- **Group**: standard base64 with ``+``, ``/``, ``=``. We map to URL-safe
  base64 (``-``, ``_``, keep ``=`` padding) so the ``/`` doesn't break the
  channel path.

Round-trip is strict: ``decode_chat_id(encode_chat_id(raw, kind)) == (kind, raw)``.
"""

from __future__ import annotations

import re
from typing import Literal

ChatType = Literal["dm", "group"]

_DM_RE = re.compile(
    r"\A[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\Z"
)


def is_dm_chat_id(s: str) -> bool:
    """True iff ``s`` is a canonical 36-char UUID-with-dashes."""
    return _DM_RE.match(s) is not None


def encode_chat_id(raw: str, chat_type: ChatType) -> str:
    """Encode a signal-cli chat identifier for use in an aios channel path.

    DMs pass through; group IDs are converted from standard base64 to URL-safe.
    """
    if chat_type == "dm":
        if not is_dm_chat_id(raw):
            raise ValueError(f"not a canonical UUID: {raw!r}")
        return raw
    return raw.replace("+", "-").replace("/", "_")


def decode_chat_id(chat_id: str) -> tuple[ChatType, str]:
    """Invert :func:`encode_chat_id`.

    Detection: 36-char canonical UUIDs are DMs, everything else is a group.
    Group IDs are reversed from URL-safe back to standard base64 so they can
    be handed to signal-cli as ``groupId``.
    """
    if is_dm_chat_id(chat_id):
        return "dm", chat_id
    return "group", chat_id.replace("-", "+").replace("_", "/")
