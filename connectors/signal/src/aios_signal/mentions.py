"""Encode ``@<uuid_prefix>`` mention syntax for outbound Signal messages.

Signal represents mentions in the wire protocol as a U+FFFC placeholder
in the message body plus a parallel ``mentions`` array of
``"start:length:uuid"`` entries.  This module turns agent-friendly text
of the form ``"hey @abcd1234, ping"`` into that wire form.

Group-only by design: the resolver matches each ``@<hex>`` candidate
against the UUIDs of the current group's members.  In a DM there is
only one possible counterparty, so callers pass an empty
``member_uuids`` list and this module is a no-op (the text passes
through unchanged).
"""

from __future__ import annotations

import re

from ._utf16 import codepoint_to_utf16_offset
from .parse import MENTION_PLACEHOLDER

# Min 8 hex chars OR a full dashed UUID.  8 hex is enough to disambiguate
# within typical group sizes.
_MENTION_RE = re.compile(
    r"@([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    r"|[0-9a-fA-F]{8,})",
    re.IGNORECASE,
)


def _resolve(candidate: str, member_uuids: list[str]) -> str | None:
    """Return the unique full UUID matching ``candidate``, or None.

    Strips dashes and lowercases both sides so a prefix like
    ``"abcd1234"`` matches ``"abcd1234-xxxx-..."``.  Returns None when
    zero or multiple members match — ambiguity is a no-op rather than
    a guess.
    """
    clean = candidate.lower().replace("-", "")
    matches = [u for u in member_uuids if u.lower().replace("-", "").startswith(clean)]
    return matches[0] if len(matches) == 1 else None


def encode_mentions(
    text: str,
    member_uuids: list[str],
) -> tuple[str, list[str]]:
    """Replace resolved ``@<hex>`` syntax with placeholders + mentions metadata.

    Returns ``(encoded_text, mentions)`` where ``mentions`` is a list of
    ``"<utf16_start>:1:<full_uuid>"`` strings (Signal's textStyles use
    UTF-16 code-unit offsets, and so do mentions).  Unresolved candidates
    are left in the text as-is so the agent can see they didn't land
    when the message arrives.
    """
    if not member_uuids or "@" not in text:
        return text, []

    resolved: list[tuple[int, int, str]] = []  # (start, end, full_uuid) in original text
    for m in _MENTION_RE.finditer(text):
        full_uuid = _resolve(m.group(1), member_uuids)
        if full_uuid is not None:
            resolved.append((m.start(), m.end(), full_uuid))

    if not resolved:
        return text, []

    # Splice right-to-left so earlier indices stay valid.
    encoded = text
    for start, end, _uuid in reversed(resolved):
        encoded = encoded[:start] + MENTION_PLACEHOLDER + encoded[end:]

    # Signal mention offsets are UTF-16 code units, not Python code points.
    mention_uuids = iter(uuid for _, _, uuid in resolved)
    mentions = [
        f"{codepoint_to_utf16_offset(encoded, i)}:1:{next(mention_uuids)}"
        for i, ch in enumerate(encoded)
        if ch == MENTION_PLACEHOLDER
    ]
    return encoded, mentions
