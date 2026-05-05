"""Encode ``@<uuid_prefix>`` mention syntax for outbound Signal messages.

Signal represents mentions in the wire protocol as a U+FFFC placeholder
in the message body plus a parallel ``mentions`` array of
``"start:length:uuid"`` entries with UTF-16 code-unit offsets.  The
flow has two stages because markdown stripping happens between mention
substitution and the final outbound message:

1. :func:`encode_mentions` — replace ``@<hex>`` syntax with U+FFFC and
   return the encoded text plus a left-to-right list of resolved UUIDs.
2. :func:`build_mention_strings` — after any further text mutation
   (markdown stripping), pair each remaining U+FFFC with its UUID and
   compute UTF-16 offsets against the *final* message body.

Group-only by design: the resolver matches each ``@<hex>`` candidate
against the UUIDs of the current group's members.  In a DM the caller
passes an empty ``member_uuids`` list and this module is a no-op.
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
    """Replace resolved ``@<hex>`` syntax with U+FFFC placeholders.

    Returns ``(encoded_text, ordered_uuids)`` where ``ordered_uuids[i]``
    is the UUID that the i-th U+FFFC in ``encoded_text`` (left-to-right)
    represents.  The caller pairs each placeholder with its UUID via
    :func:`build_mention_strings` *after* any subsequent text mutation
    (e.g. markdown stripping) — markdown removal preserves placeholder
    order but shifts character offsets, so offsets must be computed on
    the final message body.

    Pre-existing U+FFFC characters in ``text`` are stripped before
    encoding.  Outbound Signal sends use U+FFFC exclusively for
    mentions; an orphan placeholder would either crash the offset/UUID
    pairing or be rejected by signal-cli.
    """
    if MENTION_PLACEHOLDER in text:
        text = text.replace(MENTION_PLACEHOLDER, "")
    if not member_uuids or "@" not in text:
        return text, []

    resolved: list[tuple[int, int, str]] = []  # (start, end, full_uuid)
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

    return encoded, [uuid for _, _, uuid in resolved]


def build_mention_strings(message: str, ordered_uuids: list[str]) -> list[str]:
    """Pair each U+FFFC in ``message`` with the next UUID and emit
    ``"<utf16_start>:1:<uuid>"`` strings.

    ``message`` is the final outbound text (already markdown-stripped);
    ``ordered_uuids`` comes from :func:`encode_mentions`.  Caller
    invariant: ``message`` contains exactly ``len(ordered_uuids)``
    placeholders, in the same left-to-right order as the source
    ``@<hex>`` matches.
    """
    if not ordered_uuids:
        return []
    uuid_iter = iter(ordered_uuids)
    return [
        f"{codepoint_to_utf16_offset(message, i)}:1:{next(uuid_iter)}"
        for i, ch in enumerate(message)
        if ch == MENTION_PLACEHOLDER
    ]
