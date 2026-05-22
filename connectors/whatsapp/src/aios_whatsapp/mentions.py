"""Encode and decode WhatsApp mentions.

WhatsApp's mention wire format is simpler than Signal's: the message
body carries an ``@<phone>`` literal (no U+FFFC placeholder, no
UTF-16 offset math) and a parallel ``ContextInfo.MentionedJID`` list
of full WhatsApp JIDs.  WhatsApp clients render the literal as a
pill when the JID resolves to a chat participant.

Outbound: scan the text for ``@<E.164>`` patterns, normalize each
match to a ``<digits>@s.whatsapp.net`` JID, and return the JID list
alongside the (unchanged) text.  The encoder doesn't validate
against the chat's actual member roster — WhatsApp's server falls
through unrendered when a mentioned JID isn't a participant, which
is the same user-facing outcome as not mentioning at all.

Inbound: the daemon hands us ``mentioned_jid: [str]``; we surface
them as ``{"jid": ..., "name": ...}`` dicts in
``metadata["mentions"]`` so the harness's existing renderer can
display them via the ``jid`` field (extension landed in PR 5).
"""

from __future__ import annotations

import re

# Mention syntax the model can write:
#   @+15551234567                          (E.164 with +)
#   @15551234567                           (bare digits, 7-15)
#   @15551234567@s.whatsapp.net            (full phone JID; round-trip path)
#   @98765@lid                             (LID JID; round-trip for @lid peers)
#
# Lookbehind uses an EXPLICIT ASCII negated class instead of ``\w``
# so the @ inside italic markers (``_@+15551234567_``) AND URL paths
# (``https://chat.example/u/@+15551234567``) STILL work / get rejected
# as intended:
#
# * ``_`` IS in ``\w`` (Python re default), so ``(?<![\w])`` would
#   silently drop italic-wrapped mentions.  The explicit class
#   excludes ``_``, letting the italic-wrapped @ match.
# * URL boundaries (``/``, ``.``) are NOT in ``\w``, so the previous
#   regex would harvest @-tags embedded in URLs.  Adding ``/`` and
#   ``.`` to the rejected set blocks URL-embedded harvesting.
#
# ``\d`` is restricted to ASCII digits via ``[0-9]`` to avoid
# Arabic-Indic / fullwidth / Devanagari digit shapes that would
# produce non-ASCII JIDs WhatsApp's server rejects.
_MENTION_RE = re.compile(
    r"(?<![A-Za-z0-9@/.])"
    r"@(?P<jid>"
    r"\+?[0-9]{7,15}"
    r"|[0-9]+@(?:s\.whatsapp\.net|lid)"
    r")"
    r"(?![A-Za-z0-9])"
)


def encode_mentions(text: str) -> tuple[str, list[str]]:
    """Extract WhatsApp mentions and return ``(text, mentioned_jids)``.

    The text is returned unchanged; the JID list is in left-to-right
    order of first appearance, deduped.  Empty list means no
    mentions were detected (DMs or text with no @-tags).
    """
    seen: set[str] = set()
    jids: list[str] = []
    for m in _MENTION_RE.finditer(text):
        raw = m.group("jid")
        if "@" in raw:
            # Already a full JID — accept verbatim.  Used for LID
            # round-trip and for models that copy the JID directly
            # out of inbound metadata.
            jid = raw
        else:
            digits = raw.lstrip("+")
            jid = f"{digits}@s.whatsapp.net"
        if jid not in seen:
            seen.add(jid)
            jids.append(jid)
    return text, jids
