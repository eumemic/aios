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

# E.164: 7-15 digits, optional leading "+".  Word-boundary lookarounds
# so an email-address-like "user@example.com" isn't grabbed.
_MENTION_RE = re.compile(r"(?<![\w@])@(\+?\d{7,15})(?![\w])")


def encode_mentions(text: str) -> tuple[str, list[str]]:
    """Extract ``@<E.164>`` mentions and return ``(text, mentioned_jids)``.

    The text is returned unchanged; the JID list is in
    left-to-right order of first appearance, deduped.  Empty list
    means no mentions were detected (DMs or text with no @-tags).
    """
    seen: set[str] = set()
    jids: list[str] = []
    for m in _MENTION_RE.finditer(text):
        digits = m.group(1).lstrip("+")
        jid = f"{digits}@s.whatsapp.net"
        if jid not in seen:
            seen.add(jid)
            jids.append(jid)
    return text, jids
