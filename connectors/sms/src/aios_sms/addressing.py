"""E.164 addressing helpers for the SMS connector.

A **single** phone-number normalizer (lifted from whatsapp
``normalize_phone``) applied **symmetrically** at every boundary where a
phone number crosses into internal state:

* the webhook ``From`` / ``To`` parse (inbound routing key + verify-key
  lookup),
* ``serve_connection`` storing its connection's ``from_number`` in the
  shared demux map,
* any management-handler ``external_account_id`` lookup (deferred slice).

If the store side and the lookup side don't canonicalize identically,
operator formatting drift (``+1 800 555-1234`` vs ``+18005551234`` vs
``18005551234``) silently misroutes — the signal ``account.strip()``
message-loss lesson (design §3.3). The verify key and the routing key
are the *same* signed ``To`` value, so a normalizer that drifts between
store and lookup fails closed against the wrong connection's token
rather than silently cross-routing (design §3.2 step 2).
"""

from __future__ import annotations

__all__ = ["normalize_e164"]


def normalize_e164(phone: str) -> str:
    """Strip whitespace + common separators and ensure a leading ``+``.

    Mirrors whatsapp ``normalize_phone`` exactly so the two connectors
    canonicalize phone numbers the same way. Trivial formatting
    differences (``+15551112222`` / ``15551112222`` / ``+1 555 111-2222``)
    all collapse to ``+15551112222``.

    This is intentionally minimal — it does **not** validate country
    codes or length, because the only invariant that matters for routing
    is that store and lookup produce byte-identical output for the same
    logical number.
    """
    s = phone.strip().replace("-", "").replace(" ", "")
    if s and not s.startswith("+"):
        s = "+" + s
    return s
