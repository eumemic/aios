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

__all__ = ["digits_only", "focal_channel", "normalize_e164", "same_number"]


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


def digits_only(phone: str) -> str:
    """Return just the digits of a phone number.

    The demux map keys on :func:`normalize_e164`, but a **digits-only**
    compare is the last-resort equality used when comparing two numbers
    that may differ only by a leading ``+`` or stray punctuation. Keeping
    this distinct from the routing key documents that the routing key is
    the normalized form, while equality is digits-only (design §3.3).
    """
    return "".join(ch for ch in phone if ch.isdigit())


def same_number(a: str, b: str) -> bool:
    """Digits-only equality of two phone numbers."""
    return digits_only(a) == digits_only(b)


def focal_channel(connector: str, external_account_id: str, chat_id: str) -> str:
    """``<connector>/<external_account_id>/<chat_id>`` — e.g.
    ``sms/+18005551234/+14155550000``.

    ``external_account_id`` is the AIOS-owned Twilio number; ``chat_id``
    is the peer. Both are stored slash-free normalized E.164 so they
    satisfy ``ConnectionCreate._no_slash`` and re-parse cleanly into the
    runtime's injected kwargs.
    """
    return f"{connector}/{external_account_id}/{chat_id}"
