"""Twilio ``X-Twilio-Signature`` verification — verify-before-parse.

Lifted from openclaw ``webhook-security.ts``
(``validateTwilioSignature`` / ``buildTwilioDataToSign`` /
``reconstructWebhookUrl``) and adapted to the connector's
route-then-verify shape (design §3.2, §5.1, §5.4).

The algorithm Twilio specifies:

1. Take the full URL Twilio POSTed to (scheme + host + **port** for SMS
   over HTTPS — Twilio drops the port only for Voice).
2. If the request is ``application/x-www-form-urlencoded``, append every
   POST parameter sorted lexicographically by key, concatenating
   ``name + value`` with no separator.
3. ``HMAC-SHA1(auth_token, data)`` and base64-encode.
4. Constant-time-compare against the ``X-Twilio-Signature`` header.

This module is **pure** — it does no network I/O and no parsing of the
raw body beyond what's needed to build the data-to-sign. The handler
must read the **raw** body and route by the normalized ``To`` number to
the connection's ``auth_token`` *before* calling :func:`is_valid`, so a
forged or misrouted request fails closed (design §3.2).

Security invariants (design §5.1):

* **No skip-verification-when-no-token fallback** (the
  GHSA-4hg8-92x6-h2f3 vuln class). A missing token is a programming
  error here — callers route-by-``To`` first and treat an absent token
  as a cold-start *transient 5xx*, never a verify-skip.
* **Constant-time compare** of the candidate signature.
* The signing URL **prefers the operator-configured public base URL**
  over header reconstruction (design §5.4); see
  :func:`reconstruct_signed_url`.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
from urllib.parse import urlsplit, urlunsplit

__all__ = [
    "build_data_to_sign",
    "compute_signature",
    "is_valid",
    "reconstruct_signed_url",
]


def build_data_to_sign(url: str, params: dict[str, str]) -> str:
    """Concatenate ``url`` + ``sortedByKey(name + value)``.

    ``params`` is the decoded ``application/x-www-form-urlencoded`` POST
    body as a flat ``{name: value}`` mapping. Twilio sorts by parameter
    name and concatenates ``name`` immediately followed by ``value`` with
    no delimiters, appended to the full signed URL.
    """
    data = url
    for key in sorted(params):
        data += key + params[key]
    return data


def compute_signature(auth_token: str, url: str, params: dict[str, str]) -> str:
    """Return the base64 ``HMAC-SHA1`` Twilio expects for this request."""
    data = build_data_to_sign(url, params)
    mac = hmac.new(auth_token.encode("utf-8"), data.encode("utf-8"), hashlib.sha1)
    return base64.b64encode(mac.digest()).decode("ascii")


def is_valid(
    auth_token: str,
    url: str,
    params: dict[str, str],
    signature: str | None,
) -> bool:
    """Constant-time-verify ``signature`` against the recomputed HMAC.

    Returns ``False`` (never raises) on a missing/empty signature so the
    handler maps the single negative result to a fail-closed ``403``. A
    *missing token* is the caller's responsibility (cold-start → 5xx);
    an empty ``auth_token`` here can never produce a valid signature, so
    it also fails closed.
    """
    if not signature:
        return False
    expected = compute_signature(auth_token, url, params)
    # constant-time compare — never short-circuit on first mismatch
    return hmac.compare_digest(expected, signature)


def reconstruct_signed_url(
    *,
    configured_base_url: str | None,
    path: str,
    query: str = "",
    forwarded_proto: str | None = None,
    forwarded_host: str | None = None,
    keep_port: bool = True,
) -> str:
    """Reconstruct the exact public URL Twilio signed.

    **Prefer the operator-configured public base URL** as the canonical
    signing origin (design §5.4): it is the URL the operator registered
    with Twilio, so it is by construction the one Twilio signed. Only
    fall back to ``X-Forwarded-Proto`` / ``X-Forwarded-Host`` when no
    configured base exists — and that fallback is the host-header
    injection surface the design hardens elsewhere (require a non-empty
    allowlist + socket-peer-IP proxy match before trusting forwarded
    headers).

    For SMS over HTTPS the **port is kept** when present (Twilio drops
    the port only for Voice). When ``keep_port`` is ``False`` the port is
    stripped from the reconstructed origin.

    ``path`` is the request path (e.g. ``/twilio/inbound``); ``query`` is
    the raw query string without the leading ``?`` (Twilio includes GET
    query params in the signed URL, empty for our POST routes).
    """
    if configured_base_url:
        base = configured_base_url.rstrip("/")
        split = urlsplit(base)
        scheme = split.scheme or "https"
        netloc = split.netloc
        # base may itself carry a path prefix (e.g. behind a sub-path
        # proxy); preserve it ahead of the route path.
        prefix = split.path.rstrip("/")
        full_path = prefix + path
    else:
        scheme = (forwarded_proto or "https").strip()
        netloc = (forwarded_host or "").strip()
        full_path = path

    if not keep_port and "@" not in netloc:
        host = netloc.rsplit(":", 1)[0] if ":" in netloc else netloc
        netloc = host

    return urlunsplit((scheme, netloc, full_path, query, ""))
