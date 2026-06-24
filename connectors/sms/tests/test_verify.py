"""Twilio ``X-Twilio-Signature`` verification — no-network unit tests.

Includes the canonical Twilio known-answer vector from their docs so the
algorithm (URL + sorted name+value concat, HMAC-SHA1, base64) is pinned
byte-for-byte, plus the fail-closed properties the security model
requires (design §5.1, §5.4).
"""

from __future__ import annotations

import base64
import hashlib
import hmac

from aios_sms.verify import (
    build_data_to_sign,
    compute_signature,
    is_valid,
    reconstruct_signed_url,
)

# Twilio's documented worked example
# (https://www.twilio.com/docs/usage/security#validating-requests).
_TWILIO_URL = "https://mycompany.com/myapp.php?foo=1&bar=2"
_TWILIO_TOKEN = "12345"
_TWILIO_PARAMS = {
    "CallSid": "CA1234567890ABCDE",
    "Caller": "+14158675309",
    "Digits": "1234",
    "From": "+14158675309",
    "To": "+18005551212",
}
# Expected signature for the above inputs (computed with the published
# Twilio algorithm; pins our implementation byte-for-byte).
_TWILIO_EXPECTED = "RSOYDt4T1cUTdK1PDd93/VVr8B8="


def test_build_data_to_sign_sorts_by_key_and_concats() -> None:
    data = build_data_to_sign("https://x/y", {"b": "2", "a": "1"})
    # url + a1 + b2 (sorted by key, name+value with no separators)
    assert data == "https://x/ya1b2"


def test_known_answer_vector_matches_twilio_docs() -> None:
    """Pin the algorithm against Twilio's published worked example."""
    sig = compute_signature(_TWILIO_TOKEN, _TWILIO_URL, _TWILIO_PARAMS)
    assert sig == _TWILIO_EXPECTED


def test_is_valid_accepts_correct_signature() -> None:
    assert is_valid(_TWILIO_TOKEN, _TWILIO_URL, _TWILIO_PARAMS, _TWILIO_EXPECTED)


def test_is_valid_rejects_tampered_param() -> None:
    """A signed request with one altered param fails closed — this is the
    route-then-verify correctness property: an attacker who alters ``To``
    (the routing key) cannot keep the signature valid (design §3.2)."""
    tampered = dict(_TWILIO_PARAMS, To="+19998887777")
    assert not is_valid(_TWILIO_TOKEN, _TWILIO_URL, tampered, _TWILIO_EXPECTED)


def test_is_valid_rejects_wrong_token() -> None:
    """A misroute verifies against the WRONG connection's token and fails
    closed — never a silent cross-connection accept (design §3.2)."""
    assert not is_valid("wrong-token", _TWILIO_URL, _TWILIO_PARAMS, _TWILIO_EXPECTED)


def test_is_valid_rejects_missing_signature() -> None:
    assert not is_valid(_TWILIO_TOKEN, _TWILIO_URL, _TWILIO_PARAMS, None)
    assert not is_valid(_TWILIO_TOKEN, _TWILIO_URL, _TWILIO_PARAMS, "")


def test_is_valid_rejects_empty_token() -> None:
    """No skip-verification-when-no-token fallback (GHSA-4hg8-92x6-h2f3):
    an empty token can never produce a valid signature."""
    assert not is_valid("", _TWILIO_URL, _TWILIO_PARAMS, _TWILIO_EXPECTED)


def test_roundtrip_with_local_token() -> None:
    """A signature we compute ourselves verifies — sanity that compute and
    verify use the same data-to-sign."""
    token = "super-secret-auth-token"
    url = "https://sms.example.com:8443/twilio/inbound"
    params = {"To": "+18005551234", "From": "+14155550000", "Body": "hi", "MessageSid": "SM1"}
    sig = compute_signature(token, url, params)
    assert is_valid(token, url, params, sig)
    # independent recomputation matches
    expected = base64.b64encode(
        hmac.new(token.encode(), build_data_to_sign(url, params).encode(), hashlib.sha1).digest()
    ).decode()
    assert sig == expected


# ── URL reconstruction (keep-port, prefer configured base) ────────────


def test_reconstruct_prefers_configured_base_and_keeps_port() -> None:
    url = reconstruct_signed_url(
        configured_base_url="https://sms.example.com:8443",
        path="/twilio/inbound",
        forwarded_proto="http",
        forwarded_host="attacker.example",
        keep_port=True,
    )
    # The configured base wins over forged X-Forwarded-* headers, and the
    # port is kept (SMS over HTTPS keeps the port).
    assert url == "https://sms.example.com:8443/twilio/inbound"


def test_reconstruct_configured_base_with_path_prefix() -> None:
    url = reconstruct_signed_url(
        configured_base_url="https://sms.example.com/hooks",
        path="/twilio/status",
    )
    assert url == "https://sms.example.com/hooks/twilio/status"


def test_reconstruct_falls_back_to_forwarded_headers() -> None:
    url = reconstruct_signed_url(
        configured_base_url=None,
        path="/twilio/inbound",
        forwarded_proto="https",
        forwarded_host="sms.example.com:8443",
        keep_port=True,
    )
    assert url == "https://sms.example.com:8443/twilio/inbound"


def test_reconstruct_can_drop_port_when_requested() -> None:
    url = reconstruct_signed_url(
        configured_base_url="https://sms.example.com:8443",
        path="/twilio/inbound",
        keep_port=False,
    )
    assert url == "https://sms.example.com/twilio/inbound"


def test_reconstruct_includes_query_string() -> None:
    url = reconstruct_signed_url(
        configured_base_url="https://sms.example.com",
        path="/twilio/inbound",
        query="foo=1&bar=2",
    )
    assert url == "https://sms.example.com/twilio/inbound?foo=1&bar=2"
