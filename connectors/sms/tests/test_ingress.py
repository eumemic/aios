"""Forwarded-header ingress trust gate (design §3.2 step 3, §5.4).

The fallback signing-URL reconstruction is the host-header-injection
surface; these pin the fail-closed properties of the gate that guards it.
"""

from __future__ import annotations

from aios_sms.ingress import (
    IngressPolicy,
    host_is_allowed,
    is_trusted_proxy,
    valid_forwarded_host,
)

ALLOWED = frozenset({"sms.example.com", "sms2.example.com"})
PROXIES = frozenset({"10.0.0.0/8", "192.0.2.7"})


# ── valid_forwarded_host ──────────────────────────────────────────────


def test_valid_forwarded_host_accepts_plain_hostname() -> None:
    assert valid_forwarded_host("sms.example.com")


def test_valid_forwarded_host_accepts_host_with_port() -> None:
    assert valid_forwarded_host("sms.example.com:8443")


def test_valid_forwarded_host_rejects_empty() -> None:
    assert not valid_forwarded_host("")


def test_valid_forwarded_host_rejects_userinfo_at_sign() -> None:
    # ``@`` userinfo smuggling — the design explicitly rejects it.
    assert not valid_forwarded_host("sms.example.com@evil.example")
    assert not valid_forwarded_host("evil.example@sms.example.com")


def test_valid_forwarded_host_rejects_non_rfc1123() -> None:
    assert not valid_forwarded_host("-bad.example")
    assert not valid_forwarded_host("bad-.example")
    assert not valid_forwarded_host("has spaces.example")


def test_valid_forwarded_host_rejects_out_of_range_port() -> None:
    assert not valid_forwarded_host("sms.example.com:0")
    assert not valid_forwarded_host("sms.example.com:99999")


# ── host_is_allowed ───────────────────────────────────────────────────


def test_host_is_allowed_matches_on_hostname_ignoring_port() -> None:
    assert host_is_allowed("sms.example.com", ALLOWED)
    assert host_is_allowed("sms.example.com:8443", ALLOWED)


def test_host_is_allowed_rejects_unlisted_host() -> None:
    assert not host_is_allowed("attacker.example", ALLOWED)


def test_host_is_allowed_empty_allowlist_fails_closed() -> None:
    # An empty allowlist disables the fallback entirely.
    assert not host_is_allowed("sms.example.com", frozenset())


# ── is_trusted_proxy ──────────────────────────────────────────────────


def test_is_trusted_proxy_matches_cidr() -> None:
    assert is_trusted_proxy("10.4.5.6", PROXIES)


def test_is_trusted_proxy_matches_bare_ip() -> None:
    assert is_trusted_proxy("192.0.2.7", PROXIES)


def test_is_trusted_proxy_rejects_outside_set() -> None:
    assert not is_trusted_proxy("203.0.113.9", PROXIES)


def test_is_trusted_proxy_empty_set_fails_closed() -> None:
    assert not is_trusted_proxy("10.0.0.1", frozenset())


def test_is_trusted_proxy_missing_peer_fails_closed() -> None:
    assert not is_trusted_proxy(None, PROXIES)


def test_is_trusted_proxy_unparseable_peer_fails_closed() -> None:
    assert not is_trusted_proxy("not-an-ip", PROXIES)


# ── IngressPolicy.trusts_forwarded (all three gates) ──────────────────


def test_policy_trusts_when_all_three_gates_pass() -> None:
    policy = IngressPolicy(allowed_hosts=ALLOWED, trusted_proxies=PROXIES)
    assert policy.trusts_forwarded(peer_ip="10.0.0.9", forwarded_host="sms.example.com")


def test_policy_refuses_untrusted_proxy() -> None:
    policy = IngressPolicy(allowed_hosts=ALLOWED, trusted_proxies=PROXIES)
    assert not policy.trusts_forwarded(peer_ip="203.0.113.1", forwarded_host="sms.example.com")


def test_policy_refuses_unlisted_host_even_from_trusted_proxy() -> None:
    policy = IngressPolicy(allowed_hosts=ALLOWED, trusted_proxies=PROXIES)
    assert not policy.trusts_forwarded(peer_ip="10.0.0.9", forwarded_host="attacker.example")


def test_policy_refuses_at_sign_host_from_trusted_proxy() -> None:
    policy = IngressPolicy(allowed_hosts=ALLOWED, trusted_proxies=PROXIES)
    assert not policy.trusts_forwarded(
        peer_ip="10.0.0.9", forwarded_host="sms.example.com@evil.example"
    )


def test_policy_empty_allowlist_disables_fallback() -> None:
    policy = IngressPolicy(allowed_hosts=frozenset(), trusted_proxies=PROXIES)
    assert not policy.trusts_forwarded(peer_ip="10.0.0.9", forwarded_host="sms.example.com")


def test_policy_empty_proxy_set_disables_fallback() -> None:
    policy = IngressPolicy(allowed_hosts=ALLOWED, trusted_proxies=frozenset())
    assert not policy.trusts_forwarded(peer_ip="10.0.0.9", forwarded_host="sms.example.com")


def test_policy_refuses_missing_forwarded_host() -> None:
    policy = IngressPolicy(allowed_hosts=ALLOWED, trusted_proxies=PROXIES)
    assert not policy.trusts_forwarded(peer_ip="10.0.0.9", forwarded_host=None)
