"""Ingress trust gate for the forwarded-header signing-URL fallback.

The webhook listener prefers the operator-configured public base URL as
the canonical ``X-Twilio-Signature`` signing URL (design §5.4). When **no**
configured base exists it must fall back to reconstructing the URL from
``X-Forwarded-Proto`` / ``X-Forwarded-Host`` — and that fallback is the
host-header-injection surface. The design hardens it (§3.2 step 3, §5.4):

* require a **non-empty ``allowedHosts`` allowlist** — never reconstruct
  from an arbitrary forwarded host;
* require a **socket-peer-IP match against the trusted-proxy set** — the
  forwarded headers are only trusted when the immediate TCP peer is a
  known proxy (a header-derived check would be circular);
* **reject ``@`` in the host** (userinfo smuggling) and validate the host
  as an RFC-1123 hostname (optionally ``:port``).

Because the HMAC key is the per-connection ``auth_token`` selected by the
*signed* ``To`` number, controlling the URL component alone cannot forge a
signature: a misconfigured or hostile forwarded host fails **closed**
(availability), never open. This gate makes that failure explicit and
auditable rather than relying on the HMAC alone to absorb a bad host.

This module is **pure** (no network, no aiohttp import) so it is unit
testable in isolation; the listener feeds it the socket-peer IP and the
forwarded headers.
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass

__all__ = ["IngressPolicy", "host_is_allowed", "is_trusted_proxy", "valid_forwarded_host"]

# RFC-1123 hostname label: letters/digits/hyphen, not starting/ending with
# a hyphen, 1-63 chars; labels joined by dots. We additionally accept an
# optional ``:port`` suffix because SMS over HTTPS keeps the port.
_LABEL = r"(?!-)[A-Za-z0-9-]{1,63}(?<!-)"
_HOSTNAME_RE = re.compile(rf"^{_LABEL}(?:\.{_LABEL})*$")


def _split_host_port(host: str) -> tuple[str, str | None]:
    """Split ``host[:port]`` → ``(host, port|None)``.

    Only splits on the **last** colon and only when the suffix is all
    digits, so IPv6 literals / malformed values don't get mangled into a
    spurious port. (We don't expect IPv6 Hosts from Twilio, but refusing
    to misparse them keeps the validator honest.)
    """
    if ":" in host:
        candidate_host, _, candidate_port = host.rpartition(":")
        if candidate_port.isdigit():
            return candidate_host, candidate_port
    return host, None


def valid_forwarded_host(host: str) -> bool:
    """Return whether ``host`` is a syntactically acceptable forwarded host.

    Rejects empty hosts, any ``@`` (userinfo smuggling), and anything that
    isn't an RFC-1123 hostname with an optional numeric ``:port`` suffix.
    """
    if not host or "@" in host:
        return False
    hostname, port = _split_host_port(host)
    if not hostname:
        return False
    if port is not None and not (0 < int(port) <= 65535):
        return False
    return bool(_HOSTNAME_RE.match(hostname))


def host_is_allowed(host: str, allowed_hosts: frozenset[str]) -> bool:
    """Return whether ``host`` (hostname, port-insensitive) is in the
    allowlist.

    The allowlist is matched on the **hostname** (the port is kept in the
    signed URL but pinned separately from config), so an operator lists
    ``sms.example.com`` once and isn't forced to enumerate ports.
    """
    if not allowed_hosts:
        # Empty allowlist ⇒ the forwarded-header fallback is disabled
        # entirely. Fail closed.
        return False
    hostname, _ = _split_host_port(host)
    return hostname in allowed_hosts


def is_trusted_proxy(peer_ip: str | None, trusted_proxies: frozenset[str]) -> bool:
    """Return whether the immediate socket peer is a trusted proxy.

    ``trusted_proxies`` may contain bare IPs or CIDR networks. An empty
    set ⇒ no proxy is trusted ⇒ the forwarded-header fallback is disabled
    (fail closed). A missing / unparseable ``peer_ip`` also fails closed.
    """
    if not trusted_proxies or not peer_ip:
        return False
    try:
        addr = ipaddress.ip_address(peer_ip)
    except ValueError:
        return False
    for entry in trusted_proxies:
        try:
            if "/" in entry:
                if addr in ipaddress.ip_network(entry, strict=False):
                    return True
            elif addr == ipaddress.ip_address(entry):
                return True
        except ValueError:
            continue
    return False


@dataclass(frozen=True, slots=True)
class IngressPolicy:
    """The forwarded-header trust policy for the listener.

    When ``public_base_url`` is set on the listener, this policy is never
    consulted (the configured base is the canonical signing URL). It only
    governs the fallback path.
    """

    allowed_hosts: frozenset[str]
    trusted_proxies: frozenset[str]

    def trusts_forwarded(self, *, peer_ip: str | None, forwarded_host: str | None) -> bool:
        """All three gates must pass to trust a forwarded host:

        1. the immediate socket peer is a trusted proxy,
        2. the forwarded host is a syntactically valid RFC-1123 host
           (no ``@``),
        3. the forwarded host is in the non-empty ``allowedHosts`` set.

        Any failure ⇒ the fallback is refused and the request fails closed.
        """
        if not is_trusted_proxy(peer_ip, self.trusted_proxies):
            return False
        if not forwarded_host or not valid_forwarded_host(forwarded_host):
            return False
        return host_is_allowed(forwarded_host, self.allowed_hosts)
