"""Connect-time IP pinning for worker-originated HTTP — the DNS-rebinding SSRF fix.

``is_safe_url`` pre-flight checks resolve a hostname, validate the IPs, then hand
the *hostname* back to httpx, which re-resolves it at connect time. That is a
resolve-then-connect TOCTOU: an attacker-controlled DNS server with TTL=0 can
return a public IP for the check and a private IP for the connection.
:class:`PinnedTransport` closes it at the transport layer: resolve ONCE, validate
every returned IP, pin the connection to one of them, and keep SNI + certificate
verification on the original hostname. Because the transport runs on every
request — including each redirect hop — the guarantee holds by construction
rather than by each call site remembering to pre-flight a URL string.

:func:`resolve_pinned_ip` is the promoted primitive from the sandbox secret
egress proxy (:mod:`aios.sandbox.secret_egress_proxy`), which re-imports it —
the egress proxy remains the security-critical consumer, so its fail-closed /
IPv4-preferred semantics are authoritative.

NOTE: ``aios.tools.url_safety`` is imported lazily inside ``resolve_pinned_ip``
(the only user) — not here. This module is imported by both ``aios.tools`` and
``aios.mcp``; a module-level import of the ``aios.tools`` package would make its
eager ``__init__`` part of this module's import graph and cycle. Deferring to
call time keeps the import graph acyclic (same precedent as the egress proxy).
"""

from __future__ import annotations

import asyncio
import socket
from typing import Any

import httpx


async def _resolve_addrinfos(host: str, port: int) -> list[tuple[Any, ...]]:
    """Async DNS resolution seam (monkeypatched in tests)."""
    loop = asyncio.get_running_loop()
    return list(await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM))


async def resolve_pinned_ip(host: str, port: int) -> str | None:
    """Re-resolve ``host`` at connect time and return ONE safe IP to pin the
    connection to, or ``None`` if it must not be reached.

    Fail-closed: a known-internal name, a resolution failure, an empty
    result, or ANY resolved IP in a blocked range (loopback / link-local /
    RFC-1918 / ULA / reserved / multicast / unspecified / CGNAT / metadata)
    returns ``None``. This is the runtime, rebinding-resistant complement to
    #872's create-time IP-literal rejection. Prefers an IPv4 address (the
    worker is IPv4-only; a first-returned global IPv6 with no route would
    502 a reachable host otherwise).
    """
    from aios.tools import url_safety  # deferred — see the import-graph note at module top

    if url_safety.is_blocked_hostname(host):
        return None
    try:
        infos = await _resolve_addrinfos(host, port)
    except OSError:
        return None
    ips = [str(info[4][0]) for info in infos]
    if not ips or any(url_safety.is_blocked_ip(ip) for ip in ips):
        return None
    return next((ip for ip in ips if ":" not in ip), ips[0])


class PinnedTransport(httpx.AsyncBaseTransport):
    """An httpx transport that pins each request's connection to a safe resolved IP.

    Wraps an inner transport (composition, not subclassing): every request's host
    is resolved-and-validated via :func:`resolve_pinned_ip`, the URL is rewritten
    to the pinned IP, and for https the ``sni_hostname`` extension keeps SNI +
    certificate verification on the real hostname. A host that resolves to any
    blocked IP raises :class:`httpx.ConnectError` (an ``httpx.HTTPError``, so it
    flows into callers' existing transport-error handling) without the inner
    transport ever being reached.

    ``allow_hosts`` (operator-controlled, e.g.
    ``Settings.oauth_allow_insecure_host_set``) bypasses pinning entirely for
    dev-internal hosts; entries match the bare host or ``host:port`` form,
    mirroring ``_guard_url``.

    The default inner transport disables keepalive
    (``max_keepalive_connections=0``): pooled connections key on IP-origin only
    and silently ignore a later request's ``sni_hostname`` on reuse, so two
    hosts sharing a pinned IP could ride one another's verified TLS session.
    """

    def __init__(
        self,
        *,
        allow_hosts: frozenset[str] = frozenset(),
        inner: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._allow_hosts = allow_hosts
        self._inner = (
            inner
            if inner is not None
            else httpx.AsyncHTTPTransport(limits=httpx.Limits(max_keepalive_connections=0))
        )

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        port = request.url.port or (443 if request.url.scheme == "https" else 80)
        if host in self._allow_hosts or f"{host}:{port}" in self._allow_hosts:
            return await self._inner.handle_async_request(request)
        pinned = await resolve_pinned_ip(host, port)
        if pinned is None:
            raise httpx.ConnectError(f"blocked: {host} resolves to a private/internal address")
        # Rewrite only the connect target; httpx brackets an IPv6 literal itself,
        # and the Host header was frozen from the original URL at Request
        # construction, so it still names the real host.
        request.url = request.url.copy_with(host=pinned)
        if request.url.scheme == "https":
            # SNI + certificate verification stay on the real hostname.
            request.extensions = {**request.extensions, "sni_hostname": host}
        return await self._inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self._inner.aclose()
