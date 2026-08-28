"""Navigation guard: agent-driven navigation goes to public http(s) only.

Scheme + resolved-address checks run before ``page.goto`` and again against
the post-commit final URL (redirect laundering). The address check is
userspace and navigate-only — subresource fetches don't pass through here;
the L3 deny-internal egress on the browser network is the boundary that
holds regardless. ``allow_private_nav`` (the hermetic-test knob, read once at
daemon start) bypasses only the address check, never the scheme check.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from urllib.parse import urlparse

from aios_browser_driver.errors import ActionError

_SCHEMES = frozenset({"http", "https"})


def _blocked(message: str) -> ActionError:
    return ActionError("navigation_failed", message)


async def check_url(url: str, *, allow_private: bool) -> None:
    """Raise ``ActionError(navigation_failed)`` unless ``url`` is public http(s)."""
    parsed = urlparse(url)
    if parsed.scheme.lower() not in _SCHEMES:
        raise _blocked(f"only http(s) URLs can be opened, not {parsed.scheme or 'a relative'} URLs")
    host = parsed.hostname
    if not host:
        raise _blocked("the URL has no host")
    if allow_private:
        return
    port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise _blocked(f"could not resolve {host}: {exc}") from exc
    if not infos:
        raise _blocked(f"could not resolve {host}")
    for _family, _type, _proto, _canon, sockaddr in infos:
        # IPv6 link-local addresses come back with a %scope suffix.
        addr = str(sockaddr[0]).split("%", 1)[0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError as exc:
            raise _blocked(f"{host} resolved to an unparseable address {addr!r}") from exc
        if not ip.is_global:
            raise _blocked(f"{host} resolves to a non-public address ({ip}); refusing to open it")
