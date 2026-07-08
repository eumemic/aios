"""Unit tests for :class:`PinnedTransport` — the connect-time IP-pinning SSRF guard.

The DNS seam (``pinned_transport._resolve_addrinfos``) is monkeypatched so no
real lookups happen, and the inner transport is an ``httpx.MockTransport`` so no
real sockets are opened. The resolver primitive itself
(:func:`aios.pinned_transport.resolve_pinned_ip`) is covered by
``TestResolvePinnedIp`` in ``tests/unit/sandbox/test_secret_egress_proxy.py``;
these tests cover the transport behavior built on it.
"""

from __future__ import annotations

import socket
from collections.abc import Awaitable, Callable

import httpx
import pytest

from aios import pinned_transport
from aios.pinned_transport import PinnedTransport


def _addrinfos(*ips: str) -> Callable[[str, int], Awaitable[list[tuple[object, ...]]]]:
    """A fake ``_resolve_addrinfos`` returning the given IPs (v6 detected by colon)."""

    async def _resolve(host: str, port: int) -> list[tuple[object, ...]]:
        out: list[tuple[object, ...]] = []
        for ip in ips:
            fam = socket.AF_INET6 if ":" in ip else socket.AF_INET
            out.append((fam, socket.SOCK_STREAM, 0, "", (ip, port)))
        return out

    return _resolve


def _inner(seen: list[httpx.Request]) -> httpx.MockTransport:
    """An inner transport recording every request it is handed."""

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=b"ok")

    return httpx.MockTransport(handler)


class TestPinnedTransport:
    async def test_rebind_to_private_ip_is_blocked_before_inner(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The fix: a host that (re)resolves to a private IP at connect time is
        refused at the transport — the inner transport is NEVER reached, so no
        connection (and no request) can leak to the rebound address."""
        monkeypatch.setattr(pinned_transport, "_resolve_addrinfos", _addrinfos("10.0.0.1"))
        seen: list[httpx.Request] = []
        transport = PinnedTransport(inner=_inner(seen))
        request = httpx.Request("GET", "https://api.example.com/secret")
        with pytest.raises(httpx.ConnectError, match="private/internal"):
            await transport.handle_async_request(request)
        assert seen == []

    async def test_public_ip_is_pinned_with_sni_and_original_host_header(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(pinned_transport, "_resolve_addrinfos", _addrinfos("93.184.216.34"))
        seen: list[httpx.Request] = []
        transport = PinnedTransport(inner=_inner(seen))
        request = httpx.Request("GET", "https://api.example.com:8443/path?q=1")
        response = await transport.handle_async_request(request)
        assert response.status_code == 200
        (inner_req,) = seen
        # Connect target is the pinned IP; the port and request-target survive.
        assert inner_req.url.host == "93.184.216.34"
        assert inner_req.url.port == 8443
        assert inner_req.url.raw_path == b"/path?q=1"
        # SNI + cert verification stay on the real hostname, and the Host header
        # (frozen at Request construction) still names it — never the IP.
        assert inner_req.extensions["sni_hostname"] == "api.example.com"
        assert inner_req.headers["host"] == "api.example.com:8443"

    @pytest.mark.parametrize(
        "ip", ["127.0.0.1", "169.254.169.254", "10.0.0.1", "192.168.1.1", "fd00::1", "100.64.0.1"]
    )
    async def test_blocked_ip_raises_connect_error(
        self, ip: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(pinned_transport, "_resolve_addrinfos", _addrinfos(ip))
        seen: list[httpx.Request] = []
        transport = PinnedTransport(inner=_inner(seen))
        with pytest.raises(httpx.ConnectError):
            await transport.handle_async_request(httpx.Request("GET", "https://api.example.com/x"))
        assert seen == []

    async def test_ipv6_pin_renders_bracketed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        v6 = "2606:2800:220:1:248:1893:25c8:1946"
        monkeypatch.setattr(pinned_transport, "_resolve_addrinfos", _addrinfos(v6))
        seen: list[httpx.Request] = []
        transport = PinnedTransport(inner=_inner(seen))
        response = await transport.handle_async_request(
            httpx.Request("GET", "https://api.example.com/x")
        )
        assert response.status_code == 200
        (inner_req,) = seen
        assert inner_req.url.host == v6
        assert f"[{v6}]" in str(inner_req.url)  # httpx brackets the literal itself

    @pytest.mark.parametrize("entry", ["workspace-mcp", "workspace-mcp:8000"])
    async def test_allow_hosts_bypasses_pinning(
        self, entry: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both forms _guard_url accepts — bare host and host:port — delegate to
        the inner transport unpinned (no resolution, no sni_hostname)."""

        async def _boom(host: str, port: int) -> list[tuple[object, ...]]:
            raise AssertionError("an allow-listed host must not be resolved")

        monkeypatch.setattr(pinned_transport, "_resolve_addrinfos", _boom)
        seen: list[httpx.Request] = []
        transport = PinnedTransport(allow_hosts=frozenset({entry}), inner=_inner(seen))
        response = await transport.handle_async_request(
            httpx.Request("GET", "http://workspace-mcp:8000/mcp")
        )
        assert response.status_code == 200
        (inner_req,) = seen
        assert inner_req.url.host == "workspace-mcp"  # original host, unpinned
        assert "sni_hostname" not in inner_req.extensions

    async def test_plain_http_pins_without_sni(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pinned_transport, "_resolve_addrinfos", _addrinfos("93.184.216.34"))
        seen: list[httpx.Request] = []
        transport = PinnedTransport(inner=_inner(seen))
        response = await transport.handle_async_request(
            httpx.Request("GET", "http://api.example.com/x")
        )
        assert response.status_code == 200
        (inner_req,) = seen
        assert inner_req.url.host == "93.184.216.34"
        assert "sni_hostname" not in inner_req.extensions  # no cert to pin on http
