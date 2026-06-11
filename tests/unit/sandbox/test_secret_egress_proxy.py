"""Unit tests for the TLS-terminating secret-egress proxy.

Boots a real :class:`SecretEgressProxy` (so we exercise the actual asyncio
TLS server, the sni_callback leaf-mint, and h11 framing), terminates TLS with
a leaf the test client trusts via the aios egress CA, and mocks the upstream
httpx transport so requests never leave the test process. DNS resolution is
monkeypatched to a fixed sentinel IP so the SSRF pin proceeds to the mock
without real lookups.

No docker, no real upstream host, no real secret egress.
"""

from __future__ import annotations

import socket
import ssl
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from datetime import UTC, datetime

import httpx
import pytest

from aios.crypto.vault import CryptoBox
from aios.harness import runtime
from aios.sandbox import secret_egress_proxy as sep
from aios.sandbox.egress_ca import get_egress_ca
from aios.sandbox.secret_egress_proxy import (
    SecretEgressProxy,
    _path_within_prefix,
    _request_path,
    _resolve_pinned_ip,
)
from aios.services.vaults import SECRET_PLACEHOLDER_PREFIX, ResolvedEnvVarCredential

SEED = bytes(range(32))
# TEST-NET-3 sentinel: what the monkeypatched resolver pins to. Never dialed
# (the upstream transport is mocked); used to prove the URL targets the
# resolved IP, not the hostname.
PINNED_IP = "203.0.113.5"


def _ph(tag: str) -> str:
    """A placeholder token shaped like the real ones (prefix + 32 hex chars)."""
    return SECRET_PLACEHOLDER_PREFIX + (tag * 32)[:32]


def _cred(
    secret_name: str, secret_value: str, allowed_hosts: tuple[str, ...], placeholder: str
) -> ResolvedEnvVarCredential:
    return ResolvedEnvVarCredential(
        credential_id="cred_" + secret_name,
        secret_name=secret_name,
        secret_value=secret_value,
        allowed_hosts=allowed_hosts,
        updated_at=datetime.now(UTC),
        placeholder=placeholder,
    )


def _fixed_resolver(ip: str | None) -> Callable[[str, int], Awaitable[str | None]]:
    async def _resolve(host: str, port: int) -> str | None:
        return ip

    return _resolve


@pytest.fixture
def crypto_box_runtime() -> Iterator[None]:
    prev = runtime.crypto_box
    runtime.crypto_box = CryptoBox(SEED)
    try:
        yield
    finally:
        runtime.crypto_box = prev


def _mock_upstream(captured: list[httpx.Request]) -> httpx.MockTransport:
    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, headers={"x-upstream": "ok"}, content=b"upstream-body")

    return httpx.MockTransport(handler)


async def _boot(
    creds: list[ResolvedEnvVarCredential], transport: httpx.MockTransport
) -> SecretEgressProxy:
    proxy = SecretEgressProxy(creds)
    await proxy._client.aclose()
    proxy._client = httpx.AsyncClient(transport=transport)
    await proxy.start()
    return proxy


def _client_ctx() -> ssl.SSLContext:
    """A client trust store containing only the aios egress CA."""
    return ssl.create_default_context(cadata=get_egress_ca().cert_pem)


async def _request(
    proxy: SecretEgressProxy,
    sni: str | None,
    path: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    content: bytes | None = None,
) -> httpx.Response:
    """One TLS request to the proxy presenting ``sni`` (or no SNI) as the host."""
    extensions = {"sni_hostname": sni} if sni is not None else {}
    async with httpx.AsyncClient(verify=_client_ctx()) as client:
        return await client.request(
            method,
            f"https://127.0.0.1:{proxy.port}{path}",
            headers=headers or {},
            content=content,
            extensions=extensions,
        )


# A single bare-host credential reused by most forwarding tests.
PH_GH = _ph("a")

ProxyAndCapture = tuple[SecretEgressProxy, list[httpx.Request]]
MakeProxy = Callable[..., Awaitable[ProxyAndCapture]]


@pytest.fixture
async def make_proxy(
    crypto_box_runtime: None, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[MakeProxy]:
    """Boot a proxy with a fixed resolver + mock upstream and stop every proxy
    it hands out at teardown. ``resolver_ip=None`` exercises the SSRF block;
    a custom ``transport`` exercises upstream behavior."""
    started: list[SecretEgressProxy] = []

    async def _make(
        creds: list[ResolvedEnvVarCredential],
        *,
        resolver_ip: str | None = PINNED_IP,
        transport: httpx.MockTransport | None = None,
    ) -> ProxyAndCapture:
        monkeypatch.setattr(sep, "_resolve_pinned_ip", _fixed_resolver(resolver_ip))
        captured: list[httpx.Request] = []
        proxy = await _boot(creds, transport or _mock_upstream(captured))
        started.append(proxy)
        return proxy, captured

    yield _make
    for proxy in started:
        await proxy.stop()


@pytest.fixture
async def gh_proxy(make_proxy: MakeProxy) -> ProxyAndCapture:
    return await make_proxy([_cred("GH_TOKEN", "ghp_REALSECRET", ("api.allowed.test",), PH_GH)])


class TestPathHelpers:
    def test_request_path_strips_query_and_fragment(self) -> None:
        assert _request_path("/a/b?x=1&y=2") == "/a/b"
        assert _request_path("/a/b#frag") == "/a/b"
        assert _request_path("/a/b?x=1#f") == "/a/b"
        assert _request_path("/a/b") == "/a/b"

    @pytest.mark.parametrize(
        "path,expected",
        [
            ("/repos/eumemic", True),
            ("/repos/eumemic/", True),
            ("/repos/eumemic/x", True),
            ("/repos/eumemic/git/ref/heads%2Ffoo", True),  # encoded slash inside subtree
            ("/repos/eumemic-evil", False),  # segment boundary
            ("/repos/eumemic-evil/x", False),
            ("/repos/other", False),
            ("/repos/eumemic/../../gists", False),  # dot climb-out
            ("/repos/eumemic/%2e%2e/x", False),  # encoded dot climb-out
            ("/repos/eumemic/..%2fgists", False),  # encoded-slash dot climb-out
            ("/repos/eumemic/x/../y", False),  # interior dot-segment
        ],
    )
    def test_path_within_prefix(self, path: str, expected: bool) -> None:
        assert _path_within_prefix("/repos/eumemic", path) is expected


class TestSwapForwarding:
    async def test_swap_in_authorization_header(
        self, gh_proxy: tuple[SecretEgressProxy, list[httpx.Request]]
    ) -> None:
        proxy, captured = gh_proxy
        r = await _request(
            proxy, "api.allowed.test", "/user", headers={"Authorization": f"Bearer {PH_GH}"}
        )
        assert r.status_code == 200
        assert captured[0].headers["authorization"] == "Bearer ghp_REALSECRET"

    async def test_swap_in_arbitrary_header(
        self, gh_proxy: tuple[SecretEgressProxy, list[httpx.Request]]
    ) -> None:
        proxy, captured = gh_proxy
        await _request(proxy, "api.allowed.test", "/x", headers={"X-Api-Key": PH_GH})
        assert captured[0].headers["x-api-key"] == "ghp_REALSECRET"

    async def test_swap_in_body(
        self, gh_proxy: tuple[SecretEgressProxy, list[httpx.Request]]
    ) -> None:
        proxy, captured = gh_proxy
        await _request(
            proxy,
            "api.allowed.test",
            "/x",
            method="POST",
            content=f'{{"token":"{PH_GH}"}}'.encode(),
        )
        assert captured[0].content == b'{"token":"ghp_REALSECRET"}'

    async def test_url_path_and_query_not_swapped(
        self, gh_proxy: tuple[SecretEgressProxy, list[httpx.Request]]
    ) -> None:
        proxy, captured = gh_proxy
        await _request(proxy, "api.allowed.test", f"/v1/{PH_GH}?token={PH_GH}")
        # The placeholder rides through the URL verbatim — never swapped there.
        assert PH_GH in str(captured[0].url)
        assert "ghp_REALSECRET" not in str(captured[0].url)

    async def test_pins_resolved_ip_and_keeps_sni_on_hostname(
        self, gh_proxy: tuple[SecretEgressProxy, list[httpx.Request]]
    ) -> None:
        proxy, captured = gh_proxy
        await _request(proxy, "api.allowed.test", "/x")
        # Rebind-safe pin: the URL targets the resolved IP, but SNI +
        # verification + Host stay on the hostname.
        assert captured[0].url.host == PINNED_IP
        assert captured[0].extensions["sni_hostname"] == "api.allowed.test"
        assert captured[0].headers["host"] == "api.allowed.test"

    async def test_response_body_streamed_back(
        self, gh_proxy: tuple[SecretEgressProxy, list[httpx.Request]]
    ) -> None:
        proxy, _ = gh_proxy
        r = await _request(proxy, "api.allowed.test", "/x")
        assert r.content == b"upstream-body"
        assert r.headers["x-upstream"] == "ok"

    async def test_keepalive_disabled_so_sni_is_per_request(self, crypto_box_runtime: None) -> None:
        # F2: pooled connections key on IP-origin only and ignore a later
        # request's sni_hostname on reuse; a zero keepalive cap forces a fresh
        # handshake per request so two allowed hosts sharing a pinned IP can't
        # ride one another's verified TLS session.
        proxy = SecretEgressProxy([])
        try:
            assert proxy._client._transport._pool._max_keepalive_connections == 0  # type: ignore[attr-defined]
        finally:
            await proxy._client.aclose()


class TestPathGating:
    @pytest.fixture
    async def repo_proxy(self, make_proxy: MakeProxy) -> ProxyAndCapture:
        return await make_proxy(
            [_cred("GH_TOKEN", "ghp_REALSECRET", ("api.allowed.test/repos/eumemic",), PH_GH)]
        )

    @pytest.mark.parametrize("path", ["/repos/eumemic", "/repos/eumemic/x"])
    async def test_swaps_within_prefix(
        self, repo_proxy: tuple[SecretEgressProxy, list[httpx.Request]], path: str
    ) -> None:
        proxy, captured = repo_proxy
        await _request(proxy, "api.allowed.test", path, headers={"Authorization": PH_GH})
        assert captured[0].headers["authorization"] == "ghp_REALSECRET"

    @pytest.mark.parametrize("path", ["/repos/eumemic-evil", "/repos/other"])
    async def test_passes_through_outside_prefix(
        self, repo_proxy: tuple[SecretEgressProxy, list[httpx.Request]], path: str
    ) -> None:
        proxy, captured = repo_proxy
        await _request(proxy, "api.allowed.test", path, headers={"Authorization": PH_GH})
        # Path gate fails → placeholder rides through unswapped.
        assert captured[0].headers["authorization"] == PH_GH

    async def test_dot_segment_climbout_passes_through(
        self, repo_proxy: tuple[SecretEgressProxy, list[httpx.Request]]
    ) -> None:
        proxy, captured = repo_proxy
        await _request(
            proxy,
            "api.allowed.test",
            "/repos/eumemic/../../gists",
            headers={"Authorization": PH_GH},
        )
        assert captured[0].headers["authorization"] == PH_GH

    async def test_mixed_case_stored_host_terminates_and_swaps(self, make_proxy: MakeProxy) -> None:
        # F1: parse_allowed_host_entry stores the host as-given; the proxy must
        # lowercase it so a legitimately allowed mixed-case host doesn't fail
        # closed and the swap still fires for a lowercase SNI.
        proxy, captured = await make_proxy(
            [_cred("GH_TOKEN", "ghp_REALSECRET", ("API.Allowed.Test",), PH_GH)]
        )
        assert proxy._allowed_hosts == frozenset({"api.allowed.test"})
        await _request(proxy, "api.allowed.test", "/x", headers={"Authorization": PH_GH})
        assert captured[0].headers["authorization"] == "ghp_REALSECRET"

    async def test_unauthorized_host_passes_placeholder_through(
        self, make_proxy: MakeProxy
    ) -> None:
        # Two allow-set hosts (so TLS terminates for both); a placeholder for
        # host A sent to host B does not swap (B has no matching cred).
        proxy, captured = await make_proxy(
            [
                _cred("A_TOKEN", "secret_A", ("api.a.test",), PH_GH),
                _cred("B_TOKEN", "secret_B", ("api.b.test",), _ph("b")),
            ]
        )
        await _request(proxy, "api.b.test", "/x", headers={"Authorization": PH_GH})
        # PH_GH belongs to api.a.test; on api.b.test it rides through raw.
        assert captured[0].headers["authorization"] == PH_GH


class TestFailClosed:
    async def test_unauthorized_sni_resets_handshake(
        self, gh_proxy: tuple[SecretEgressProxy, list[httpx.Request]]
    ) -> None:
        proxy, captured = gh_proxy
        with pytest.raises((httpx.ConnectError, ssl.SSLError)):
            await _request(proxy, "evil.test", "/x")
        assert captured == []  # no leaf, no termination, no upstream call

    async def test_absent_sni_resets_handshake(
        self, gh_proxy: tuple[SecretEgressProxy, list[httpx.Request]]
    ) -> None:
        proxy, captured = gh_proxy
        # No sni_hostname extension → IP-literal SNI is absent → fail closed.
        with pytest.raises((httpx.ConnectError, ssl.SSLError)):
            await _request(proxy, None, "/x")
        assert captured == []

    async def test_ssrf_block_returns_502_without_egress(self, make_proxy: MakeProxy) -> None:
        proxy, captured = await make_proxy(
            [_cred("GH_TOKEN", "ghp_REALSECRET", ("api.allowed.test",), PH_GH)],
            resolver_ip=None,
        )
        r = await _request(proxy, "api.allowed.test", "/x", headers={"Authorization": PH_GH})
        assert r.status_code == 502
        assert captured == []  # the secret never egressed

    async def test_upstream_error_returns_502(self, make_proxy: MakeProxy) -> None:
        async def boom(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("simulated")

        proxy, _ = await make_proxy(
            [_cred("GH_TOKEN", "ghp_REALSECRET", ("api.allowed.test",), PH_GH)],
            transport=httpx.MockTransport(boom),
        )
        r = await _request(proxy, "api.allowed.test", "/x")
        assert r.status_code == 502


class TestOutboundOnly:
    async def test_response_echoing_secret_is_not_redacted(self, make_proxy: MakeProxy) -> None:
        # #881: responses are NOT scrubbed; the env egress allowlist is the
        # hard boundary, not this proxy. A reflecting host returns the real
        # secret straight back into the sandbox.
        async def echo(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"you sent ghp_REALSECRET")

        proxy, _ = await make_proxy(
            [_cred("GH_TOKEN", "ghp_REALSECRET", ("api.allowed.test",), PH_GH)],
            transport=httpx.MockTransport(echo),
        )
        r = await _request(proxy, "api.allowed.test", "/x")
        assert r.content == b"you sent ghp_REALSECRET"


class TestResolvePinnedIp:
    """Direct tests of the resolve-time SSRF gate (monkeypatching the DNS seam,
    no real lookups)."""

    @staticmethod
    def _addrinfos(*ips: str) -> Callable[[str, int], Awaitable[list[tuple[object, ...]]]]:
        async def _resolve(host: str, port: int) -> list[tuple[object, ...]]:
            out: list[tuple[object, ...]] = []
            for ip in ips:
                fam = socket.AF_INET6 if ":" in ip else socket.AF_INET
                out.append((fam, socket.SOCK_STREAM, 0, "", (ip, port)))
            return out

        return _resolve

    async def test_public_ip_returned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sep, "_resolve_addrinfos", self._addrinfos("93.184.216.34"))
        assert await _resolve_pinned_ip("api.example.com", 443) == "93.184.216.34"

    @pytest.mark.parametrize(
        "ip", ["127.0.0.1", "169.254.169.254", "10.0.0.1", "192.168.1.1", "fd00::1", "100.64.0.1"]
    )
    async def test_blocked_ip_returns_none(self, ip: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sep, "_resolve_addrinfos", self._addrinfos(ip))
        assert await _resolve_pinned_ip("api.example.com", 443) is None

    async def test_any_blocked_ip_in_set_blocks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A good public IP alongside an internal one still fails closed.
        monkeypatch.setattr(sep, "_resolve_addrinfos", self._addrinfos("93.184.216.34", "10.0.0.1"))
        assert await _resolve_pinned_ip("api.example.com", 443) is None

    async def test_blocked_hostname_returns_none_without_resolving(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(host: str, port: int) -> object:
            raise AssertionError("must not resolve a blocked hostname")

        monkeypatch.setattr(sep, "_resolve_addrinfos", _boom)
        assert await _resolve_pinned_ip("metadata.google.internal", 443) is None

    async def test_prefers_ipv4(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # F11: AAAA first, A second → pin the reachable IPv4 (worker is v4-only).
        monkeypatch.setattr(
            sep, "_resolve_addrinfos", self._addrinfos("2606:2800:220::1", "93.184.216.34")
        )
        assert await _resolve_pinned_ip("api.example.com", 443) == "93.184.216.34"

    @pytest.mark.parametrize("exc", [socket.gaierror("nope"), OSError("EAI_SYSTEM")])
    async def test_resolution_failure_returns_none(
        self, exc: Exception, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # F8: gaierror AND any other OSError fail closed (not just gaierror).
        async def _raise(host: str, port: int) -> list[tuple[object, ...]]:
            raise exc

        monkeypatch.setattr(sep, "_resolve_addrinfos", _raise)
        assert await _resolve_pinned_ip("api.example.com", 443) is None


class TestLifecycle:
    async def test_stop_before_start_is_safe(self, crypto_box_runtime: None) -> None:
        # F5: stop() must None-guard the server and still close the client.
        proxy = SecretEgressProxy([])
        await proxy.stop()
        assert proxy._client.is_closed

    async def test_double_stop_is_safe(
        self, gh_proxy: tuple[SecretEgressProxy, list[httpx.Request]]
    ) -> None:
        proxy, _ = gh_proxy
        await proxy.stop()
        await proxy.stop()  # idempotent


def test_module_exports() -> None:
    from aios.sandbox import secret_egress_proxy

    assert secret_egress_proxy.__all__ == ["SecretEgressProxy"]
