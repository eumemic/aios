"""The worker-controlled credential resolver (eumemic/aios#2042).

These tests drive the REAL resolver over real UDP/TCP sockets with wire-format
queries — the same path the netns ``:53`` DNAT puts every sandbox lookup on.
The property under test is the one the fix rests on: a credential host resolves
ONLY to the non-routable sentinel, no matter what the upstream would have said,
so no sampled-or-unsampled real address can ever reach the sandbox.
"""

from __future__ import annotations

import asyncio
import socket
import struct
from collections.abc import AsyncIterator

import pytest

from aios.sandbox.credential_dns import (
    CREDENTIAL_SENTINEL_IP,
    CredentialDnsError,
    CredentialDnsResolver,
)

CREDENTIAL_HOST = "api.github.com"
# A live pool member no sampler ever returned — the ordinary case against a
# rotating ~60s-TTL pool, and the exact address that used to egress directly
# with a literal placeholder.
UNSAMPLED_IP = "140.82.113.22"

QTYPE_A = 1
QTYPE_AAAA = 28
QTYPE_HTTPS = 65


def _query(name: str, qtype: int = QTYPE_A, *, query_id: int = 0x4242) -> bytes:
    labels = b"".join(bytes([len(x)]) + x.encode() for x in name.split("."))
    return (
        struct.pack("!HHHHHH", query_id, 0x0100, 1, 0, 0, 0)
        + labels
        + b"\x00"
        + struct.pack("!HH", qtype, 1)
    )


def _answers(response: bytes) -> list[str]:
    """Dotted-quads of every A record in ``response``."""
    (ancount,) = struct.unpack_from("!H", response, 6)
    if ancount == 0:
        return []
    # Single-question, single-answer shape is all this resolver ever emits.
    return [socket.inet_ntoa(response[-4:])]


class _StubUpstream:
    """A fake upstream that answers EVERYTHING with the unsampled pool address.

    If the resolver ever forwards a credential name, the sandbox learns this
    address and the fix is a fiction — so this stub is the trap that proves it
    doesn't.
    """

    def __init__(self, reply_ip: str = UNSAMPLED_IP) -> None:
        self.reply_ip = reply_ip
        self.queries: list[bytes] = []
        self._transport: asyncio.DatagramTransport | None = None
        self.port = 0

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setblocking(False)
        sock.bind(("127.0.0.1", 0))
        self.port = sock.getsockname()[1]
        outer = self

        class _Proto(asyncio.DatagramProtocol):
            def connection_made(self, transport: asyncio.BaseTransport) -> None:
                assert isinstance(transport, asyncio.DatagramTransport)
                outer._transport = transport

            def datagram_received(self, data: bytes, addr: tuple[str | int, ...]) -> None:
                outer.queries.append(data)
                response = (
                    struct.pack("!HHHHHH", struct.unpack_from("!H", data, 0)[0], 0x8180, 1, 1, 0, 0)
                    + data[12:]
                    + struct.pack("!HHHIH", 0xC00C, 1, 1, 60, 4)
                    + socket.inet_aton(outer.reply_ip)
                )
                assert outer._transport is not None
                outer._transport.sendto(response, addr)

        await loop.create_datagram_endpoint(_Proto, sock=sock)

    def stop(self) -> None:
        if self._transport is not None:
            self._transport.close()


async def _udp_ask(port: int, query: bytes, wait_s: float = 5.0) -> bytes:
    loop = asyncio.get_running_loop()
    reply: asyncio.Future[bytes] = loop.create_future()

    class _Proto(asyncio.DatagramProtocol):
        def datagram_received(self, data: bytes, addr: object) -> None:
            if not reply.done():
                reply.set_result(data)

    transport, _ = await loop.create_datagram_endpoint(_Proto, remote_addr=("127.0.0.1", port))
    try:
        transport.sendto(query)
        return await asyncio.wait_for(reply, wait_s)
    finally:
        transport.close()


@pytest.fixture
async def upstream() -> AsyncIterator[_StubUpstream]:
    stub = _StubUpstream()
    await stub.start()
    yield stub
    stub.stop()


@pytest.fixture
async def resolver(upstream: _StubUpstream) -> AsyncIterator[CredentialDnsResolver]:
    r = CredentialDnsResolver([CREDENTIAL_HOST], upstream="127.0.0.1")
    r._upstream = "127.0.0.1"
    # Point the forwarder at the stub's ephemeral port.
    original = r._forward

    async def _forward(query: bytes) -> bytes:
        return await _udp_ask(upstream.port, query)

    r._forward = _forward  # type: ignore[method-assign]
    assert original is not None
    await r.start()
    yield r
    await r.stop()


class TestCredentialNamesNeverResolveToARealAddress:
    """The core #2042 property."""

    @pytest.mark.asyncio
    async def test_credential_host_answers_sentinel(self, resolver: CredentialDnsResolver) -> None:
        response = await _udp_ask(resolver.port, _query(CREDENTIAL_HOST))
        assert _answers(response) == [CREDENTIAL_SENTINEL_IP]

    @pytest.mark.asyncio
    async def test_credential_host_is_never_forwarded(
        self, resolver: CredentialDnsResolver, upstream: _StubUpstream
    ) -> None:
        """Not forwarded at all — the sandbox cannot learn a pool address even
        if the upstream is willing to hand one over."""
        await _udp_ask(resolver.port, _query(CREDENTIAL_HOST))
        assert upstream.queries == []

    @pytest.mark.asyncio
    async def test_unsampled_pool_address_never_reaches_the_sandbox(
        self, resolver: CredentialDnsResolver
    ) -> None:
        """The upstream would answer with the address that used to fail open;
        the sandbox still only ever sees the sentinel."""
        response = await _udp_ask(resolver.port, _query(CREDENTIAL_HOST))
        assert UNSAMPLED_IP not in _answers(response)

    @pytest.mark.asyncio
    async def test_case_insensitive_match(
        self, resolver: CredentialDnsResolver, upstream: _StubUpstream
    ) -> None:
        """DNS 0x20 case randomization must not evade the match."""
        response = await _udp_ask(resolver.port, _query("ApI.GiThUb.CoM"))
        assert _answers(response) == [CREDENTIAL_SENTINEL_IP]
        assert upstream.queries == []

    @pytest.mark.asyncio
    async def test_response_echoes_query_id(self, resolver: CredentialDnsResolver) -> None:
        response = await _udp_ask(resolver.port, _query(CREDENTIAL_HOST, query_id=0xBEEF))
        assert struct.unpack_from("!H", response, 0)[0] == 0xBEEF

    @pytest.mark.asyncio
    @pytest.mark.parametrize("qtype", [QTYPE_AAAA, QTYPE_HTTPS, 64])
    async def test_non_a_records_are_nodata(
        self, resolver: CredentialDnsResolver, upstream: _StubUpstream, qtype: int
    ) -> None:
        """AAAA / HTTPS / SVCB for a credential host return NODATA.

        An AAAA answer, or an ``ipv4hint``/``ipv6hint`` inside an HTTPS/SVCB
        record, would smuggle a real pool address back into the sandbox behind
        the resolver's back — reopening the address-keyed hole via a different
        record type. The proxy is IPv4-only in any case.
        """
        response = await _udp_ask(resolver.port, _query(CREDENTIAL_HOST, qtype))
        assert struct.unpack_from("!H", response, 6)[0] == 0  # ancount
        assert struct.unpack_from("!H", response, 2)[0] & 0x000F == 0  # NOERROR, not NXDOMAIN
        assert upstream.queries == []

    @pytest.mark.asyncio
    async def test_subdomain_of_credential_host_is_not_intercepted(
        self, resolver: CredentialDnsResolver
    ) -> None:
        """Exact-match only: a subdomain is a different name with a different
        upstream, and hijacking it would break unrelated traffic."""
        response = await _udp_ask(resolver.port, _query(f"evil.{CREDENTIAL_HOST}"))
        assert _answers(response) == [UNSAMPLED_IP]


class TestOrdinaryResolutionIsUnaffected:
    """Interception, not a blackhole: everything else forwards verbatim."""

    @pytest.mark.asyncio
    async def test_non_credential_host_is_forwarded(
        self, resolver: CredentialDnsResolver, upstream: _StubUpstream
    ) -> None:
        response = await _udp_ask(resolver.port, _query("pypi.org"))
        assert _answers(response) == [UNSAMPLED_IP]
        assert len(upstream.queries) == 1

    @pytest.mark.asyncio
    async def test_docker_network_alias_still_resolves(
        self, resolver: CredentialDnsResolver, upstream: _StubUpstream
    ) -> None:
        """``aios-worker`` (the proxy alias) resolves through the forwarder, so
        in-sandbox tooling that depends on Docker aliases keeps working."""
        response = await _udp_ask(resolver.port, _query("aios-worker"))
        assert _answers(response) == [UNSAMPLED_IP]

    @pytest.mark.asyncio
    async def test_tcp_queries_are_served(self, resolver: CredentialDnsResolver) -> None:
        """A client that falls back to TCP (or is configured for it) must hit
        the same policy — the netns DNAT covers tcp/53 too."""
        reader, writer = await asyncio.open_connection("127.0.0.1", resolver.port)
        try:
            query = _query(CREDENTIAL_HOST)
            writer.write(struct.pack("!H", len(query)) + query)
            await writer.drain()
            length = struct.unpack("!H", await reader.readexactly(2))[0]
            response = await reader.readexactly(length)
        finally:
            writer.close()
        assert _answers(response) == [CREDENTIAL_SENTINEL_IP]

    @pytest.mark.asyncio
    async def test_missing_upstream_fails_start(self) -> None:
        """Provisioning must not succeed when ordinary DNS cannot be relayed."""
        r = CredentialDnsResolver([CREDENTIAL_HOST], upstream=None)
        r._upstream = None
        with pytest.raises(CredentialDnsError, match="no IPv4 upstream"):
            await r.start()


class TestFailClosed:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "query",
        [
            # An upstream may interpret the first question even though our policy
            # requires exactly one. Never forward this parser differential.
            _query(CREDENTIAL_HOST)[:4]
            + struct.pack("!H", 2)
            + _query(CREDENTIAL_HOST)[6:]
            + _query("pypi.org")[12:],
            # A compression pointer in the question is rejected by our parser;
            # an upstream must not get a chance to interpret it differently.
            struct.pack("!HHHHHH", 0x4242, 0x0100, 1, 0, 0, 0)
            + b"\xc0\x0c"
            + struct.pack("!HH", QTYPE_A, 1),
            # RFC 1035 limits an individual label to 63 octets.
            struct.pack("!HHHHHH", 0x4242, 0x0100, 1, 0, 0, 0)
            + bytes([64])
            + b"a" * 64
            + b"\0"
            + struct.pack("!HH", QTYPE_A, 1),
            # Four maximal labels exceed the 255-octet encoded-name limit.
            struct.pack("!HHHHHH", 0x4242, 0x0100, 1, 0, 0, 0)
            + (bytes([63]) + b"a" * 63) * 4
            + b"\0"
            + struct.pack("!HH", QTYPE_A, 1),
        ],
        ids=["multiple-questions", "compressed-question", "long-label", "long-name"],
    )
    async def test_unparseable_query_is_not_forwarded(
        self,
        resolver: CredentialDnsResolver,
        upstream: _StubUpstream,
        query: bytes,
    ) -> None:
        response = await _udp_ask(resolver.port, query)
        assert struct.unpack_from("!H", response, 2)[0] & 0x000F == 2  # SERVFAIL
        assert upstream.queries == []

    @pytest.mark.asyncio
    async def test_malformed_query_does_not_crash_the_resolver(self) -> None:
        r = CredentialDnsResolver([CREDENTIAL_HOST], upstream="127.0.0.1")
        await r.start()
        try:
            for junk in (b"", b"\x00", b"\xff" * 40):
                await r.answer(junk)
            response = await _udp_ask(r.port, _query(CREDENTIAL_HOST))
            assert _answers(response) == [CREDENTIAL_SENTINEL_IP]
        finally:
            await r.stop()

    @pytest.mark.asyncio
    async def test_bind_failure_raises_credential_dns_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A resolver that cannot start must fail the provision, not run
        without interception."""
        r = CredentialDnsResolver([CREDENTIAL_HOST])

        async def _boom(*args: object, **kwargs: object) -> None:
            raise OSError("no sockets today")

        monkeypatch.setattr(asyncio.get_running_loop(), "create_datagram_endpoint", _boom)
        with pytest.raises(CredentialDnsError):
            await r.start()

    def test_sentinel_uses_an_ordinary_routable_destination(self) -> None:
        """The sentinel reaches nat OUTPUT without runtime-specific routes."""
        assert CREDENTIAL_SENTINEL_IP == "1.1.1.1"

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self) -> None:
        r = CredentialDnsResolver([CREDENTIAL_HOST], upstream="127.0.0.1")
        await r.start()
        await r.stop()
        await r.stop()

    @pytest.mark.asyncio
    async def test_stop_reaps_in_flight_udp_queries(self) -> None:
        """Shutdown must not leak UDP response tasks into later tests."""
        r = CredentialDnsResolver([CREDENTIAL_HOST], upstream="127.0.0.1")
        started = asyncio.Event()

        async def _blocked_forward(query: bytes) -> bytes:
            started.set()
            await asyncio.Event().wait()
            return query

        r._forward = _blocked_forward  # type: ignore[method-assign]
        await r.start()
        ask = asyncio.create_task(_udp_ask(r.port, _query("ordinary.example")))
        await started.wait()
        await r.stop()
        ask.cancel()
        await asyncio.gather(ask, return_exceptions=True)
        assert r._udp_protocol is None


class TestNoRequestContentIsLogged:
    """#2041 round-1 defect must not reappear one layer down: query names are
    sandbox-controlled and must never be logged."""

    @pytest.mark.asyncio
    async def test_query_names_are_not_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        r = CredentialDnsResolver([CREDENTIAL_HOST], upstream="127.0.0.1")
        await r.start()
        try:
            with caplog.at_level("DEBUG"):
                await _udp_ask(r.port, _query("super-secret-internal-host.example"))
                await _udp_ask(r.port, _query(CREDENTIAL_HOST))
            assert "super-secret-internal-host" not in caplog.text
        finally:
            await r.stop()
