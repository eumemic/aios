"""Worker-controlled resolver that binds credential-host policy to NAMES.

The fix for eumemic/aios#2042. Before this module, credential-host egress
interception was keyed on **sampled IP addresses**: the nat-OUTPUT DNAT that
redirects credential traffic to the secret-egress proxy was generated only for
the addresses ``getent ahostsv4`` happened to return inside the sandbox netns
at provision/refresh time. ``api.github.com`` serves a rotating ~60s-TTL pool
and answers with only a *subset* per query, so "an address we never sampled"
is the ordinary case — and such an address matched no rule, left under the
Unrestricted default-``ACCEPT`` policy, and carried the literal
``AIOS_SECRET_PLACEHOLDER_*`` to the real upstream, which answered ``401``.
That is the misleading "flaky auth, retry fixes it" signature. More probing
shrinks the window and cannot close it: a sample is not a guarantee.

**This module removes the sample from the decision.** A per-session resolver
runs on the worker; every DNS query leaving the sandbox netns is redirected to
it (a nat-OUTPUT DNAT on ``:53``, inserted at the TOP of the chain so no
in-netns resolver — including Docker's embedded DNS at ``127.0.0.11`` — can
answer first), with the rewritten flow source-NATed so Docker's loopback source
can cross the bridge to the worker. For a **credential host** the resolver never forwards and never
returns a real address: it answers ``A`` with a single fixed, non-routable
sentinel (:data:`CREDENTIAL_SENTINEL_IP`), and answers every other record type
for that name with NODATA — so ``AAAA``, and the ``ipv4hint``/``ipv6hint``
carrying ``HTTPS``/``SVCB`` records, cannot smuggle a pool address back in.
Everything else is forwarded verbatim to the worker's own upstream resolver, so
ordinary sandbox name resolution (including Docker network aliases such as
``aios-worker``) is unchanged.

This interception also makes DNS the first **UDP** hop from a sandbox to the
worker (every other chokepoint hop — tool broker, git proxy, secret-egress
proxy — is TCP, where the accepted socket's local address is pinned to the
SYN's destination). A multihomed worker must therefore answer from the address
the sandbox queried, not from whatever the route back picks; see
:func:`_reply_pktinfo`.

The netns then carries exactly one credential rule, and it is keyed on a
constant this worker chose:

    -t nat -A OUTPUT -d <sentinel> -p tcp --dport 443 \
        -j DNAT --to-destination <proxy_ip>:<proxy_port>

so **an address nobody ever sampled is structurally incapable of bypassing the
proxy** — not because we enumerated it, but because the name can no longer
resolve to it inside the sandbox.

Fail-closed by construction, at three layers:

* the sentinel is RFC 3927 link-local (``169.254.0.0/16``) and is **not routed
  anywhere**: if the DNAT is missing or malformed, a credential connection dies
  in the sandbox's own stack instead of reaching a real upstream. A broken rule
  can only deny, never leak;
* if this resolver cannot start, :class:`~aios.sandbox.secret_egress_proxy.SecretEgressProxy`
  start fails, provisioning raises, and the sandbox is never handed back — a
  sandbox that cannot protect a credential is not allowed to send one;
* if the interception cannot be installed in the netns, the apply script exits
  nonzero and the read-back verify refuses the provision
  (``aios.sandbox.setup``).

Nothing request-shaped is ever logged here: query names are sandbox-controlled
and this module is on no post-swap path, so log events carry counters and
error *types* only (the #2041 round-1 defect — a scanner that logged its own
match, where the match can be a real secret — is not re-introduced one layer
down).
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import socket
import struct
from collections.abc import Iterable
from pathlib import Path

from aios.logging import get_logger

log = get_logger("aios.sandbox.credential_dns")


# The single address every credential host resolves to inside a sandbox.
#
# RFC 3927 link-local, deliberately NOT routable: the sandbox has no route that
# carries it off the box, so the ONLY way a packet addressed here reaches
# anything is the nat-OUTPUT DNAT that rewrites it to the secret-egress proxy.
# A missing/mis-installed DNAT therefore fails CLOSED (the connection dies in
# the sandbox's own stack) instead of failing open to the real upstream, which
# is exactly the property the sampled-IP scheme lacked.
CREDENTIAL_SENTINEL_IP = "169.254.53.53"

# TTL on the sentinel answer. Short so a client that caches across a sandbox
# recycle re-asks, but the value is not load-bearing: the netns interception
# means every answer for these names comes from here regardless.
_SENTINEL_TTL = 30

# DNS record/class numbers we care about.
_QTYPE_A = 1
_QCLASS_IN = 1

_MAX_UDP_RESPONSE = 512
_MAX_UDP_QUERY = 65535
_MAX_TCP_MESSAGE = 65535
_UPSTREAM_TIMEOUT_S = 5.0
_TCP_IDLE_TIMEOUT_S = 15.0
# UDP bind(0) and TCP bind(0) consult separate port spaces, so a dual-protocol
# bind on one port can collide with a live listener under parallel e2e. Retry
# a fresh ephemeral pair; SO_REUSEADDR / SO_REUSEPORT would not make a live
# listener shareable and must not — two sessions must not share a resolver.
_BIND_ATTEMPTS = 16

# Standard resolver config on the worker; its first nameserver is the upstream
# ordinary (non-credential) queries are forwarded to. In a containerized worker
# this is Docker's embedded DNS, which is what resolves network aliases like
# ``aios-worker`` — so forwarding there keeps sandbox name resolution
# byte-identical to what it was before interception.
_RESOLV_CONF = Path("/etc/resolv.conf")

# ``struct in_pktinfo`` — ``{int ipi_ifindex; struct in_addr ipi_spec_dst;
# struct in_addr ipi_addr;}``, 12 bytes with no padding, native byte order.
_PKTINFO_FORMAT = "I4s4s"
_PKTINFO_SIZE = struct.calcsize(_PKTINFO_FORMAT)
# Linux-only; absent on macOS, where a developer's worker and sandbox share a
# single address anyway.
_IP_PKTINFO: int | None = getattr(socket, "IP_PKTINFO", None)


class CredentialDnsError(RuntimeError):
    """The resolver could not be brought up. Provisioning must fail closed."""


def _parse_question(message: bytes) -> tuple[str, int, int, int] | None:
    """Return ``(name, qtype, qclass, question_end)`` for a single-question query.

    ``name`` is lowercased with no trailing dot, so DNS 0x20 case games cannot
    slip a credential host past the match. Returns ``None`` for anything that
    is not a well-formed single-question query — such a message is forwarded
    verbatim rather than interpreted, since we only ever special-case names we
    have positively identified.
    """
    if len(message) < 12:
        return None
    (qdcount,) = struct.unpack_from("!H", message, 4)
    if qdcount != 1:
        return None
    labels: list[str] = []
    offset = 12
    while True:
        if offset >= len(message):
            return None
        length = message[offset]
        # Compression pointers are illegal in a question section; refuse to
        # interpret rather than guess.
        if length & 0xC0:
            return None
        offset += 1
        if length == 0:
            break
        label = message[offset : offset + length]
        if len(label) != length:
            return None
        labels.append(label.decode("ascii", errors="replace").lower())
        offset += length
    if offset + 4 > len(message):
        return None
    qtype, qclass = struct.unpack_from("!HH", message, offset)
    return ".".join(labels), qtype, qclass, offset + 4


def _response_header(message: bytes, *, ancount: int, rcode: int = 0) -> bytes:
    """Build a response header echoing the query's id and RD flag."""
    (query_id,) = struct.unpack_from("!H", message, 0)
    (flags,) = struct.unpack_from("!H", message, 2)
    recursion_desired = flags & 0x0100
    # QR=1, AA=1, RA=1 + the query's RD + rcode.
    response_flags = 0x8000 | 0x0400 | 0x0080 | recursion_desired | (rcode & 0x000F)
    return struct.pack("!HHHHHH", query_id, response_flags, 1, ancount, 0, 0)


def _sentinel_answer(message: bytes, question_end: int) -> bytes:
    """``NOERROR`` with a single A record pointing at the sentinel."""
    question = message[12:question_end]
    rdata = socket.inet_aton(CREDENTIAL_SENTINEL_IP)
    answer = struct.pack("!HHHIH", 0xC00C, _QTYPE_A, _QCLASS_IN, _SENTINEL_TTL, len(rdata)) + rdata
    return _response_header(message, ancount=1) + question + answer


def _nodata_answer(message: bytes, question_end: int) -> bytes:
    """``NOERROR`` with zero answers — the name exists, this type does not.

    Used for every non-``A`` query for a credential host (``AAAA``, and the
    ``HTTPS``/``SVCB`` records whose ``ipv4hint``/``ipv6hint`` would otherwise
    hand the sandbox a real pool address behind the resolver's back).
    """
    return _response_header(message, ancount=0) + message[12:question_end]


def _servfail(message: bytes) -> bytes:
    if len(message) < 12:
        return b""
    (query_id,) = struct.unpack_from("!H", message, 0)
    return struct.pack("!HHHHHH", query_id, 0x8182, 0, 0, 0, 0)


def _resolv_conf_nameserver(path: Path = _RESOLV_CONF) -> str | None:
    """First ``nameserver`` in the worker's resolv.conf, if any."""
    try:
        text = path.read_text()
    except OSError:
        return None
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "nameserver" and ":" not in parts[1]:
            return parts[1]
    return None


def _reply_pktinfo(ancdata: Iterable[tuple[int, int, bytes]]) -> bytes | None:
    """Packed ``in_pktinfo`` that sources a reply from the address that was queried.

    A wildcard-bound (``0.0.0.0``) UDP socket that answers with plain
    ``sendto`` lets the ROUTE pick the reply's source address, which on a
    multihomed worker is not the address the client sent to. That is fatal
    here: the sandbox reaches the resolver at the ``aios-worker`` address
    (``host-gateway`` → the ``docker0`` gateway when the worker runs on the
    host), while the route back to the sandbox leaves via the
    ``aios-sandbox`` bridge — so the reply arrives with a source the
    sandbox's conntrack entry does not match, and is discarded. glibc's stub
    resolver uses a *connected* socket, so the kernel drops the mismatched
    reply before the process ever sees it and every lookup times out.

    The fix is what every real resolver does: read ``IP_PKTINFO`` off the
    query and echo the local address back on the reply.
    ``ipi_spec_dst`` is the kernel's own answer to "what source would you use
    to reply to this packet" (``fib_compute_spec_dst``), and on send it is the
    field that selects the source; ``ipi_ifindex`` 0 and ``ipi_addr`` are
    ignored there, so they go out zeroed.
    """
    if _IP_PKTINFO is None:
        return None
    for level, cmsg_type, cmsg_data in ancdata:
        if level != socket.IPPROTO_IP or cmsg_type != _IP_PKTINFO:
            continue
        if len(cmsg_data) < _PKTINFO_SIZE:
            continue
        _ifindex, spec_dst, _hdr_dst = struct.unpack_from(_PKTINFO_FORMAT, cmsg_data)
        return struct.pack(_PKTINFO_FORMAT, 0, spec_dst, b"\x00" * 4)
    return None


class CredentialDnsResolver:
    """Per-session DNS server that answers credential hosts with the sentinel.

    Owned by :class:`~aios.sandbox.secret_egress_proxy.SecretEgressProxy`: it
    is started and stopped with the proxy and is seeded from the SAME name set
    that gates the proxy's leaf minting, so the names this resolver hijacks and
    the names the proxy will terminate TLS for can never drift apart.
    """

    def __init__(self, credential_hosts: Iterable[str], *, upstream: str | None = None) -> None:
        # Names are matched exactly (lowercased): a credential host's
        # SUBDOMAIN is a different name with a different upstream and is not
        # ours to intercept.
        self._hosts: frozenset[str] = frozenset(
            host.lower().rstrip(".") for host in credential_hosts if host
        )
        self._upstream: str | None = upstream or _resolv_conf_nameserver()
        self._port: int | None = None
        self._udp_sock: socket.socket | None = None
        self._udp_tasks: set[asyncio.Task[None]] = set()
        self._tcp_server: asyncio.Server | None = None
        self._tcp_conns: set[asyncio.Task[None]] = set()

    @property
    def port(self) -> int:
        assert self._port is not None, "CredentialDnsResolver.start() has not completed"
        return self._port

    @property
    def hosts(self) -> frozenset[str]:
        return self._hosts

    async def start(self) -> None:
        """Bind UDP+TCP on the same ephemeral port.

        Raises :class:`CredentialDnsError` if either bind fails — the caller
        turns that into a failed provision, because a sandbox whose credential
        names cannot be pinned must not be handed a credential.

        TCP is bound first so the kernel hands us a port that is free in the
        TCP space; UDP is then attached to that port. The opposite order is
        what fails under parallel e2e: ``UDP bind(0)`` does not consult TCP
        listeners, so it can land on a port another process already holds,
        and the subsequent TCP bind raises ``EADDRINUSE`` (errno 98). The
        netns DNAT uses a single ``dns_port`` for both ``udp/53`` and
        ``tcp/53`` (:func:`aios.sandbox.setup._nat_dnat_lines`), so the
        sockets must share a port — splitting them is not an option.
        ``EADDRINUSE`` on the UDP attach (or a race) retries with a new
        ephemeral port; any other bind error still fails closed.
        """
        for attempt in range(_BIND_ATTEMPTS):
            try:
                await self._bind()
                break
            except OSError as exc:
                await self.stop()
                # Raise from the loop, never from a post-loop state check: a
                # resolver that bound once keeps ``_port`` across ``stop()``,
                # so "did we bind?" must be answered by control flow or a
                # failed re-``start()`` would return a stale port with nothing
                # listening on it.
                if exc.errno != errno.EADDRINUSE or attempt + 1 == _BIND_ATTEMPTS:
                    raise CredentialDnsError("credential DNS resolver failed to bind") from exc
            except BaseException as exc:
                await self.stop()
                raise CredentialDnsError("credential DNS resolver failed to bind") from exc
        log.info(
            "credential_dns.started",
            port=self._port,
            credential_host_count=len(self._hosts),
            has_upstream=self._upstream is not None,
        )

    async def _bind(self) -> None:
        loop = asyncio.get_running_loop()
        self._tcp_server = await asyncio.start_server(self._handle_tcp, "0.0.0.0", 0)
        sockets = self._tcp_server.sockets
        assert sockets, "asyncio.start_server returned no sockets"
        port: int = sockets[0].getsockname()[1]
        udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Assigned before the first fallible call so a later failure still
        # closes the descriptor through :meth:`stop`.
        self._udp_sock = udp_sock
        udp_sock.setblocking(False)
        if _IP_PKTINFO is not None:
            # See :func:`_reply_pktinfo`: without this the reply to a
            # sandbox query can leave with the wrong source address.
            udp_sock.setsockopt(socket.IPPROTO_IP, _IP_PKTINFO, 1)
        udp_sock.bind(("0.0.0.0", port))
        # Read the socket directly rather than through a DatagramTransport:
        # the transport has no way to attach the ``IP_PKTINFO`` control
        # message the reply needs.
        loop.add_reader(udp_sock.fileno(), self._read_udp)
        self._port = port

    async def stop(self) -> None:
        if self._udp_sock is not None:
            with contextlib.suppress(Exception):
                asyncio.get_running_loop().remove_reader(self._udp_sock.fileno())
            self._udp_sock.close()
            self._udp_sock = None
        for task in list(self._udp_tasks):
            task.cancel()
        if self._udp_tasks:
            await asyncio.gather(*self._udp_tasks, return_exceptions=True)
        if self._tcp_server is not None:
            self._tcp_server.close()
            for task in list(self._tcp_conns):
                task.cancel()
            if self._tcp_conns:
                await asyncio.gather(*self._tcp_conns, return_exceptions=True)
            with contextlib.suppress(Exception):
                await self._tcp_server.wait_closed()
            self._tcp_server = None
        log.info("credential_dns.stopped", port=self._port)

    async def answer(self, query: bytes) -> bytes:
        """Resolve one wire-format query into one wire-format response.

        The whole name-based policy is these few lines: a credential host is
        answered from here and NEVER forwarded (so the sandbox cannot learn a
        real address for it, sampled or not); everything else is forwarded
        untouched (so ordinary resolution is unaffected).
        """
        parsed = _parse_question(query)
        if parsed is None:
            return await self._forward(query)
        name, qtype, qclass, question_end = parsed
        if name not in self._hosts:
            return await self._forward(query)
        if qclass != _QCLASS_IN:
            return _nodata_answer(query, question_end)
        if qtype == _QTYPE_A:
            return _sentinel_answer(query, question_end)
        # AAAA / HTTPS / SVCB / anything else for a credential name: NODATA.
        # An IPv6 address or an ipv4hint here would be a real pool address
        # reaching the sandbox — the exact thing this resolver exists to
        # prevent — and the proxy is IPv4-only anyway.
        return _nodata_answer(query, question_end)

    async def _forward(self, query: bytes) -> bytes:
        """Relay a non-credential query to the worker's upstream resolver.

        A forwarding failure is a SERVFAIL for THAT query only. It is never a
        fallback to un-intercepted resolution: credential names are answered
        locally and never touch this path.
        """
        if self._upstream is None or len(query) < 12:
            return _servfail(query)
        loop = asyncio.get_running_loop()
        try:
            transport, protocol = await loop.create_datagram_endpoint(
                _ForwardProtocol, remote_addr=(self._upstream, 53)
            )
        except OSError:
            return _servfail(query)
        try:
            transport.sendto(query)
            return await asyncio.wait_for(protocol.reply, _UPSTREAM_TIMEOUT_S)
        except (TimeoutError, OSError):
            log.warning("credential_dns.upstream_timeout")
            return _servfail(query)
        finally:
            transport.close()

    def _read_udp(self) -> None:
        """Read one query off the wildcard socket (level-triggered: the loop
        re-enters while more are queued)."""
        sock = self._udp_sock
        if sock is None:
            return
        try:
            data, ancdata, _flags, addr = sock.recvmsg(
                _MAX_UDP_QUERY, socket.CMSG_SPACE(_PKTINFO_SIZE)
            )
        except (BlockingIOError, InterruptedError):
            return
        except OSError as exc:
            log.warning("credential_dns.udp_error", error_type=type(exc).__name__)
            return
        task = asyncio.get_running_loop().create_task(
            self._respond_udp(data, addr, _reply_pktinfo(ancdata))
        )
        self._udp_tasks.add(task)
        task.add_done_callback(self._udp_tasks.discard)

    async def _respond_udp(self, data: bytes, addr: tuple[str, int], pktinfo: bytes | None) -> None:
        try:
            response = await self.answer(data)
        except Exception as exc:
            log.warning("credential_dns.udp_error", error_type=type(exc).__name__)
            response = _servfail(data)
        sock = self._udp_sock
        if not response or sock is None:
            return
        # Truncate rather than fragment; a client that needs the full answer
        # retries over TCP, which we also serve.
        payload = response[:_MAX_UDP_RESPONSE]
        try:
            if pktinfo is None:
                sock.sendto(payload, addr)
            else:
                assert _IP_PKTINFO is not None
                sock.sendmsg([payload], [(socket.IPPROTO_IP, _IP_PKTINFO, pktinfo)], 0, addr)
        except OSError as exc:
            log.warning("credential_dns.udp_error", error_type=type(exc).__name__)

    async def _handle_tcp(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._tcp_conns.add(task)
        try:
            while True:
                header = await asyncio.wait_for(reader.readexactly(2), _TCP_IDLE_TIMEOUT_S)
                (length,) = struct.unpack("!H", header)
                if length == 0 or length > _MAX_TCP_MESSAGE:
                    return
                query = await asyncio.wait_for(reader.readexactly(length), _TCP_IDLE_TIMEOUT_S)
                response = await self.answer(query)
                if not response:
                    return
                writer.write(struct.pack("!H", len(response)) + response)
                await writer.drain()
        except (TimeoutError, asyncio.IncompleteReadError, ConnectionError):
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Type only — never the message, and never the query bytes.
            log.warning("credential_dns.tcp_error", error_type=type(exc).__name__)
        finally:
            if task is not None:
                self._tcp_conns.discard(task)
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()


class _ForwardProtocol(asyncio.DatagramProtocol):
    """One-shot upstream relay for non-credential queries."""

    def __init__(self) -> None:
        self.reply: asyncio.Future[bytes] = asyncio.get_running_loop().create_future()

    def datagram_received(self, data: bytes, addr: tuple[str | int, ...]) -> None:
        if not self.reply.done():
            self.reply.set_result(data)

    def error_received(self, exc: Exception) -> None:
        if not self.reply.done():
            self.reply.set_exception(exc)


__all__ = [
    "CREDENTIAL_SENTINEL_IP",
    "CredentialDnsError",
    "CredentialDnsResolver",
]
