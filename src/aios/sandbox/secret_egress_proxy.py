"""Per-session TLS-terminating egress proxy that swaps credential
placeholders for real vaulted secrets on the way out of the sandbox.

The generalization of :mod:`aios.sandbox.git_proxy` for arbitrary
``environment_variable`` credentials (#876). Where ``git_proxy`` is a
plaintext HTTP forward proxy that injects a fixed ``Authorization`` header,
this proxy **terminates TLS**: it transparently intercepts an HTTPS
connection (the host comes from the TLS ClientHello SNI), mints a leaf cert
for that host from the aios egress CA (which every sandbox already trusts,
#875), reads the request, replaces the opaque per-session placeholder token
with the real secret as a literal substring **in request headers + body
only** (never the URL path/query — the URL carries the placeholder
verbatim), forwards to the real host over a freshly-verified upstream TLS
connection, and streams the response straight back.

Security posture:

* **Fail closed everywhere.** The leaf-mint host check is THE enforcement
  point — the egress CA is unconstrained, so every sandbox trusts any leaf
  it signs; the proxy refuses to mint a leaf for a host outside the
  session's resolved allow-set (or for an absent SNI), so the handshake
  aborts before anything terminates or swaps. A blocked upstream resolution
  returns a 502 and makes no connection.
* **SNI is authoritative.** The host is taken from the ClientHello SNI and
  drives the allow-set gate, the placeholder→secret swap, AND the upstream
  connection — never the request ``Host`` header, never the original
  destination IP (which would trust sandbox-controlled DNS). The proxy
  re-resolves the SNI host itself at connect time, blocks internal/SSRF
  ranges, and pins the upstream connection to a checked IP so DNS rebinding
  can't redirect an authorized name to an attacker address.
* **Outbound-only.** Response bodies are NOT scrubbed (#881); the hard
  exfiltration boundary is the environment egress allowlist, not this proxy.
* **Secrets live only in worker memory** (sourced from
  ``ProvisioningPlan.env_var_credentials``); they never enter a log, the
  spec, or the container. Only placeholders are ever container-visible.

Named residuals (operator-facing):

* IP-literal HTTPS destinations have no SNI and are therefore rejected by
  construction — the proxy serves hostname-addressed egress only. Once #878
  redirects allowed-host traffic here, a sandbox ``curl https://<ip>`` to an
  allowed host's IP gets its handshake reset rather than connecting.
* The request body is buffered whole to swap across chunk boundaries (memory
  bound = request size); responses stay streamed. Very large request uploads
  are a documented residual.
* The upstream connection pins a single resolved IP (IPv4 preferred); no
  dual-stack failover.

Lifecycle mirrors ``git_proxy``: started per session, stopped on release /
recycle. Wiring into provisioning, the iptables chokepoint, and
recycle-on-rotation are follow-ups (#877/#878).
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import os
import re
import secrets
import socket
import ssl
import tempfile
import weakref
from collections.abc import Iterable
from typing import Any

import h11
import httpx
from cryptography.hazmat.primitives import serialization

from aios.logging import get_logger
from aios.models.vaults import parse_allowed_host_entry
from aios.sandbox.egress_ca import EgressCA, get_egress_ca, mint_server_leaf
from aios.services.vaults import ResolvedEnvVarCredential

# NOTE: ``aios.tools.url_safety`` is imported lazily inside ``_resolve_pinned_ip``
# (the only user) — not here. A module-level ``from aios.tools import ...`` makes
# the ``aios.tools`` package (whose ``__init__`` imports ``bash`` → ``sandbox.spec``,
# which binds ``SecretEgressProxy`` back from this module) part of this module's
# import graph, so a standalone ``import aios.sandbox.secret_egress_proxy`` cycles
# and ImportErrors. Deferring to call time keeps the import graph acyclic.

log = get_logger("aios.sandbox.secret_egress_proxy")

# HTTPS only — the swap path terminates TLS; plaintext :80 is never MITM'd
# (the placeholder rides through inert there). Upstream is always :443.
_UPSTREAM_PORT = 443
# Generous upstream bound — a sandbox CLI may stream a large authenticated
# upload or download. Matches git_proxy.
_UPSTREAM_TIMEOUT_S = 300.0
_READ_CHUNK = 65536
# Idle bound on a single inbound read: a client that stalls mid-request (or
# never sends one) must not pin a handler indefinitely. Bounds idle time
# between chunks, not total transfer, so a slow-but-steady upload is unaffected.
_INBOUND_IDLE_TIMEOUT_S = 60.0

# Stripped before forwarding the request upstream. ``authorization`` is
# deliberately ABSENT: the placeholder rides in it and we swap+forward it.
# ``host`` is stripped because we set it explicitly to the SNI host;
# ``content-length`` because the body length changes after the swap.
_HOP_BY_HOP_REQUEST_HEADERS: frozenset[str] = frozenset(
    {
        "host",
        "connection",
        "content-length",
        "transfer-encoding",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "upgrade",
        "expect",
    }
)

# Stripped from the response before streaming it back. ``content-encoding``
# is dropped because httpx decodes the body (we re-stream decoded bytes).
# ``content-length`` is handled separately in ``_stream_response`` (kept when
# the length is still accurate, dropped when we re-decode the body).
_HOP_BY_HOP_RESPONSE_HEADERS: frozenset[str] = frozenset(
    {
        "connection",
        "content-encoding",
        "transfer-encoding",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "upgrade",
    }
)


def _request_path(target: str) -> str:
    """The path portion of an HTTP request-target (drop ?query / #frag).

    The path gate is evaluated on the path only; the query/fragment are
    irrelevant to credential scope (and the URL is forwarded verbatim).
    """
    return target.split("?", 1)[0].split("#", 1)[0]


def _path_within_prefix(prefix: str, path: str) -> bool:
    """Segment-boundary match with dot-segment climb-out rejection (#876).

    ``/repos/eumemic`` matches ``/repos/eumemic`` and ``/repos/eumemic/...``
    but NOT ``/repos/eumemic-evil`` (segment boundary). The remainder after
    the matched prefix is attacker-shaped, so any dot-segment that would
    climb out of the subtree once the upstream normalizes — raw ``..``,
    percent-encoded ``%2e%2e``, or a ``\\``-as-``/`` variant — fails the
    gate. A plain percent-encoded slash *inside* the subtree (e.g.
    ``git/ref/heads%2Ffoo``) is legitimate and keeps matching.

    Note: a True result guarantees the path has no literal dot-segment,
    which is ALSO what makes httpx's own path normalization in
    ``build_request`` harmless (there is nothing left for it to collapse).
    Never drop the dot-segment rejection on the theory that httpx normalizes
    anyway — the two normalizers would then diverge from the upstream's.
    """
    norm = "/" + path.lstrip("/")
    p = prefix.rstrip("/")
    if not (norm == p or norm.startswith(p + "/")):
        return False
    rest = norm[len(p) :]
    decoded = re.sub(r"%2f|%5c", "/", rest, flags=re.IGNORECASE)
    decoded = re.sub(r"%2e", ".", decoded, flags=re.IGNORECASE)
    decoded = decoded.replace("\\", "/")
    return not any(seg in (".", "..") for seg in decoded.split("/"))


def _bracket(ip: str) -> str:
    """Wrap an IPv6 literal in brackets for use in a URL authority."""
    return f"[{ip}]" if ":" in ip else ip


def _apply_swaps_str(value: str, swaps: list[tuple[str, str]]) -> str:
    for placeholder, secret in swaps:
        value = value.replace(placeholder, secret)
    return value


def _apply_swaps_bytes(data: bytes, swaps: list[tuple[str, str]]) -> bytes:
    for placeholder, secret in swaps:
        data = data.replace(placeholder.encode(), secret.encode())
    return data


def _swap_header_value(name: str, value: str, swaps: list[tuple[str, str]]) -> str:
    """Swap placeholders in a single forwarded header value.

    ``Authorization: Basic <b64>`` is decoded first: HTTP Basic auth carries
    the credential as ``base64("user:pass")``, and base64 shifts byte
    boundaries so the injected ``AIOS_SECRET_PLACEHOLDER_…`` does not appear as
    a literal substring of the encoded value — a raw ``.replace()`` over the
    header is a no-op and the placeholder rides out to the remote (git over
    HTTPS, ``curl -u``, …). So when a Basic credential is present we decode the
    ``user:pass``, run the swap over the *decoded* text, and re-encode. Every
    other header value (``Bearer …``, ``X-Api-Key: …``, custom schemes) keeps
    the literal-substring swap.
    """
    if name.lower() == "authorization":
        decoded = _decode_basic_credential(value)
        if decoded is not None:
            swapped = _apply_swaps_str(decoded, swaps)
            reencoded = base64.b64encode(swapped.encode("latin-1")).decode("ascii")
            return f"Basic {reencoded}"
    return _apply_swaps_str(value, swaps)


def _decode_basic_credential(value: str) -> str | None:
    """Decode an ``Authorization: Basic <b64>`` value to its ``user:pass``.

    Returns ``None`` (leaving the value to the literal-substring swap) when the
    scheme is not ``Basic`` or the credential is not valid base64 / not
    decodable text — fail open to the existing path rather than mangle a header
    we don't understand.
    """
    parts = value.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "basic":
        return None
    try:
        raw = base64.b64decode(parts[1].strip(), validate=True)
        return raw.decode("latin-1")
    except (binascii.Error, ValueError):
        return None


async def _resolve_addrinfos(host: str, port: int) -> list[tuple[Any, ...]]:
    """Async DNS resolution seam (monkeypatched in tests)."""
    loop = asyncio.get_running_loop()
    return list(await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM))


async def _resolve_pinned_ip(host: str, port: int) -> str | None:
    """Re-resolve ``host`` at connect time and return ONE safe IP to pin the
    upstream connection to, or ``None`` if it must not be reached.

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


def _h11_send(conn: h11.Connection, event: h11.Event) -> bytes:
    """``conn.send`` returns ``None`` for events that frame no bytes."""
    data = conn.send(event)
    return data if data is not None else b""


class SecretEgressProxy:
    """One per session; holds the placeholder→secret map in memory, terminates
    TLS for allow-listed hosts, and swaps placeholders for secrets on egress.

    Lifetime is bounded by the session container's lifetime: when the
    container is released or recycled the proxy is stopped and the in-memory
    secret map is dropped.
    """

    def __init__(self, credentials: Iterable[ResolvedEnvVarCredential]) -> None:
        # Flatten (cred, allowed-host entry) into swap rules. The host is
        # lowercased ONCE here so the SNI gate and the swap map compare
        # against a normalized host — parse_allowed_host_entry stores the
        # host as-given (mixed case is valid), and the incoming SNI is
        # lowercased, so the stored side must be too or a legitimately
        # allowed mixed-case host fails closed.
        self._rules: list[tuple[str, str | None, str, str]] = []
        for cred in credentials:
            for entry in cred.allowed_hosts:
                host, prefix = parse_allowed_host_entry(entry)
                self._rules.append((host.lower(), prefix, cred.placeholder, cred.secret_value))
        # The leaf-mint gate AND the cache bound: only an SNI host in this set
        # ever reaches the mint path, so a hostile sandbox can't flood
        # distinct SNIs to balloon the leaf cache.
        self._allowed_hosts: frozenset[str] = frozenset(h for h, _, _, _ in self._rules)
        self._ca: EgressCA = get_egress_ca()
        self._leaf_ctx: dict[str, ssl.SSLContext] = {}
        self._sni: weakref.WeakKeyDictionary[ssl.SSLObject, str] = weakref.WeakKeyDictionary()
        self._client = httpx.AsyncClient(
            timeout=_UPSTREAM_TIMEOUT_S,
            follow_redirects=False,
            # Force a fresh upstream handshake per request. Pooled connections
            # key on IP-origin only and silently ignore a later request's
            # sni_hostname on reuse, so two allowed hosts sharing a pinned IP
            # could ride one another's verified TLS session — cross-host
            # secret confusion. A zero keepalive cap makes the per-request
            # sni_hostname always authoritative.
            limits=httpx.Limits(max_keepalive_connections=0),
        )
        # Per-session URL guard (parity with git_proxy): not a true isolation
        # primitive, just limits blast radius if the port is exposed.
        self._secret = secrets.token_urlsafe(32)
        self._server: asyncio.Server | None = None
        self._port: int | None = None
        # In-flight connection handlers, tracked so stop() can cancel them
        # (releasing the secret map) rather than block on wait_closed().
        self._conns: set[asyncio.Task[None]] = set()

    @property
    def secret(self) -> str:
        return self._secret

    @property
    def port(self) -> int:
        assert self._port is not None, "SecretEgressProxy.start() has not completed"
        return self._port

    async def start(self) -> None:
        """Bind ``0.0.0.0:0`` and begin serving TLS. ``asyncio.start_server``
        binds synchronously, so the port is available the instant the await
        returns and a bind failure raises immediately (no async bind window
        to poll)."""
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.sni_callback = self._sni_callback
        try:
            self._server = await asyncio.start_server(self._handle, "0.0.0.0", 0, ssl=ctx)
            self._port = self._server.sockets[0].getsockname()[1]
        except BaseException:
            # Bind never completed; the caller drops its reference, so nothing
            # else calls stop() — close the httpx client here or it leaks.
            await self.stop()
            raise
        log.info("secret_egress_proxy.started", port=self._port)

    async def stop(self) -> None:
        """Stop serving and drop the in-memory secret map.

        ``Server.wait_closed()`` only JOINS in-flight handlers, so a handler
        streaming a slow upstream would block teardown for up to
        ``_UPSTREAM_TIMEOUT_S`` (and keep the secret swap map alive that
        whole time). ``abort_clients()`` tears down every accepted transport —
        including a connection whose TLS handshake is still completing and
        whose handler is therefore not yet in ``_conns`` — and cancelling the
        tracked handlers unwinds their frames promptly. None-guarded so a
        failed-bind start() can still clean up.
        """
        try:
            if self._server is not None:
                self._server.close()
                self._server.abort_clients()
                tasks = list(self._conns)
                for task in tasks:
                    task.cancel()
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                await self._server.wait_closed()
        finally:
            # Drop the secret map and per-host leaves (the documented contract);
            # they are otherwise only reclaimed when the proxy object is GC'd.
            self._rules = []
            self._leaf_ctx = {}
            await self._client.aclose()
        log.info("secret_egress_proxy.stopped", port=self._port)

    def _sni_callback(
        self, sslobj: ssl.SSLObject, server_name: str | None, sslctx: ssl.SSLContext
    ) -> int | None:
        """Select (mint) the per-host leaf during the TLS handshake.

        THE fail-closed enforcement point: refuse to mint for an absent SNI
        or a host outside the allow-set, so the handshake aborts before
        anything terminates. Returns an OpenSSL alert (handshake reset) on
        any refusal or error; ``None`` on success.
        """
        try:
            if not server_name:
                return ssl.ALERT_DESCRIPTION_UNRECOGNIZED_NAME
            host = server_name.lower()
            if host not in self._allowed_hosts:
                return ssl.ALERT_DESCRIPTION_UNRECOGNIZED_NAME
            sslobj.context = self._leaf_context(host)
            self._sni[sslobj] = host
            return None
        except Exception:
            log.warning("secret_egress_proxy.sni_callback_error", server_name=server_name)
            return ssl.ALERT_DESCRIPTION_INTERNAL_ERROR

    def _leaf_context(self, host: str) -> ssl.SSLContext:
        """A TLS server context presenting a leaf for ``host``, minted once
        per host and cached for the session.

        ``load_cert_chain`` has no in-memory variant, so the leaf key is
        written to a 0600 temp file and unlinked immediately — always, even
        if the load raises. The key is an ephemeral per-host leaf (not the CA
        key, not the secret)."""
        cached = self._leaf_ctx.get(host)
        if cached is not None:
            return cached
        cert, key = mint_server_leaf(self._ca, host)
        pem = cert.public_bytes(serialization.Encoding.PEM) + key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        fd, tmp = tempfile.mkstemp(suffix=".pem")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(pem)
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(tmp)
        finally:
            os.unlink(tmp)
        self._leaf_ctx[host] = ctx
        return ctx

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._conns.add(task)
        try:
            await self._serve_one(reader, writer)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Log only the exception TYPE, never str(exc): post-swap, an
            # exception message can carry the real secret (e.g. an illegal-
            # header error quotes the offending header value verbatim).
            log.warning("secret_egress_proxy.handler_error", error_type=type(exc).__name__)
        finally:
            if task is not None:
                self._conns.discard(task)
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    async def _serve_one(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        ssl_object = writer.get_extra_info("ssl_object")
        host = self._sni.get(ssl_object) if ssl_object is not None else None
        if host is None:
            # The sni_callback gate guarantees an allow-set host before we get
            # here; this is purely defensive (e.g. a dropped weakref).
            return

        conn = h11.Connection(h11.SERVER)
        # Read the head FIRST, gate, and only THEN read the body. A request to
        # a host whose egress is blocked gets a 502 before its body is ever
        # drained, so a sandbox can't force the worker to buffer an arbitrary
        # payload toward a host it will never reach.
        request = await self._read_head(conn, reader)
        if request is None:
            return  # client hung up / stalled before a complete request head

        method = request.method.decode("ascii")
        target = request.target.decode("latin-1")  # request-target is bytes
        path = _request_path(target)

        # Per-request swap map: only creds whose host matches the SNI host AND
        # whose path gate passes. A non-matching cred's placeholder rides
        # through unswapped (clean fail — blocking the host is #878's job).
        swaps = [
            (placeholder, secret)
            for h, prefix, placeholder, secret in self._rules
            if h == host and (prefix is None or _path_within_prefix(prefix, path))
        ]

        pinned = await _resolve_pinned_ip(host, _UPSTREAM_PORT)
        if pinned is None:
            # A final status before the body also tells an Expect: 100-continue
            # client not to send it (RFC 7231 §5.1.1).
            await self._send_simple(conn, writer, 502, b"egress blocked")
            return

        # We are the TLS-terminating origin, so we own the 100-continue
        # handshake the upstream never sees (we strip Expect): tell the client
        # to send the body now that the host has cleared the egress gate.
        if conn.client_is_waiting_for_100_continue:
            writer.write(_h11_send(conn, h11.InformationalResponse(status_code=100, headers=[])))
            await writer.drain()

        body = await self._read_body(conn, reader)
        if body is None:
            return  # client stalled mid-body

        fwd_headers = self._forward_headers(request.headers.raw_items(), host, swaps)
        swapped_body = _apply_swaps_bytes(body, swaps)
        # The URL carries the request-target verbatim — the placeholder is
        # NEVER swapped in the path/query, only in headers and body.
        url = f"https://{_bracket(pinned)}:{_UPSTREAM_PORT}{target}"
        try:
            upstream_req = self._client.build_request(
                method,
                url,
                headers=fwd_headers,
                content=swapped_body,
                # Pin the connection to the resolved IP (URL host) while
                # keeping SNI + cert verification on the real hostname.
                extensions={"sni_hostname": host},
            )
            upstream = await self._client.send(upstream_req, stream=True)
        except Exception as exc:
            # Broad by design: build_request can raise non-RequestError (e.g.
            # UnicodeEncodeError on a non-ASCII swapped header value), and the
            # model should see a clean 502 and retry rather than a reset
            # connection. Log only the type — str(exc) can carry the secret.
            log.warning(
                "secret_egress_proxy.upstream_error", host=host, error_type=type(exc).__name__
            )
            await self._send_simple(conn, writer, 502, b"upstream error")
            return
        try:
            await self._stream_response(conn, writer, upstream)
        finally:
            await upstream.aclose()

    async def _read_until_data(self, reader: asyncio.StreamReader) -> bytes | None:
        """One inbound read bounded by the idle timeout; ``None`` on stall."""
        try:
            return await asyncio.wait_for(reader.read(_READ_CHUNK), _INBOUND_IDLE_TIMEOUT_S)
        except TimeoutError:
            return None

    async def _read_head(
        self, conn: h11.Connection, reader: asyncio.StreamReader
    ) -> h11.Request | None:
        """Read up to and including the request head (the ``Request`` event)."""
        while True:
            event = conn.next_event()
            if event is h11.NEED_DATA:
                data = await self._read_until_data(reader)
                if not data:
                    return None  # EOF or idle timeout before the head
                conn.receive_data(data)
                continue
            if isinstance(event, h11.Request):
                return event
            return None  # connection closed before a request

    async def _read_body(self, conn: h11.Connection, reader: asyncio.StreamReader) -> bytes | None:
        """Read the request body (h11 ``Data`` events) up to ``EndOfMessage``.

        The body is buffered whole so the placeholder can be swapped across
        chunk boundaries; this runs only after the egress gate has passed.
        """
        body = bytearray()
        while True:
            event = conn.next_event()
            if event is h11.NEED_DATA:
                data = await self._read_until_data(reader)
                if data is None:
                    return None  # idle timeout mid-body
                conn.receive_data(data)
                continue
            if isinstance(event, h11.Data):
                body += event.data
            else:  # EndOfMessage / ConnectionClosed / PAUSED
                return bytes(body)

    def _forward_headers(
        self, raw_items: list[tuple[bytes, bytes]], host: str, swaps: list[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        fwd: list[tuple[str, str]] = []
        for raw_name, raw_value in raw_items:
            name = raw_name.decode("latin-1")
            if name.lower() in _HOP_BY_HOP_REQUEST_HEADERS:
                continue
            fwd.append((name, _swap_header_value(name, raw_value.decode("latin-1"), swaps)))
        # Forward to (and name) the SNI host we terminated for.
        fwd.append(("host", host))
        return fwd

    async def _stream_response(
        self, conn: h11.Connection, writer: asyncio.StreamWriter, upstream: httpx.Response
    ) -> None:
        # httpx decodes content-encoding in aiter_bytes, so a content-length
        # from an encoded response would be wrong — drop it then and let h11
        # re-frame as chunked. With no content-encoding the length is accurate;
        # keep it so a known-length or bodiless response (e.g. a HEAD reply) is
        # length-framed instead of carrying a spurious Transfer-Encoding.
        had_encoding = any(name.lower() == b"content-encoding" for name, _ in upstream.headers.raw)
        resp_headers: list[tuple[bytes, bytes]] = []
        for name, value in upstream.headers.raw:
            lname = name.decode("latin-1").lower()
            if lname == "content-length":
                if not had_encoding:
                    resp_headers.append((name, value))
                continue
            if lname in _HOP_BY_HOP_RESPONSE_HEADERS:
                continue
            resp_headers.append((name, value))
        resp_headers.append((b"connection", b"close"))
        writer.write(
            _h11_send(conn, h11.Response(status_code=upstream.status_code, headers=resp_headers))
        )
        # Responses are NOT scrubbed (#881) — stream straight back.
        async for chunk in upstream.aiter_bytes():
            framed = _h11_send(conn, h11.Data(data=chunk))
            if framed:
                writer.write(framed)
                await writer.drain()
        writer.write(_h11_send(conn, h11.EndOfMessage()))
        await writer.drain()

    async def _send_simple(
        self, conn: h11.Connection, writer: asyncio.StreamWriter, status: int, body: bytes
    ) -> None:
        if conn.our_state is not h11.SEND_RESPONSE:
            return
        headers = [(b"content-length", str(len(body)).encode()), (b"connection", b"close")]
        writer.write(_h11_send(conn, h11.Response(status_code=status, headers=headers)))
        writer.write(_h11_send(conn, h11.Data(data=body)))
        writer.write(_h11_send(conn, h11.EndOfMessage()))
        await writer.drain()


__all__ = ["SecretEgressProxy"]
