"""A recording TLS upstream for the secret-egress swap e2e.

The secret-egress proxy terminates the sandbox's TLS, swaps the opaque
placeholder for the real vaulted secret, and forwards the request to the
*real* credential host over a freshly-verified upstream TLS connection. Every
existing test stops short of that last hop: the unit suite mocks the httpx
upstream transport, and the materialized-placeholder e2e only proves the
placeholder reaches the container env. Neither observes the swap **fire**.

:class:`RecorderUpstream` is the missing observer. It stands up a real
asyncio TLS server that presents a leaf for an allow-listed credential host
(minted from the worker's egress CA, so the proxy's upstream verification —
which pins SNI + cert to the real hostname — passes exactly as it would
against the genuine host), records every inbound request it receives, and
returns a fixed 200 body. Point the proxy's upstream resolver at it (see
``redirect_secret_egress_upstream``) and a real in-sandbox ``curl
https://<host>`` carrying the placeholder in an Authorization header lands
here with the **real secret** swapped in — or, if the swap never fired, the
placeholder. The test asserts on what was recorded.

Bound to ``127.0.0.1`` in the worker/test process; never container-visible.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import ssl
import tempfile
from dataclasses import dataclass, field

import h11
from cryptography.hazmat.primitives import serialization

from aios.sandbox.egress_ca import EgressCA, get_egress_ca
from tests.helpers.tls import mint_leaf


@dataclass
class RecordedRequest:
    """One inbound request the upstream observed, post-swap."""

    method: str
    target: str
    headers: list[tuple[str, str]]
    body: bytes

    def header(self, name: str) -> str | None:
        """First value of header ``name`` (case-insensitive), or ``None``."""
        lname = name.lower()
        for k, v in self.headers:
            if k.lower() == lname:
                return v
        return None


@dataclass
class RecorderUpstream:
    """A TLS server that records requests and replies 200.

    ``hostname`` is the credential host whose leaf it presents. ``ca`` defaults
    to the worker's process egress CA (the same one the proxy mints its own
    leaves from and verifies upstream against). The server binds an ephemeral
    loopback port; ``ip``/``port`` are populated after :meth:`start`.
    """

    hostname: str
    ca: EgressCA = field(default_factory=get_egress_ca)
    response_status: int = 200
    response_body: bytes = b'{"recorder":"ok"}'
    requests: list[RecordedRequest] = field(default_factory=list)
    _server: asyncio.Server | None = field(default=None, init=False, repr=False)
    _leaf_path: str | None = field(default=None, init=False, repr=False)
    ip: str = field(default="127.0.0.1", init=False)
    port: int = field(default=0, init=False)

    async def start(self) -> None:
        leaf, key = mint_leaf(self.ca, self.hostname)
        pem = leaf.public_bytes(serialization.Encoding.PEM) + key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        fd, path = tempfile.mkstemp(suffix=".pem")
        with os.fdopen(fd, "wb") as fh:
            fh.write(pem)
        self._leaf_path = path
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(path)
        self._server = await asyncio.start_server(self._handle, self.ip, 0, ssl=ctx)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            self._server.abort_clients()
            await self._server.wait_closed()
            self._server = None
        if self._leaf_path is not None:
            try:
                os.unlink(self._leaf_path)
            finally:
                self._leaf_path = None

    async def __aenter__(self) -> RecorderUpstream:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            conn = h11.Connection(h11.SERVER)
            request: h11.Request | None = None
            body = bytearray()
            while True:
                event = conn.next_event()
                if event is h11.NEED_DATA:
                    data = await asyncio.wait_for(reader.read(65536), 30.0)
                    if not data:
                        return
                    conn.receive_data(data)
                    continue
                if isinstance(event, h11.Request):
                    request = event
                elif isinstance(event, h11.Data):
                    body += event.data
                else:  # EndOfMessage / ConnectionClosed / PAUSED
                    break
            if request is None:
                return
            self.requests.append(
                RecordedRequest(
                    method=request.method.decode("ascii"),
                    target=request.target.decode("latin-1"),
                    headers=[
                        (k.decode("latin-1"), v.decode("latin-1"))
                        for k, v in request.headers.raw_items()
                    ],
                    body=bytes(body),
                )
            )
            headers = [
                (b"content-length", str(len(self.response_body)).encode()),
                (b"content-type", b"application/json"),
                (b"connection", b"close"),
            ]
            writer.write(
                conn.send(h11.Response(status_code=self.response_status, headers=headers)) or b""
            )
            writer.write(conn.send(h11.Data(data=self.response_body)) or b"")
            writer.write(conn.send(h11.EndOfMessage()) or b"")
            await writer.drain()
        except (TimeoutError, ConnectionError, ssl.SSLError, h11.ProtocolError):
            return
        finally:
            writer.close()
            with contextlib.suppress(ConnectionError, ssl.SSLError):
                await writer.wait_closed()


__all__ = ["RecordedRequest", "RecorderUpstream"]
