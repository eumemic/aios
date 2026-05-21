"""JSON-RPC 2.0 over line-delimited TCP for the WhatsApp daemon.

:class:`RpcClient` opens a fresh connection per call.
:class:`RpcListener` holds one persistent connection for the daemon's
notification stream and yields ``(method, params)`` tuples.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
from collections.abc import AsyncIterator
from typing import Any

import structlog

from .errors import ListenerClosedError, RpcError, RpcTimeoutError

log = structlog.get_logger(__name__)


class RpcClient:
    """Fresh-TCP-per-call JSON-RPC client for the WhatsApp daemon."""

    def __init__(self, host: str, port: int, *, timeout: float = 30.0) -> None:
        self._host = host
        self._port = port
        self._timeout = timeout
        self._ids = itertools.count(1)

    async def call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Invoke ``method`` with ``params``. Returns the ``result`` field.

        Raises :class:`RpcTimeoutError` if the call exceeds ``timeout``.
        Raises :class:`RpcError` on transport failure or an RPC error response.
        """
        request: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
            "id": next(self._ids),
        }
        if params is not None:
            request["params"] = params

        try:
            return await asyncio.wait_for(self._call(request), timeout=self._timeout)
        except TimeoutError as e:
            raise RpcTimeoutError(f"RPC {method!r} timed out after {self._timeout}s") from e

    async def _call(self, request: dict[str, Any]) -> Any:
        try:
            reader, writer = await asyncio.open_connection(self._host, self._port)
        except OSError as e:
            raise RpcError(f"failed to connect to {self._host}:{self._port}: {e}") from e

        try:
            writer.write(json.dumps(request).encode("utf-8") + b"\n")
            await writer.drain()
            try:
                line = await reader.readline()
            except OSError as e:
                raise RpcError(f"RPC read failed: {e}") from e
            if not line:
                raise RpcError("RPC connection closed before response")
            try:
                message = json.loads(line)
            except json.JSONDecodeError as e:
                raise RpcError(f"RPC returned non-JSON: {line!r}") from e
            if "error" in message:
                err = message["error"]
                if isinstance(err, dict):
                    raise RpcError(
                        err.get("message", str(err)),
                        code=err.get("code"),
                        data=err.get("data"),
                    )
                raise RpcError(str(err))
            return message.get("result")
        finally:
            writer.close()
            with contextlib.suppress(OSError):
                await writer.wait_closed()


class RpcListener:
    """Persistent-connection listener for the daemon's notification stream."""

    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = port
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None

    async def connect(self) -> None:
        """Open the persistent TCP connection. Call once before iterating."""
        try:
            self._reader, self._writer = await asyncio.open_connection(self._host, self._port)
        except OSError as e:
            raise ListenerClosedError(
                f"failed to connect listener to {self._host}:{self._port}: {e}"
            ) from e

    async def notifications(self) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """Yield ``(method, params)`` pairs for each notification frame.

        RPC responses (frames carrying ``id``) and frames with non-string
        ``method`` / non-dict ``params`` are dropped.  Raises
        :class:`ListenerClosedError` when the connection drops.
        """
        if self._reader is None:
            raise ListenerClosedError("listener not connected")
        while True:
            try:
                line = await self._reader.readline()
            except OSError as e:
                raise ListenerClosedError(f"listener read failed: {e}") from e
            if not line:
                raise ListenerClosedError("listener connection closed")
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                log.warning("rpc.listener.bad_json", raw=line[:200].decode("latin-1"))
                continue
            method = message.get("method")
            if not isinstance(method, str) or not method:
                continue
            params = message.get("params")
            if not isinstance(params, dict):
                log.warning("rpc.listener.bad_params", method=method)
                continue
            yield method, params

    async def aclose(self) -> None:
        if self._writer is not None:
            self._writer.close()
            with contextlib.suppress(OSError):
                await self._writer.wait_closed()
            self._writer = None
            self._reader = None
