"""The driver's AF_UNIX request server: one newline-delimited JSON request per
connection, one JSON response line back, then close.

Bound BEFORE the browser launches so ``browser-cli`` (which connect-retries)
can attach the instant the container starts; the per-op work then awaits
browser readiness inside the handler.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import Awaitable, Callable

from aios_browser_driver.sockpath import socket_path

# A request line is small (op + args + a bounded text field); this guards the
# reader against a peer that never sends a newline.
_MAX_REQUEST_BYTES = 1_048_576

DispatchFn = Callable[[str], Awaitable[str]]


async def _handle(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter, dispatch: DispatchFn
) -> None:
    try:
        try:
            raw = await reader.readline()
        except (ValueError, asyncio.LimitOverrunError):
            # A line longer than the read limit (no newline within it) is a
            # malformed request, not a transport fault: answer with an envelope
            # (dispatch rejects a non-JSON line as invalid_request) instead of
            # dropping the connection, so the exit-0/ok:false contract holds.
            raw = b"\x00"
        if not raw:
            return
        response = await dispatch(raw.decode("utf-8", "replace").rstrip("\n"))
        writer.write(response.encode("utf-8") + b"\n")
        await writer.drain()
    finally:
        writer.close()
        with contextlib.suppress(OSError, asyncio.CancelledError):
            await writer.wait_closed()


async def serve(dispatch: DispatchFn, *, ready: asyncio.Event | None = None) -> None:
    """Bind the socket (from ``socket_path()``) and serve forever. Sets
    ``ready`` once bound so the caller can sequence startup; runs until
    cancelled."""
    path = socket_path()
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    # A stale socket from a prior boot would refuse ``bind``.
    with contextlib.suppress(FileNotFoundError):
        os.unlink(path)

    server = await asyncio.start_unix_server(
        lambda r, w: _handle(r, w, dispatch), path=path, limit=_MAX_REQUEST_BYTES
    )
    os.chmod(path, 0o600)
    if ready is not None:
        ready.set()
    try:
        async with server:
            await server.serve_forever()
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(path)
