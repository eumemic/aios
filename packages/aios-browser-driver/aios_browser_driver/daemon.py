"""``aios-browser-driver`` — the image CMD.

Binds the request socket and serves. In this skeleton the host answers only
``status``; PR2 swaps in the real Chromium-backed ``BrowserHost`` (same
:class:`~aios_browser_driver.dispatch.Host` protocol), leaving this wiring
unchanged.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
import sys

from ulid import ULID

from aios_browser_driver.browser_protocol import BrowserRequest, BrowserResponse
from aios_browser_driver.dispatch import dispatch
from aios_browser_driver.server import serve

log = logging.getLogger("aios_browser_driver")


class SkeletonHost:
    """The PR1 placeholder host: answers ``status`` with a valid envelope and
    raises ``NotImplementedError`` (→ ``unknown_op``) for every other op until
    PR2 wires Chromium. It exists so the image boots, the socket serves, and the
    invocation/exit-code contract is exercisable before the browser lands."""

    def __init__(self) -> None:
        self.boot = str(ULID())
        self.epoch = 0

    async def handle(self, request: BrowserRequest, *, deadline: float) -> BrowserResponse:
        if request.op == "status":
            return BrowserResponse(ok=True, boot=self.boot, epoch=self.epoch)
        raise NotImplementedError(request.op)


async def _run() -> None:
    host = SkeletonHost()
    ready = asyncio.Event()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)

    server_task = asyncio.create_task(serve(lambda raw: dispatch(raw, host), ready=ready))

    # Wait for the socket to bind — but if serve() dies during startup (e.g.
    # a PermissionError binding /run/aios as uid 1000), re-raise its exception
    # so the process CRASHES visibly instead of hanging forever on ready.wait()
    # with a live-but-never-serving container.
    ready_task = asyncio.create_task(ready.wait())
    done, _ = await asyncio.wait({ready_task, server_task}, return_when=asyncio.FIRST_COMPLETED)
    if server_task in done:
        ready_task.cancel()
        await server_task  # startup failed — surface the cause
    log.info("driver ready (boot=%s)", host.boot)

    stop_task = asyncio.create_task(stop.wait())
    done, _ = await asyncio.wait({stop_task, server_task}, return_when=asyncio.FIRST_COMPLETED)
    stop_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await stop_task
    if server_task in done:
        await server_task  # serve() exited on its own — re-raise any error
    else:
        server_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await server_task


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_run())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
