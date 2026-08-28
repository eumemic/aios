"""``aios-browser-driver`` — the image CMD.

Binds the request socket FIRST (so ``browser-cli`` can attach the instant the
container starts; early requests wait on the host's readiness), then launches
Chromium. Every startup or relaunch failure crashes the process visibly — a
crash-looping container beats a live one that answers nothing.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import sys

from aios_browser_driver.dispatch import dispatch
from aios_browser_driver.host import BrowserHost
from aios_browser_driver.server import serve

log = logging.getLogger("aios_browser_driver")

# The hermetic-test knob: bypasses the navigate guard's public-address check
# (never the scheme check) so e2e suites can serve fixtures from loopback.
# Unsettable through aios by construction — the browser container spec pins
# ``environment={}``.
_ALLOW_PRIVATE_NAV_ENV = "AIOS_BROWSER_DRIVER_ALLOW_PRIVATE_NAV"


async def _run() -> None:
    host = BrowserHost(allow_private_nav=os.environ.get(_ALLOW_PRIVATE_NAV_ENV) == "1")
    ready = asyncio.Event()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)

    server_task = asyncio.create_task(serve(lambda raw: dispatch(raw, host), ready=ready))

    try:
        # Wait for the socket to bind — but if serve() dies during startup
        # (e.g. a PermissionError binding /run/aios as uid 1000), re-raise its
        # exception so the process CRASHES visibly instead of hanging forever
        # on ready.wait() with a live-but-never-serving container.
        ready_task = asyncio.create_task(ready.wait())
        done, _ = await asyncio.wait({ready_task, server_task}, return_when=asyncio.FIRST_COMPLETED)
        if server_task in done:
            ready_task.cancel()
            await server_task  # startup failed — surface the cause

        await host.start()  # raises if Chromium cannot launch — crash visibly
        log.info("driver ready (boot=%s)", host.boot)

        stop_task = asyncio.create_task(stop.wait())
        failed_task = asyncio.create_task(host.failed())
        done, _ = await asyncio.wait(
            {stop_task, server_task, failed_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in (stop_task, failed_task):
            if task not in done:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        if failed_task in done:
            await failed_task  # the browser died and could not relaunch
        if server_task in done:
            await server_task  # serve() exited on its own — re-raise any error
        else:
            server_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await server_task
    finally:
        await host.close()


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
