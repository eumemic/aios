"""``python -m aios_signal`` entry point.

Reads ``AIOS_URL`` and ``AIOS_RUNTIME_TOKEN`` from env (the SDK does
this automatically inside ``HttpConnector.__init__``).  Deployment-shape
fields like ``AIOS_SIGNAL_CONFIG_DIR`` feed pydantic-settings; the
phone (the account identity) lives on each connection's encrypted
secrets and is fetched per-connection in ``serve_connection``.

SIGINT and SIGTERM are trapped at runtime so ``SignalConnector.teardown``
gets a chance to fire — ``asyncio.run`` traps SIGINT by default but
not SIGTERM, and a default SIGTERM would kill Python before the daemon
subprocess is cleaned up.  The trap converts the signal to a cancel on
the connector task, the cancellation unwinds the TaskGroup in
:meth:`SignalConnector.run`, and the surrounding ``try/finally`` runs
``teardown`` which sends SIGTERM to the daemon subprocess and waits.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal

from .config import Settings
from .connector import SignalConnector


async def _serve(connector: SignalConnector, stop: asyncio.Event) -> None:
    """Run the connector until cancelled or ``stop`` is set.

    Factored out of :func:`main` so a unit test can exercise the
    cancel-on-stop path without process-level signal handling.
    """
    connector_task = asyncio.create_task(connector.run(), name="aios-signal-run")
    stop_task = asyncio.create_task(stop.wait(), name="aios-signal-stop-wait")
    try:
        await asyncio.wait(
            {connector_task, stop_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
    finally:
        stop_task.cancel()
        if not connector_task.done():
            connector_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await connector_task


async def _async_main() -> None:
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    await _serve(SignalConnector(Settings()), stop)


def main() -> None:
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
