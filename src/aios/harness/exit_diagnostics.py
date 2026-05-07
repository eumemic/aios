"""Wire faulthandler, asyncio loop exception handler, and atexit so any
worker-process exit produces an auditable log line.

Without these, native crashes, unretrieved task exceptions, and ordinary
process exits can all leave zero trace in the structured log stream —
exactly the failure shape that made the silent-exit incident
undiagnosable. Lives in its own module (separate from
:mod:`aios.harness.worker`) so unit tests can exercise it without
pulling in :mod:`aios.harness.procrastinate_app`'s module-level
``Settings()`` validation.
"""

from __future__ import annotations

import asyncio
import atexit
import faulthandler
from typing import Any

import structlog


def install_exit_diagnostics(log: structlog.stdlib.BoundLogger) -> None:
    faulthandler.enable()

    def _on_loop_exception(_loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        exc = context.get("exception")
        log.error(
            "worker.task_exception",
            message=context.get("message"),
            error_type=type(exc).__name__ if exc is not None else None,
            error=str(exc) if exc is not None else None,
            exc_info=exc,
        )

    asyncio.get_running_loop().set_exception_handler(_on_loop_exception)

    atexit.register(lambda: log.info("worker.exit"))
