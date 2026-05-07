"""Worker exit-diagnostics wiring (#268).

The silent-exit failure mode the issue describes was identifiable by *no*
log output for ~2 hours after the worker died. These tests pin the three
mechanisms :func:`install_exit_diagnostics` wires to ensure that any
future exit produces something we can audit.
"""

from __future__ import annotations

import asyncio
import atexit
import faulthandler
from collections.abc import Callable, Iterator
from typing import Any

import pytest
import structlog

from aios.harness.worker import install_exit_diagnostics


@pytest.fixture
def captured_atexit(monkeypatch: pytest.MonkeyPatch) -> list[Callable[[], None]]:
    """Replace :func:`atexit.register` with a list-appender for the test.

    Real registration would queue a callback that fires at interpreter
    shutdown — long after pytest collects results — and would emit a
    ``worker.exit`` line into whatever logger config is live then.
    """
    captured: list[Callable[[], None]] = []
    monkeypatch.setattr(atexit, "register", lambda fn: captured.append(fn) or fn)
    return captured


@pytest.fixture
def capture_logs() -> Iterator[structlog.testing.LogCapture]:
    """Pin a structlog ``LogCapture`` so emitted events are inspectable."""
    cap = structlog.testing.LogCapture()
    structlog.configure(processors=[cap])
    try:
        yield cap
    finally:
        structlog.reset_defaults()


def test_install_enables_faulthandler(captured_atexit: list[Callable[[], None]]) -> None:
    """``faulthandler.enable()`` runs idempotently and stays enabled.

    We can't synthesize a ``SIGSEGV`` without killing the test runner, so
    we settle for the observable proxy: ``is_enabled()`` is True after
    the helper runs.
    """

    async def _run() -> None:
        log = structlog.get_logger("aios.worker")
        install_exit_diagnostics(log)

    asyncio.run(_run())
    assert faulthandler.is_enabled()
    assert len(captured_atexit) == 1


def test_install_routes_loop_exception_through_log(
    captured_atexit: list[Callable[[], None]],
    capture_logs: structlog.testing.LogCapture,
) -> None:
    """Uncaught task exceptions land in structlog as ``worker.task_exception``.

    Without this, asyncio's default handler prints a free-form
    ``Task exception was never retrieved`` warning to stderr — invisible
    to a JSON log consumer and absent until the offending Task is
    garbage-collected.
    """

    async def _run() -> None:
        log = structlog.get_logger("aios.worker")
        install_exit_diagnostics(log)
        loop = asyncio.get_running_loop()
        loop.call_exception_handler(
            {"message": "synthetic task failure", "exception": RuntimeError("boom")}
        )

    asyncio.run(_run())

    matches = [e for e in capture_logs.entries if e.get("event") == "worker.task_exception"]
    assert len(matches) == 1
    entry = matches[0]
    assert entry["log_level"] == "error"
    assert entry["message"] == "synthetic task failure"
    assert entry["error"] == "boom"
    assert entry["error_type"] == "RuntimeError"


def test_install_registers_atexit_hook(
    captured_atexit: list[Callable[[], None]],
    capture_logs: structlog.testing.LogCapture,
) -> None:
    """The atexit hook is the safety net for "did the process actually exit?"

    Invokes the captured callback manually (rather than waiting for
    interpreter shutdown) and asserts a ``worker.exit`` log line emerges.
    """

    async def _run() -> None:
        log = structlog.get_logger("aios.worker")
        install_exit_diagnostics(log)

    asyncio.run(_run())

    assert len(captured_atexit) == 1, "expected exactly one atexit registration"

    capture_logs.entries.clear()
    captured_atexit[0]()

    matches: list[dict[str, Any]] = [
        e for e in capture_logs.entries if e.get("event") == "worker.exit"
    ]
    assert len(matches) == 1
    assert matches[0]["log_level"] == "info"
