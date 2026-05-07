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
    cap = structlog.testing.LogCapture()
    structlog.configure(processors=[cap])
    try:
        yield cap
    finally:
        structlog.reset_defaults()


@pytest.fixture
def installed(
    captured_atexit: list[Callable[[], None]],
    capture_logs: structlog.testing.LogCapture,
) -> tuple[list[Callable[[], None]], structlog.testing.LogCapture]:
    """Run :func:`install_exit_diagnostics` inside an asyncio loop."""

    async def _run() -> None:
        install_exit_diagnostics(structlog.get_logger("aios.worker"))

    asyncio.run(_run())
    return captured_atexit, capture_logs


def test_enables_faulthandler(installed: object) -> None:
    assert faulthandler.is_enabled()


def test_routes_loop_exception_through_log(
    captured_atexit: list[Callable[[], None]],
    capture_logs: structlog.testing.LogCapture,
) -> None:
    async def _run() -> None:
        install_exit_diagnostics(structlog.get_logger("aios.worker"))
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


def test_atexit_hook_emits_worker_exit(
    installed: tuple[list[Callable[[], None]], structlog.testing.LogCapture],
) -> None:
    captured_atexit, capture_logs = installed
    assert len(captured_atexit) == 1

    capture_logs.entries.clear()
    captured_atexit[0]()

    matches: list[dict[str, Any]] = [
        e for e in capture_logs.entries if e.get("event") == "worker.exit"
    ]
    assert len(matches) == 1
    assert matches[0]["log_level"] == "info"
