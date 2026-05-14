"""Cover the SIGINT/SIGTERM shutdown path and the daemon's process-group
isolation.

Without these fixes, ``pkill -f aios_signal`` would kill the Python
connector but leave the signal-cli JVM running with its TCP port and
SQLite lock still held — a footgun observed during PR 8 smoke (#10).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any

import pytest

from aios_signal import daemon as daemon_module
from aios_signal.__main__ import _serve


class _StubConnector:
    """Minimal stand-in for :class:`SignalConnector` exposing only ``run``."""

    def __init__(self, behavior: Awaitable[None]) -> None:
        self._behavior = behavior

    async def run(self) -> None:
        await self._behavior


async def test_serve_stops_when_stop_event_is_set() -> None:
    """A signal handler flipping ``stop`` cancels the connector task so
    its ``async with`` blocks unwind and ``teardown`` runs."""
    teardown_called = asyncio.Event()

    async def _long_running() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            teardown_called.set()

    connector = _StubConnector(_long_running())
    stop = asyncio.Event()

    async def _flip_stop() -> None:
        await asyncio.sleep(0.01)
        stop.set()

    await asyncio.gather(_serve(connector, stop), _flip_stop())  # type: ignore[arg-type]
    assert teardown_called.is_set()


async def test_serve_returns_if_connector_finishes_first() -> None:
    """Connector exiting (e.g. fatal error) returns control without
    requiring the stop signal — the operator sees the original
    exception in the traceback."""
    connector = _StubConnector(asyncio.sleep(0))
    stop = asyncio.Event()
    await _serve(connector, stop)  # type: ignore[arg-type]
    # No assertion: the only requirement is that we return.


async def test_spawn_starts_new_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``start_new_session=True`` detaches the daemon from the parent's
    session + process group so a SIGTERM targeting the connector's
    pgroup does not pre-empt the daemon's controlled shutdown."""
    captured: dict[str, Any] = {}

    class _FakeProc:
        pid = 1234
        returncode: int | None = None

    async def _fake_spawn(*args: Any, **kwargs: Any) -> _FakeProc:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_spawn)

    proc = await daemon_module._spawn_subprocess(["signal-cli", "daemon"])

    assert proc.pid == 1234
    assert captured["kwargs"].get("start_new_session") is True
    assert captured["args"] == ("signal-cli", "daemon")
