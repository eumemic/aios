"""Cover the daemon's process-group isolation.

Without ``start_new_session=True`` on the signal-cli subprocess, a
``pkill -f aios_signal`` (or any process-group-targeted signal hitting
the connector) would also reach the JVM and leave the daemon's TCP port
+ SQLite lock dangling — a footgun observed during PR 8 smoke (#10).

The SIGINT/SIGTERM shutdown path itself now lives in
:meth:`HttpConnector.run_until_stopped` and is exercised in
``packages/aios-connector-http/tests/test_runner.py``
(``TestRunUntilStopped``) rather than per-connector.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from aios_signal import daemon as daemon_module


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
