"""Unit tests for pure helpers in daemon.py.

Covers ``subprocess_alive`` (transient-listener-drop vs. fatal-crash
discrimination) and the stdout drain's JSON-exception observability
hook.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

from structlog.testing import capture_logs

from aios_signal import daemon as daemon_module
from aios_signal.daemon import SignalDaemon
from aios_signal.errors import DaemonCrashError


def _daemon(phones: list[str], config_dir: Path) -> SignalDaemon:
    return SignalDaemon(
        phones=phones,
        config_dir=config_dir,
        cli_bin="signal-cli",
        host="127.0.0.1",
        port=0,
    )


# ── subprocess_alive: distinguishes transient listener drop from crash ──


async def test_subprocess_alive_false_before_spawn(tmp_path: Path) -> None:
    """No subprocess + no crash future yet → not alive (listener drops
    before spawn can't be transient)."""
    d = _daemon(["+15550000000"], tmp_path)
    assert d.subprocess_alive() is False


async def test_subprocess_alive_true_while_running(tmp_path: Path) -> None:
    """A live subprocess (returncode None) with an unresolved crash
    future is alive → a listener drop here is transient, reconnect."""

    class _FakeProc:
        returncode: int | None = None

    d = _daemon(["+15550000000"], tmp_path)
    d._proc = _FakeProc()  # type: ignore[assignment]
    d._crash_future = asyncio.get_running_loop().create_future()
    assert d.subprocess_alive() is True


async def test_subprocess_alive_false_after_exit(tmp_path: Path) -> None:
    """Once the subprocess exits and ``_watch_exit`` sets the crash
    future, the daemon is no longer alive → a listener drop is fatal."""

    class _FakeProc:
        returncode: int | None = 1

    d = _daemon(["+15550000000"], tmp_path)
    d._proc = _FakeProc()  # type: ignore[assignment]
    fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    fut.set_exception(DaemonCrashError("signal-cli exited with code 1"))
    d._crash_future = fut
    assert d.subprocess_alive() is False
    # Consume the exception so the test doesn't emit a "never retrieved" warning.
    assert fut.exception() is not None


# ── daemon stdout observability ────────────────────────────────────────


async def _drain_lines(*lines: bytes) -> list[MutableMapping[str, Any]]:
    reader = asyncio.StreamReader()
    for line in lines:
        reader.feed_data(line)
    reader.feed_eof()

    with capture_logs() as logs:
        await daemon_module._drain(reader, "signal.daemon.stdout")
    return logs


async def test_daemon_stdout_exception_logs_warning_with_fields() -> None:
    daemon_module.daemon_exception_count = 0
    line = {
        "error": {
            "code": -32603,
            "message": 'Unexpected error: Cannot invoke "org.whispersystems.signalservice.api.push.ServiceId.toString()" because the return value of "org.asamk.signal.manager.api.MessageEnvelope.getServerGuid()" is null',
        },
        "exception": {
            "type": "java.lang.NullPointerException",
            "message": 'Cannot invoke "org.whispersystems.signalservice.api.push.ServiceId.toString()" because the return value of "org.asamk.signal.manager.api.MessageEnvelope.getServerGuid()" is null',
        },
    }

    logs = await _drain_lines(json.dumps(line).encode() + b"\n")

    assert any(e["event"] == "signal.daemon.stdout" and e["log_level"] == "info" for e in logs)
    warnings = [e for e in logs if e["event"] == "signal.daemon.exception"]
    assert warnings == [
        {
            "event": "signal.daemon.exception",
            "log_level": "warning",
            "exception_type": "java.lang.NullPointerException",
            "exception_message": 'Cannot invoke "org.whispersystems.signalservice.api.push.ServiceId.toString()" because the return value of "org.asamk.signal.manager.api.MessageEnvelope.getServerGuid()" is null',
            "count": 1,
        }
    ]
    assert daemon_module.daemon_exception_count == 1


async def test_daemon_stdout_normal_json_line_does_not_log_warning() -> None:
    daemon_module.daemon_exception_count = 0
    logs = await _drain_lines(b'{"jsonrpc":"2.0","method":"receive","params":{}}\n')

    assert any(e["event"] == "signal.daemon.stdout" and e["log_level"] == "info" for e in logs)
    assert [e for e in logs if e["event"] == "signal.daemon.exception"] == []
    assert daemon_module.daemon_exception_count == 0
