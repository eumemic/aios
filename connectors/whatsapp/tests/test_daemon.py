"""Lifecycle tests for ``WhatsappDaemon`` against a Python stand-in.

The real daemon is a Go binary; tests use ``tests/fake_daemon.py``
wrapped by the ``fake_daemon_bin`` conftest fixture so the subprocess
spawn / readiness-probe / crash-detection paths exercise real OS-level
process boundaries without a Go toolchain at test time.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import pytest

from aios_whatsapp import daemon as daemon_mod
from aios_whatsapp.daemon import WhatsappDaemon
from aios_whatsapp.errors import DaemonCrashError, ListenerClosedError


async def test_daemon_spawn_and_version(
    fake_daemon_bin: Path, unused_port: int, tmp_path: Path
) -> None:
    daemon = WhatsappDaemon(
        daemon_bin=str(fake_daemon_bin),
        host="127.0.0.1",
        port=unused_port,
        store_dir=tmp_path / "store",
    )
    async with daemon:
        result = await daemon.rpc.call("version")
        assert result == {"name": "whatsapp-daemon-fake", "version": "test"}


async def test_daemon_creates_store_dir(
    fake_daemon_bin: Path, unused_port: int, tmp_path: Path
) -> None:
    store_dir = tmp_path / "nested" / "store"
    assert not store_dir.exists()
    daemon = WhatsappDaemon(
        daemon_bin=str(fake_daemon_bin),
        host="127.0.0.1",
        port=unused_port,
        store_dir=store_dir,
    )
    async with daemon:
        pass
    assert store_dir.is_dir()


async def test_daemon_crash_propagates_to_listener(
    fake_daemon_bin: Path, unused_port: int, tmp_path: Path
) -> None:
    """A subprocess exit makes the listener stream raise ``ListenerClosedError``.

    This is the path the connector's ``_drain_notifications`` observes
    so the per-connection serve task tears down on daemon death.
    """
    daemon = WhatsappDaemon(
        daemon_bin=str(fake_daemon_bin),
        host="127.0.0.1",
        port=unused_port,
        store_dir=tmp_path / "store",
    )
    async with daemon:
        # ``crash`` sys.exits the fake daemon shortly after closing the
        # socket; fire-and-forget the call and suppress the resulting
        # RPC error (the connection close races the exit).
        with contextlib.suppress(Exception):
            await daemon.rpc.call("crash")
        with pytest.raises(ListenerClosedError):
            async for _ in daemon.listener.notifications():
                pass


async def test_daemon_readiness_times_out_when_no_listener(
    tmp_path: Path, unused_port: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A spawned process that never binds raises ``DaemonCrashError``."""
    # Sleeper accepts our argv but never binds — models "binary spawned
    # but never listened".
    sleeper = tmp_path / "sleeper"
    sleeper.write_text("#!/bin/sh\nsleep 5\n")
    sleeper.chmod(0o755)

    monkeypatch.setattr(daemon_mod, "READY_POLL_ATTEMPTS", 3)
    monkeypatch.setattr(daemon_mod, "READY_POLL_INTERVAL_S", 0.05)

    daemon = WhatsappDaemon(
        daemon_bin=str(sleeper),
        host="127.0.0.1",
        port=unused_port,
        store_dir=tmp_path / "store",
    )
    with pytest.raises(DaemonCrashError):
        await daemon.__aenter__()
