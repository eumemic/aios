"""End-to-end tests for daemon.py against a Python stand-in binary.

Unit tests for daemon.py have to exercise the actual subprocess spawn
to be meaningful — port allocation, TCP readiness polling, stdio drain,
and crash-fatal future propagation all depend on the OS-level process
boundary.  Signal's test_daemon.py skips this because signal-cli would
need a Java toolchain in CI; we mock signal-cli's place with a tiny
Python script (``tests/fake_daemon.py``) wrapped by a shell wrapper
fixture (``fake_daemon_bin``).
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

import pytest

from aios_whatsapp import daemon as daemon_mod
from aios_whatsapp.daemon import WhatsappDaemon
from aios_whatsapp.errors import DaemonCrashError


async def test_daemon_spawn_and_version(
    fake_daemon_bin: Path, unused_port: int, tmp_path: Path
) -> None:
    """Daemon spawns, becomes ready (TCP open + ``version`` answers), and
    round-trips a follow-up ``version`` call.
    """
    daemon = WhatsappDaemon(
        daemon_bin=str(fake_daemon_bin),
        host="127.0.0.1",
        port=unused_port,
        store_dir=tmp_path / "store",
    )
    async with daemon:
        result = await daemon.version()
        assert result == {"name": "whatsapp-daemon-fake", "version": "test"}


async def test_daemon_creates_store_dir(
    fake_daemon_bin: Path, unused_port: int, tmp_path: Path
) -> None:
    """``serve_connection`` may be the first thing to ever touch the
    per-phone dir; daemon spawn should create it rather than fail on a
    missing parent.
    """
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


async def test_daemon_crash_sets_crashed_future(
    fake_daemon_bin: Path, unused_port: int, tmp_path: Path
) -> None:
    """A subprocess exit while the daemon is running flips the
    ``crashed()`` future to a :class:`DaemonCrashError`.  The connector
    awaits this alongside the listener stream so a daemon death tears
    the per-connection serve task down promptly.
    """
    daemon = WhatsappDaemon(
        daemon_bin=str(fake_daemon_bin),
        host="127.0.0.1",
        port=unused_port,
        store_dir=tmp_path / "store",
    )
    async with daemon:
        # The fake daemon's ``crash`` method sys.exits with code 1 after
        # a short delay.  Fire-and-forget the call: it closes the socket
        # without responding, then exits; suppress the resulting RPC
        # error so the exit-watcher gets a chance to fire.
        with contextlib.suppress(Exception):
            await daemon.rpc.call("crash")
        with pytest.raises(DaemonCrashError):
            await asyncio.wait_for(daemon.crashed(), timeout=5.0)


async def test_daemon_readiness_times_out_when_no_listener(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the spawned process never opens a listener (wrong binary,
    immediate-crash, etc.), ``__aenter__`` exhausts the readiness
    attempts and raises :class:`DaemonCrashError`.
    """
    # A wrapper that just sleeps — accepts (and ignores) our argv but
    # never binds the port.  Models the "binary spawned but never
    # listened" failure mode without depending on the OS shell's flag
    # handling.
    sleeper = tmp_path / "sleeper"
    sleeper.write_text("#!/bin/sh\nsleep 5\n")
    sleeper.chmod(0o755)

    # Shrink the readiness window so the test doesn't sit through the
    # production default (~30 s).
    monkeypatch.setattr(daemon_mod, "READY_POLL_ATTEMPTS", 3)
    monkeypatch.setattr(daemon_mod, "READY_POLL_INTERVAL_S", 0.05)

    daemon = WhatsappDaemon(
        daemon_bin=str(sleeper),
        host="127.0.0.1",
        port=_unused_port_sync(),
        store_dir=tmp_path / "store",
    )
    with pytest.raises(DaemonCrashError):
        await daemon.__aenter__()


def _unused_port_sync() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
    return port
