"""Shared test fixtures for aios-whatsapp."""

from __future__ import annotations

import os
import socket
import sys
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# HttpConnector reads AIOS_URL / AIOS_RUNTIME_TOKEN at __init__ time.
os.environ.setdefault("AIOS_URL", "http://test")
os.environ.setdefault("AIOS_RUNTIME_TOKEN", "aios_runtime_test")

from aios_whatsapp.config import Settings
from aios_whatsapp.connector import WhatsappConnector, _WhatsappConnectionState
from aios_whatsapp.daemon import WhatsappDaemon

CONNECTION_ID = "conn_test"
PHONE = "+15551112222"
BOT_JID = "15551112222@s.whatsapp.net"
PEER_JID = "15553334444@s.whatsapp.net"
GROUP_JID = "111222333@g.us"


def _unused_port() -> int:
    """Bind to an OS-assigned port and immediately release it.

    There's a (negligible) race window between us releasing the port and
    the test using it — small enough to ignore for unit tests, and the
    daemon spawn would fail loudly if another process grabbed it.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
    return port


@pytest.fixture
def unused_port() -> int:
    return _unused_port()


@pytest.fixture
def connector(tmp_path: Path) -> WhatsappConnector:
    """WhatsappConnector with a stubbed daemon and one registered connection.

    Tool-method tests invoke ``connector.whatsapp_send(...)`` directly
    (bypassing the SDK's focal-channel injection); the daemon's
    ``rpc.call`` is a mock that captures args and returns whatever the
    test sets on it.
    """
    cfg = Settings(data_dir=tmp_path / "data")
    c = WhatsappConnector(cfg)
    fake_daemon = MagicMock(spec=WhatsappDaemon)
    fake_daemon.rpc = MagicMock()
    fake_daemon.rpc.call = AsyncMock(return_value=None)
    c.state[CONNECTION_ID] = _WhatsappConnectionState(phone=PHONE, daemon=fake_daemon)
    c.emit_inbound = AsyncMock(return_value={"appended_event_id": "ev_1"})  # type: ignore[method-assign]
    return c


@pytest.fixture
def fake_daemon_bin(tmp_path: Path) -> Iterator[Path]:
    """A path that behaves like the real ``whatsapp-daemon`` for spawn tests.

    The real daemon is a Go binary; for Python unit tests we wrap a
    Python stand-in (``tests/fake_daemon.py``) in a tiny shell script
    that invokes the current interpreter.  This avoids both a Go
    toolchain dependency and any reliance on ``python3``-on-PATH being
    the same interpreter the test suite is running under.
    """
    script = Path(__file__).parent / "fake_daemon.py"
    wrapper = tmp_path / "whatsapp-daemon"
    # ``exec`` so the wrapper doesn't sit between our subprocess and
    # the Python interpreter when SIGTERM arrives — SIGTERM should hit
    # python directly so its KeyboardInterrupt path runs.
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
    wrapper.chmod(0o755)
    yield wrapper
