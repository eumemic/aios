"""Shared test fixtures for aios-whatsapp."""

from __future__ import annotations

import os
import socket
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

# HttpConnector reads AIOS_URL / AIOS_RUNTIME_TOKEN at __init__ time.
os.environ.setdefault("AIOS_URL", "http://test")
os.environ.setdefault("AIOS_RUNTIME_TOKEN", "aios_runtime_test")


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
