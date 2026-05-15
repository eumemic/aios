"""The double-timeout give-up path must close the subprocess transport.

When the post-SIGKILL drain in ``run_subprocess_with_timeout`` also
times out (a child wedged in uninterruptible D-state — ``docker``
against a flapping daemon, ``git`` on a dead mount), the give-up branch
must close ``proc._transport`` so the cancelled ``communicate()``
doesn't strand the parent-side stdout/stderr pipe FDs. Every container
provision and host git-clone funnels through this runner; pre-fix,
repeated wedges exhaust the worker's FD ceiling and starve both
connection pools. The mechanism is documented at the fix site in
``_subprocess.py``.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from aios.sandbox import _subprocess
from aios.sandbox._subprocess import run_subprocess_with_timeout


class _WedgedProc:
    """A child stuck in D-state: ``communicate()`` never completes and
    the pending SIGKILL from ``kill()`` never takes effect."""

    def __init__(self) -> None:
        self.returncode: int | None = None
        # The sole non-blocking handle to the parent-side pipe FDs.
        # asyncio.subprocess.Process.kill() also delegates to the
        # transport, so route kill through the same mock — the fake
        # can't drift from Process's real kill/_transport coupling.
        self._transport = MagicMock(name="transport")

    def kill(self) -> None:
        self._transport.kill()

    async def communicate(self) -> tuple[bytes, bytes]:
        await asyncio.sleep(3600)  # outlives both (tiny) timeouts
        raise AssertionError("unreachable: communicate must be cancelled")


async def test_give_up_path_closes_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    """Outer drain times out → SIGKILL → inner drain *also* times out.

    The give-up branch must close ``proc._transport`` so the parent-side
    stdout/stderr pipe FDs are released; otherwise they leak for as long
    as the wedged child lives (effectively forever under a flapping
    daemon)."""
    proc = _WedgedProc()

    async def _fake_spawn(*_argv: Any, **_kwargs: Any) -> _WedgedProc:
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_spawn)
    # Keep the secondary drain tiny so the test is fast and deterministic.
    monkeypatch.setattr(_subprocess, "_DRAIN_AFTER_KILL_TIMEOUT_S", 0.01)

    rc, out, err, timed_out = await run_subprocess_with_timeout(["docker", "ps"], timeout_s=0.01)

    # The (-1, b"", b"", True) tuple is only reachable after proc.kill()
    # on the SIGKILL path, so this also pins that we took the give-up
    # branch rather than the happy or drain-succeeded paths.
    assert (rc, out, err, timed_out) == (-1, b"", b"", True)

    proc._transport.close.assert_called_once()
