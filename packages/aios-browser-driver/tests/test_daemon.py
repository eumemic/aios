"""Daemon startup must fail loudly, not silently.

If ``serve()`` dies before it signals readiness (e.g. a ``PermissionError``
binding ``/run/aios`` as uid 1000), the process must CRASH with that error —
never hang forever on ``ready.wait()`` as a live-but-never-serving container.
"""

from __future__ import annotations

import asyncio

import pytest
from aios_browser_driver import daemon
from aios_browser_driver.server import DispatchFn


async def test_run_reraises_a_startup_failure_instead_of_hanging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _failing_serve(dispatch: DispatchFn, *, ready: asyncio.Event | None = None) -> None:
        raise PermissionError("cannot bind driver socket")

    monkeypatch.setattr(daemon, "serve", _failing_serve)

    # Without surfacing the startup crash, _run would block on ready.wait()
    # forever and this would raise TimeoutError instead of PermissionError.
    with pytest.raises(PermissionError):
        await asyncio.wait_for(daemon._run(), timeout=5)
