"""Unit tests for ``_periodic_heartbeat``.

The heartbeat task is the bridge between worker liveness and the
container HEALTHCHECK that reads file mtime. The tests cover three
behaviors:

1. The task touches the heartbeat file on its first iteration.
2. The task continues to refresh the mtime on subsequent iterations
   (so a healthy worker keeps the file fresh).
3. An ``OSError`` from a single ``touch()`` is caught and logged, not
   propagated — a transient filesystem glitch shouldn't take down the
   worker.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from aios.harness import worker as worker_mod


class TestPeriodicHeartbeat:
    async def test_touches_file_on_first_iteration(self, tmp_path: Path) -> None:
        target = tmp_path / "alive"
        with patch.object(worker_mod, "_HEARTBEAT_FILE", target):
            task = asyncio.create_task(worker_mod._periodic_heartbeat(interval=0))
            try:
                await asyncio.sleep(0.05)
                assert target.exists()
            finally:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

    async def test_refreshes_mtime_on_subsequent_iterations(self, tmp_path: Path) -> None:
        target = tmp_path / "alive"
        target.touch()
        target_st = target.stat()
        # Start mtime in the past so any refresh is detectable.
        import os as _os

        _os.utime(target, (target_st.st_atime - 60, target_st.st_mtime - 60))
        before = target.stat().st_mtime

        with patch.object(worker_mod, "_HEARTBEAT_FILE", target):
            task = asyncio.create_task(worker_mod._periodic_heartbeat(interval=0))
            try:
                await asyncio.sleep(0.05)
                assert target.stat().st_mtime > before
            finally:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

    async def test_oserror_does_not_kill_task(self, tmp_path: Path) -> None:
        target = tmp_path / "alive"

        call_count = 0
        original_touch = Path.touch

        def flaky_touch(self: Path, *args: object, **kwargs: object) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise PermissionError("simulated tmpfs glitch")
            original_touch(self)

        with (
            patch.object(worker_mod, "_HEARTBEAT_FILE", target),
            patch.object(Path, "touch", flaky_touch),
        ):
            task = asyncio.create_task(worker_mod._periodic_heartbeat(interval=0))
            try:
                await asyncio.sleep(0.05)
                # Task is still running (didn't propagate the exception)
                assert not task.done(), "heartbeat task should survive a transient touch failure"
                # And subsequent calls succeeded
                assert call_count >= 2
                assert target.exists()
            finally:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
