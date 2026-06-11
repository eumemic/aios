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


class TestResolveHeartbeatPath:
    """``_resolve_heartbeat_path`` must keep the container HEALTHCHECK
    contract while never handing back a path the host user can't write.

    In a container (``/.dockerenv`` present) the Dockerfile HEALTHCHECK
    stats ``/var/run/aios-worker-alive``, so that exact path must be
    returned. On the host the path must fall back to a user-writable
    temp directory so the periodic touch doesn't EACCES-spam the log
    every 15 s and so anything consuming the file actually finds it.
    """

    def test_uses_container_path_when_in_container(self) -> None:
        with patch.object(worker_mod, "is_running_in_container", return_value=True):
            assert worker_mod._resolve_heartbeat_path() == Path("/var/run/aios-worker-alive")

    def test_falls_back_to_tmpdir_on_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import tempfile

        monkeypatch.delenv("TMPDIR", raising=False)
        with patch.object(worker_mod, "is_running_in_container", return_value=False):
            resolved = worker_mod._resolve_heartbeat_path()

        # Never the unwritable container path on the host.
        assert resolved != Path("/var/run/aios-worker-alive")
        assert resolved == Path(tempfile.gettempdir()) / "aios-worker-alive"

    def test_host_path_honors_tmpdir_env(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("TMPDIR", str(tmp_path))
        with patch.object(worker_mod, "is_running_in_container", return_value=False):
            resolved = worker_mod._resolve_heartbeat_path()

        assert resolved == tmp_path / "aios-worker-alive"

    def test_host_path_is_writable(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("TMPDIR", str(tmp_path))
        with patch.object(worker_mod, "is_running_in_container", return_value=False):
            resolved = worker_mod._resolve_heartbeat_path()

        # The whole point of the fallback: the host user can actually
        # create the file without EACCES.
        resolved.touch()
        assert resolved.exists()


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
