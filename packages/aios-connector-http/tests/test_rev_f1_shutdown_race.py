"""F1-regress: shutdown while a heartbeat publication thread is in flight.

Cancelling the heartbeat task cancels the awaiting coroutine but not the
``asyncio.to_thread`` worker. That worker could still RENAME_EXCHANGE a fresh,
all-healthy inode into the path AFTER shutdown stale-dated the old identity,
so a successor would inherit the exiting process's fresh snapshot.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from aios_connector_http.healthcheck import main
from aios_connector_http.runner import HttpConnector, _ConnectionState


class _C(HttpConnector):
    connector = "probe"


def _slow_second_fsync(monkeypatch: pytest.MonkeyPatch) -> threading.Event:
    in_fsync = threading.Event()
    real_fsync = os.fsync
    calls = {"n": 0}

    def slow_fsync(fd: int) -> None:
        calls["n"] += 1
        if calls["n"] == 2:  # 1st = claim, 2nd = first refresh
            in_fsync.set()
            time.sleep(0.5)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", slow_fsync)
    return in_fsync


def _connector(serve_status: Any) -> _C:
    c = _C(base_url="http://x", token="aios_runtime_x")
    c._connections = {"conn_1": _ConnectionState("conn_1", "a", serve_status=serve_status)}
    c._discovery_cursor = 0
    c.HEARTBEAT_INTERVAL = 0.0
    return c


async def _shutdown_during_inflight_refresh(path: Path, in_fsync: threading.Event) -> None:
    c = _connector("serving")
    task = asyncio.create_task(c._heartbeat_loop(path))
    await asyncio.to_thread(in_fsync.wait, 5)
    assert in_fsync.is_set()
    task.cancel()  # exactly what run()'s finally does
    with pytest.raises(asyncio.CancelledError):
        await task
    await c._remove_owned_heartbeat(path)
    # Give any orphaned publication thread time to land.
    await asyncio.sleep(1.0)


def test_shutdown_during_inflight_refresh_leaves_no_fresh_heartbeat(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "alive"
    monkeypatch.setenv("AIOS_CONNECTOR_HEARTBEAT_PATH", str(path))
    in_fsync = _slow_second_fsync(monkeypatch)
    asyncio.run(_shutdown_during_inflight_refresh(path, in_fsync))
    with pytest.raises(SystemExit):
        main()


def test_successor_after_inflight_shutdown_is_not_healthy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "alive"
    monkeypatch.setenv("AIOS_CONNECTOR_HEARTBEAT_PATH", str(path))
    in_fsync = _slow_second_fsync(monkeypatch)

    async def scenario() -> bool:
        await _shutdown_during_inflight_refresh(path, in_fsync)
        successor = _connector("starting")
        done = asyncio.Event()

        async def hook() -> None:
            done.set()
            await asyncio.Event().wait()  # park after one iteration

        successor._heartbeat_iteration_hook = hook
        task = asyncio.create_task(successor._heartbeat_loop(path))
        await asyncio.wait_for(done.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return successor._heartbeat_owned

    owned = asyncio.run(scenario())
    # The predecessor's inode was stale, so the successor reclaims it and
    # publishes its own (all-unhealthy) attribution.
    assert owned
    assert json.loads(path.read_bytes())["healthy_connection_ids"] == []
    with pytest.raises(SystemExit):
        main()
