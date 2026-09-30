"""Reproduction + guards for heartbeat succession across a process restart.

Property (PR #2355 finding F1): After a process restart, the Docker
healthcheck must not report healthy until *this* process's own transports are
ready. A heartbeat written by an earlier process must never count as the
current process's health, even while it is still fresh.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path

import pytest
from aios_connector_http.healthcheck import main
from aios_connector_http.runner import HttpConnector, _ConnectionState


class _Connector(HttpConnector):
    connector = "probe"


def _mk(connection_id: str, status: str) -> _ConnectionState:
    state = _ConnectionState(connection_id=connection_id, external_account_id="acct")
    state.serve_status = status  # type: ignore[assignment]
    return state


async def _run_one_iteration(connector: HttpConnector, path: Path) -> None:
    done = asyncio.Event()

    async def _hook() -> None:
        done.set()
        # Park forever after the first publish so the loop does not sleep/rerun.
        await asyncio.Event().wait()

    connector._heartbeat_iteration_hook = _hook
    task = asyncio.create_task(connector._heartbeat_loop(path))
    try:
        await asyncio.wait_for(done.wait(), timeout=5)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


def test_successor_does_not_inherit_predecessor_fresh_health(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Predecessor leaves a fresh all-healthy heartbeat; successor still starting.

    The successor process shares the container heartbeat path. Its own
    transports are still ``starting`` (nothing serving yet). After one heartbeat
    iteration the container healthcheck must NOT report healthy, because a fresh
    heartbeat authored by the *predecessor* must not stand in for the
    successor's own readiness.
    """
    path = tmp_path / "alive"
    monkeypatch.setenv("AIOS_CONNECTOR_HEARTBEAT_PATH", str(path))

    async def scenario() -> None:
        # --- Predecessor: publishes a fresh "conn_1 serving" heartbeat, then
        # shuts down the way run()'s finally block does (clears in-memory
        # ownership but LEAVES the fresh file in place).
        predecessor = _Connector(base_url="http://x", token="aios_runtime_x")
        predecessor._connections = {"conn_1": _mk("conn_1", "serving")}
        predecessor._discovery_cursor = 0
        await _run_one_iteration(predecessor, path)
        assert predecessor._heartbeat_owned is True
        # Sanity: the predecessor's published content is all-healthy and fresh.
        published = json.loads(path.read_text())
        assert published["healthy_connection_ids"] == ["conn_1"]
        # run()'s shutdown finally relinquishes ownership. On the buggy head this
        # leaves the file fresh; the fix back-dates the owned inode to stale.
        await predecessor._remove_owned_heartbeat(path)

        # --- Successor: fresh process, same path. Discovery has run
        # (cursor set) but conn_1's transport is still ``starting``.
        successor = _Connector(base_url="http://x", token="aios_runtime_x")
        successor._connections = {"conn_1": _mk("conn_1", "starting")}
        successor._discovery_cursor = 0
        await _run_one_iteration(successor, path)

        # The container probe must fail closed: nothing the successor serves is
        # ready, so it must not report healthy off the predecessor's file.
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1, (
            "successor reported healthy off the predecessor's fresh heartbeat"
        )

    asyncio.run(scenario())


def test_successor_reports_healthy_once_its_own_transport_serves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Over-correction guard: the successor MUST become healthy once ready.

    The degenerate form of the F1 fix -- 'never trust an existing file' or
    'never publish healthy after a restart' -- would keep the container
    permanently unhealthy. Assert the legitimate case: once the successor's own
    transport is serving, its heartbeat reports healthy again.
    """
    path = tmp_path / "alive"
    monkeypatch.setenv("AIOS_CONNECTOR_HEARTBEAT_PATH", str(path))

    async def scenario() -> None:
        # Predecessor publishes and shuts down, leaving a fresh file.
        predecessor = _Connector(base_url="http://x", token="aios_runtime_x")
        predecessor._connections = {"conn_1": _mk("conn_1", "serving")}
        predecessor._discovery_cursor = 0
        await _run_one_iteration(predecessor, path)
        await predecessor._remove_owned_heartbeat(path)

        # Successor starts, transport still starting -> unhealthy.
        successor = _Connector(base_url="http://x", token="aios_runtime_x")
        successor._connections = {"conn_1": _mk("conn_1", "starting")}
        successor._discovery_cursor = 0
        await _run_one_iteration(successor, path)
        with pytest.raises(SystemExit):
            main()

        # Successor's own transport comes up.
        successor._connections["conn_1"].serve_status = "serving"
        await _run_one_iteration(successor, path)

        # Now the probe must succeed (no SystemExit).
        main()
        published = json.loads(path.read_text())
        assert published["healthy_connection_ids"] == ["conn_1"]

    asyncio.run(scenario())
