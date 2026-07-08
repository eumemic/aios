"""Integration coverage for worker interrupt-listener reconnect."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any
from unittest.mock import MagicMock

import asyncpg
import pytest

from aios.db.listen import SESSION_INTERRUPT_CHANNEL
from aios.db.pool import create_pool, listener_application_name
from aios.harness.worker import _run_interrupt_listener
from tests.conftest import needs_docker

pytestmark = [pytest.mark.integration, needs_docker]


class _FakeInflightToolRegistry:
    def __init__(self) -> None:
        self.cancelled: asyncio.Queue[str] = asyncio.Queue()

    def cancel_step(self, session_id: str) -> bool:
        self.cancelled.put_nowait(session_id)
        return True

    def cancel_session(self, session_id: str) -> int:
        return 0


async def _listener_backend_pid(db_url: str) -> int:
    app_name = listener_application_name()
    observer = await asyncpg.connect(db_url)
    try:
        deadline = asyncio.get_running_loop().time() + 5.0
        while True:
            pid = await observer.fetchval(
                "SELECT pid FROM pg_stat_activity "
                "WHERE datname = current_database() "
                "AND application_name = $1 "
                "AND pid <> pg_backend_pid() "
                "ORDER BY backend_start DESC LIMIT 1",
                app_name,
            )
            if pid is not None:
                return int(pid)
            if asyncio.get_running_loop().time() > deadline:
                raise AssertionError("interrupt listener backend did not appear")
            await asyncio.sleep(0.1)
    finally:
        await observer.close()


async def _notify_interrupt(db_url: str, session_id: str) -> None:
    conn = await asyncpg.connect(db_url)
    try:
        await conn.execute("SELECT pg_notify($1, $2)", SESSION_INTERRUPT_CHANNEL, session_id)
    finally:
        await conn.close()


async def test_interrupt_listener_reconnects_after_backend_termination(
    migrated_db_url: str,
    _reset_db_state: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("aios.harness.worker._LISTEN_RECONNECT_BACKOFF_SECONDS", 0.1)
    fake_registry = _FakeInflightToolRegistry()
    registry = MagicMock()
    registry.cancel_step.side_effect = fake_registry.cancel_step
    registry.cancel_session.side_effect = fake_registry.cancel_session
    # No locally-tracked sessions in this test, so the #1756 reconnect
    # redrive (``_redrive_interrupts_for_tracked_sessions``) is a no-op —
    # this test exercises the LIVE NOTIFY path post-reconnect, not the redrive.
    registry.tracked_session_ids.return_value = set()
    pool = await create_pool(migrated_db_url, min_size=1, max_size=2)
    listener_task = asyncio.create_task(
        _run_interrupt_listener(migrated_db_url, registry, pool),
        name="test-interrupt-listener",
    )

    killer: asyncpg.Connection[Any] | None = None
    try:
        first_pid = await _listener_backend_pid(migrated_db_url)
        killer = await asyncpg.connect(migrated_db_url)
        terminated = await killer.fetchval("SELECT pg_terminate_backend($1)", first_pid)
        assert terminated is True

        deadline = asyncio.get_running_loop().time() + 5.0
        while True:
            new_pid = await _listener_backend_pid(migrated_db_url)
            if new_pid != first_pid:
                break
            if asyncio.get_running_loop().time() > deadline:
                raise AssertionError(
                    "interrupt listener did not reconnect after backend termination"
                )
            await asyncio.sleep(0.1)

        await _notify_interrupt(migrated_db_url, "sess_after_reconnect")
        assert (
            await asyncio.wait_for(fake_registry.cancelled.get(), timeout=5.0)
            == "sess_after_reconnect"
        )
    finally:
        if killer is not None:
            await killer.close()
        listener_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await listener_task
        await pool.close()
