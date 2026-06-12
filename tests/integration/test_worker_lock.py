"""Integration tests for ``_acquire_worker_lock``.

The lock-wait path matters during rolling deploys: when Coolify (or any
other orchestrator) starts the new worker container before the old one
has fully shut down, the new one must wait briefly for the old one's
advisory lock to release rather than exiting immediately. The tests
exercise the three paths through ``_acquire_worker_lock``:

1. lock free → returns a held connection
2. lock held briefly → waits, then returns a held connection
3. lock held longer than timeout → returns ``None`` after deadline
"""

from __future__ import annotations

import asyncio
from typing import Any

import asyncpg
import pytest

from aios.db.pool import LISTENER_TCP_KEEPALIVE_SETTINGS
from aios.harness.worker import (
    _acquire_worker_lock,
    _make_advisory_lock_termination_listener,
    _SupervisedTaskFailure,
)

pytestmark = pytest.mark.integration


class _FakeLog:
    """Captures structured-log calls (info/error with kwargs)."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, Any]]] = []

    def info(self, event: str, **kwargs: Any) -> None:
        self.events.append(("info", event, kwargs))

    def error(self, event: str, **kwargs: Any) -> None:
        self.events.append(("error", event, kwargs))


@pytest.fixture
def log() -> _FakeLog:
    return _FakeLog()


class TestAcquireWorkerLock:
    async def test_acquires_when_free(self, db_url: str, log: _FakeLog) -> None:
        conn = await _acquire_worker_lock(db_url, log, timeout_seconds=2.0)
        assert conn is not None
        observer = await asyncpg.connect(db_url)
        try:
            held_by_other = await observer.fetchval(
                "SELECT pg_try_advisory_lock(hashtextextended('aios_worker_connector_supervisor', 0))"
            )
            assert held_by_other is False, "lock should be unavailable to a second connection"
        finally:
            await observer.close()
            await conn.close()

    async def test_waits_then_acquires_after_release(self, db_url: str, log: _FakeLog) -> None:
        holder = await asyncpg.connect(db_url)
        await holder.fetchval(
            "SELECT pg_advisory_lock(hashtextextended('aios_worker_connector_supervisor', 0))"
        )

        async def release_after_delay() -> None:
            await asyncio.sleep(1.0)
            await holder.close()

        release_task = asyncio.create_task(release_after_delay())
        conn = await _acquire_worker_lock(
            db_url, log, timeout_seconds=10.0, poll_interval_seconds=0.1
        )
        await release_task
        assert conn is not None
        await conn.close()

    async def test_returns_none_after_timeout(self, db_url: str, log: _FakeLog) -> None:
        holder = await asyncpg.connect(db_url)
        try:
            await holder.fetchval(
                "SELECT pg_advisory_lock(hashtextextended('aios_worker_connector_supervisor', 0))"
            )
            conn = await _acquire_worker_lock(
                db_url, log, timeout_seconds=1.0, poll_interval_seconds=0.1
            )
            assert conn is None
            kinds = {e[0] for e in log.events}
            event_names = {e[1] for e in log.events}
            assert kinds == {"info", "error"}
            assert "worker.lock_busy.waiting" in event_names
            assert "worker.duplicate_instance_refused" in event_names
        finally:
            await holder.close()

    async def test_lock_connection_uses_tcp_keepalive_settings(
        self, db_url: str, log: _FakeLog
    ) -> None:
        conn = await _acquire_worker_lock(db_url, log, timeout_seconds=2.0)
        assert conn is not None
        try:
            for key, expected in LISTENER_TCP_KEEPALIVE_SETTINGS.items():
                assert await conn.fetchval(f"SHOW {key}") == expected
        finally:
            await conn.close()

    async def test_backend_termination_trips_lock_listener(
        self, db_url: str, log: _FakeLog
    ) -> None:
        conn = await _acquire_worker_lock(db_url, log, timeout_seconds=2.0)
        assert conn is not None
        latch = asyncio.Event()
        fatal: _SupervisedTaskFailure = {"exception": None}
        conn.add_termination_listener(
            _make_advisory_lock_termination_listener(latch=latch, fatal=fatal)
        )
        terminator = await asyncpg.connect(db_url)
        try:
            pid = conn.get_server_pid()
            terminated = await terminator.fetchval("SELECT pg_terminate_backend($1)", pid)
            assert terminated is True
            await asyncio.wait_for(latch.wait(), timeout=5.0)
            assert isinstance(fatal["exception"], RuntimeError)
        finally:
            await terminator.close()
            if not conn.is_closed():
                await conn.close()
