from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import asyncpg
import pytest

from aios.harness.production_watchdogs import (
    HeldConnectionWatchdog,
    ThroughputDeadMan,
)


class _Holder:
    def __init__(self, connection: object) -> None:
        self._in_use = object()
        self._con = connection


class _Pool:
    def __init__(self, holder: _Holder) -> None:
        self._holders = [holder]

    def get_size(self) -> int:
        return 1

    def get_idle_size(self) -> int:
        return 0


@pytest.mark.asyncio
async def test_held_connection_watchdog_captures_parked_holder(tmp_path: Path) -> None:
    parked = asyncio.Event()
    connection = object()

    async def holder() -> None:
        conn = connection
        await parked.wait()
        assert conn

    task = asyncio.create_task(holder(), name="session:test-session")
    await asyncio.sleep(0)
    inspector = AsyncMock(return_value=[{"pid": 42, "state": "idle"}])
    journal = AsyncMock(return_value=[{"seq": 7, "kind": "span"}])
    alarm = MagicMock()
    watchdog = HeldConnectionWatchdog(
        _Pool(_Holder(connection)),
        threshold_seconds=0,
        rate_limit_seconds=60,
        specimen_dir=tmp_path,
        inspect_pg=inspector,
        load_journal=journal,
        alarm=alarm,
    )

    specimens = await watchdog.check_once()

    assert len(specimens) == 1
    assert specimens[0]["owner_id"] == "test-session"
    assert "await parked.wait()" in "\n".join(specimens[0]["task_stack"])
    assert specimens[0]["journal_events"] == [{"seq": 7, "kind": "span"}]
    path = next(
        iter(await asyncio.to_thread(lambda: list(tmp_path.glob("held-connection-*.json"))))
    )
    written = json.loads(await asyncio.to_thread(path.read_text))
    assert written["pg_stat_activity"] == [{"pid": 42, "state": "idle"}]
    alarm.assert_called_once()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_dead_man_alarms_only_when_claimed_work_stalls() -> None:
    alarm = MagicMock()
    monitor = ThroughputDeadMan(threshold_seconds=600, rate_limit_seconds=60, alarm=alarm)

    assert not monitor.observe(claimed=0, completed=0, now=0)
    assert not monitor.observe(claimed=1, completed=0, now=1)
    assert monitor.observe(claimed=1, completed=0, now=602)
    assert not monitor.observe(claimed=1, completed=0, now=603)
    assert not monitor.observe(claimed=1, completed=1, now=604)
    assert alarm.call_count == 1


@pytest.mark.asyncio
async def test_held_connection_watchdog_matches_asyncpg_proxy(tmp_path: Path) -> None:
    """A borrower frame holds a proxy while PoolConnectionHolder holds raw con."""
    parked = asyncio.Event()
    raw = object()
    holder_obj = _Holder(raw)

    # Use asyncpg's production proxy type. Bypass its constructor only because
    # the raw connection is deliberately inert; preserve its real slots/layout.
    proxy = asyncpg.pool.PoolConnectionProxy.__new__(asyncpg.pool.PoolConnectionProxy)
    proxy._con = raw
    proxy._holder = holder_obj

    async def borrower() -> None:
        session_id = "proxy-session"
        conn = proxy
        await parked.wait()
        assert conn and session_id

    task = asyncio.create_task(borrower())
    await asyncio.sleep(0)
    watchdog = HeldConnectionWatchdog(
        _Pool(holder_obj),
        threshold_seconds=0,
        rate_limit_seconds=60,
        specimen_dir=tmp_path,
        inspect_pg=AsyncMock(return_value=[]),
        load_journal=AsyncMock(return_value=[]),
        alarm=MagicMock(),
    )
    specimen = (await watchdog.check_once())[0]
    assert specimen["owner_id"] == "proxy-session"
    assert "await parked.wait()" in "\n".join(specimen["task_stack"])
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_runner_reconnects_after_failed_capture_tick(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Telemetry failure is isolated: a later tick reconnects and runs."""
    import aios.harness.production_watchdogs as module

    class Inspector:
        def __init__(self, *, fail: bool) -> None:
            self.fail = fail
            self.closed = False
            self.dead_man_seen = asyncio.Event()

        def is_closed(self) -> bool:
            return self.closed

        async def close(self) -> None:
            self.closed = True

        async def fetch(
            self, query: str, *args: object, **kwargs: object
        ) -> list[dict[str, object]]:
            if self.fail:
                raise OSError("telemetry database disconnected")
            return []

        async def fetchrow(self, query: str, *args: object, **kwargs: object) -> dict[str, int]:
            self.dead_man_seen.set()
            return {"claimed": 0, "completed": 0}

    first, second = Inspector(fail=True), Inspector(fail=False)
    connect = AsyncMock(side_effect=[first, second])
    monkeypatch.setattr(asyncpg, "connect", connect)
    raw = object()
    runner = asyncio.create_task(
        module.run_production_watchdogs(
            _Pool(_Holder(raw)),
            "postgresql://example/db",
            held_threshold_seconds=0,
            dead_man_threshold_seconds=10,
            interval_seconds=0.001,
            rate_limit_seconds=10,
            specimen_dir=tmp_path,
            journal_limit=10,
            operation_timeout_seconds=0.1,
        )
    )
    await asyncio.wait_for(second.dead_man_seen.wait(), 1)
    assert connect.await_count == 2
    assert first.closed
    assert not runner.done()
    runner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await runner


@pytest.mark.asyncio
async def test_runner_counts_session_and_workflow_wake_completions_symmetrically(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Healthy workflow wakes count as throughput even when the run parks."""
    import aios.harness.production_watchdogs as module

    class Inspector:
        def __init__(self) -> None:
            self.closed = False
            self.queries: list[str] = []
            self.observed = asyncio.Event()

        def is_closed(self) -> bool:
            return self.closed

        async def close(self) -> None:
            self.closed = True

        async def fetch(self, query: str, *args: object, **kwargs: object) -> list[object]:
            return []

        async def fetchrow(self, query: str, *args: object, **kwargs: object) -> dict[str, int]:
            self.queries.append(query)
            self.observed.set()
            return {"claimed": 1, "completed": 1}

    inspector = Inspector()
    monkeypatch.setattr(asyncpg, "connect", AsyncMock(return_value=inspector))
    runner = asyncio.create_task(
        module.run_production_watchdogs(
            _Pool(_Holder(object())),
            "postgresql://example/db",
            held_threshold_seconds=999,
            dead_man_threshold_seconds=0.001,
            interval_seconds=0.001,
            rate_limit_seconds=10,
            specimen_dir=tmp_path,
            journal_limit=10,
            operation_timeout_seconds=0.1,
        )
    )
    await asyncio.wait_for(inspector.observed.wait(), 1)
    query = inspector.queries[-1]
    assert "procrastinate_events" in query
    assert "harness.wake_session" in query
    assert "harness.wake_workflow" in query
    assert "run_completed" not in query
    assert not runner.done()
    runner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await runner
