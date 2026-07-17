from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

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
