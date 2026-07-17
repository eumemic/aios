"""Out-of-band worker watchdogs and freeze-specimen capture."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import traceback
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import asyncpg

from aios.db.pool import LISTENER_TCP_KEEPALIVE_SETTINGS, normalize_dsn
from aios.logging import get_logger

log = get_logger("aios.worker.watchdogs")

try:
    from prometheus_client import Counter

    _ALARMS = Counter("aios_worker_watchdog_alarms_total", "Worker watchdog alarms", ["kind"])
except Exception:  # pragma: no cover
    _ALARMS = None


def _emit_alarm(kind: str, specimen: dict[str, Any]) -> None:
    if _ALARMS is not None:
        with contextlib.suppress(Exception):
            _ALARMS.labels(kind=kind).inc()
    log.error(f"worker.{kind}_alarm", alarm_event=True, **specimen)


def _task_for_connection(connection: object) -> asyncio.Task[Any] | None:
    current = asyncio.current_task()
    for task in asyncio.all_tasks():
        if task is current:
            continue
        for frame in task.get_stack():
            if any(value is connection for value in frame.f_locals.values()):
                return task
    return None


def _owner_id(task: asyncio.Task[Any] | None) -> str | None:
    if task is None:
        return None
    for frame in reversed(task.get_stack()):
        for key in ("session_id", "run_id"):
            value = frame.f_locals.get(key)
            if isinstance(value, str):
                return value
    name = task.get_name()
    return name.split(":", 1)[1] if ":" in name else None


class HeldConnectionWatchdog:
    """Observe asyncpg pool holders without borrowing from the watched pool."""

    def __init__(
        self,
        pool: Any,
        *,
        threshold_seconds: float,
        rate_limit_seconds: float,
        specimen_dir: Path,
        inspect_pg: Callable[[], Awaitable[list[dict[str, Any]]]],
        load_journal: Callable[[str | None], Awaitable[list[dict[str, Any]]]],
        alarm: Callable[[str, dict[str, Any]], None] = _emit_alarm,
    ) -> None:
        self.pool = pool
        self.threshold_seconds = threshold_seconds
        self.rate_limit_seconds = rate_limit_seconds
        self.specimen_dir = specimen_dir
        self.inspect_pg = inspect_pg
        self.load_journal = load_journal
        self.alarm = alarm
        self._first_seen: dict[int, float] = {}
        self._last_alarm: dict[int, float] = {}

    async def check_once(self, *, now: float | None = None) -> list[dict[str, Any]]:
        stamp = time.monotonic() if now is None else now
        holders = [holder for holder in self.pool._holders if holder._in_use is not None]
        active = {id(holder) for holder in holders}
        self._first_seen = {key: value for key, value in self._first_seen.items() if key in active}
        specimens: list[dict[str, Any]] = []
        for holder in holders:
            key = id(holder)
            held_seconds = stamp - self._first_seen.setdefault(key, stamp)
            if held_seconds < self.threshold_seconds:
                continue
            if stamp - self._last_alarm.get(key, float("-inf")) < self.rate_limit_seconds:
                continue
            task = _task_for_connection(holder._con)
            owner_id = _owner_id(task)
            specimen = {
                "captured_at": datetime.now(UTC).isoformat(),
                "held_seconds": held_seconds,
                "owner_id": owner_id,
                "task": repr(task),
                "coroutine": repr(task.get_coro()) if task is not None else None,
                "task_stack": [
                    line
                    for frame in (task.get_stack() if task is not None else [])
                    for line in traceback.format_stack(frame)
                ],
                "pool": {
                    "size": self.pool.get_size(),
                    "idle_size": self.pool.get_idle_size(),
                    "holders": len(self.pool._holders),
                    "in_use": len(holders),
                },
                "pg_stat_activity": await self.inspect_pg(),
                "journal_events": await self.load_journal(owner_id),
            }
            self.specimen_dir.mkdir(parents=True, exist_ok=True)
            path = self.specimen_dir / f"held-connection-{time.time_ns()}.json"
            await asyncio.to_thread(path.write_text, json.dumps(specimen, default=str, indent=2))
            specimen["specimen_path"] = str(path)
            self.alarm("held_connection_watchdog", specimen)
            self._last_alarm[key] = stamp
            specimens.append(specimen)
        return specimens


class ThroughputDeadMan:
    def __init__(
        self,
        *,
        threshold_seconds: float,
        rate_limit_seconds: float,
        alarm: Callable[[str, dict[str, Any]], None] = _emit_alarm,
    ) -> None:
        self.threshold_seconds = threshold_seconds
        self.rate_limit_seconds = rate_limit_seconds
        self.alarm = alarm
        self._stalled_since: float | None = None
        self._last_alarm = float("-inf")

    def observe(self, *, claimed: int, completed: int, now: float | None = None) -> bool:
        stamp = time.monotonic() if now is None else now
        if claimed == 0 or completed > 0:
            self._stalled_since = None
            return False
        if self._stalled_since is None:
            self._stalled_since = stamp
            return False
        stalled_seconds = stamp - self._stalled_since
        if (
            stalled_seconds < self.threshold_seconds
            or stamp - self._last_alarm < self.rate_limit_seconds
        ):
            return False
        self.alarm(
            "throughput_dead_man",
            {
                "claimed_jobs": claimed,
                "completed_steps": completed,
                "stalled_seconds": stalled_seconds,
            },
        )
        self._last_alarm = stamp
        return True


async def run_production_watchdogs(
    pool: Any,
    db_url: str,
    *,
    held_threshold_seconds: float,
    dead_man_threshold_seconds: float,
    interval_seconds: float,
    rate_limit_seconds: float,
    specimen_dir: Path,
    journal_limit: int,
) -> None:
    """Run both observers using a substrate-different dedicated connection."""
    inspector = await asyncpg.connect(
        normalize_dsn(db_url), server_settings=LISTENER_TCP_KEEPALIVE_SETTINGS
    )

    async def inspect_pg() -> list[dict[str, Any]]:
        rows = await inspector.fetch(
            "SELECT pid, application_name, state, wait_event_type, wait_event, "
            "query_start, xact_start, query FROM pg_stat_activity "
            "WHERE datname = current_database() AND pid <> pg_backend_pid()"
        )
        return [dict(row) for row in rows]

    async def load_journal(owner_id: str | None) -> list[dict[str, Any]]:
        if owner_id is None:
            return []
        rows = await inspector.fetch(
            "SELECT seq, kind, data, created_at FROM events WHERE session_id = $1 "
            "UNION ALL SELECT seq, type::text AS kind, data, created_at "
            "FROM wf_run_events WHERE run_id = $1 ORDER BY seq DESC LIMIT $2",
            owner_id,
            journal_limit,
        )
        return [dict(row) for row in rows]

    watchdog = HeldConnectionWatchdog(
        pool,
        threshold_seconds=held_threshold_seconds,
        rate_limit_seconds=rate_limit_seconds,
        specimen_dir=specimen_dir,
        inspect_pg=inspect_pg,
        load_journal=load_journal,
    )
    dead_man = ThroughputDeadMan(
        threshold_seconds=dead_man_threshold_seconds,
        rate_limit_seconds=rate_limit_seconds,
    )
    try:
        while True:
            await asyncio.sleep(interval_seconds)
            await watchdog.check_once()
            row = await inspector.fetchrow(
                "SELECT "
                "(SELECT count(*) FROM procrastinate_jobs WHERE status = 'doing') AS claimed, "
                "(SELECT count(*) FROM events WHERE kind = 'span' "
                "AND data->>'event' = 'step_end' "
                "AND created_at >= now() - make_interval(secs => $1)) AS completed",
                interval_seconds,
            )
            dead_man.observe(claimed=int(row["claimed"]), completed=int(row["completed"]))
    finally:
        await inspector.close()
