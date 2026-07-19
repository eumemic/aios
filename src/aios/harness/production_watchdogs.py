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


def _task_for_connection(
    connection: object, holder: object | None = None
) -> asyncio.Task[Any] | None:
    """Find the borrower of an asyncpg holder's raw connection.

    Application frames contain ``PoolConnectionProxy``, not the raw
    ``holder._con``.  Match both the proxy's raw connection and holder.
    """
    current = asyncio.current_task()
    for task in list(asyncio.all_tasks())[:1000]:
        if task is current:
            continue
        for frame in task.get_stack(limit=32):
            for value in list(frame.f_locals.values())[:128]:
                if (
                    value is connection
                    or getattr(value, "_con", None) is connection
                    or (holder is not None and getattr(value, "_holder", None) is holder)
                ):
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
        operation_timeout_seconds: float = 5.0,
        max_specimens: int = 20,
    ) -> None:
        self.pool = pool
        self.threshold_seconds = threshold_seconds
        self.rate_limit_seconds = rate_limit_seconds
        self.specimen_dir = specimen_dir
        self.inspect_pg = inspect_pg
        self.load_journal = load_journal
        self.alarm = alarm
        self.operation_timeout_seconds = operation_timeout_seconds
        self.max_specimens = max_specimens
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
            task = _task_for_connection(holder._con, holder)
            owner_id = _owner_id(task)
            specimen = {
                "captured_at": datetime.now(UTC).isoformat(),
                "held_seconds": held_seconds,
                "owner_id": owner_id,
                "task": repr(task),
                "coroutine": repr(task.get_coro()) if task is not None else None,
                "task_stack": [
                    line[-4096:]
                    for frame in (task.get_stack(limit=32) if task is not None else [])
                    for line in traceback.format_stack(frame, limit=1)
                ],
                "pool": {
                    "size": self.pool.get_size(),
                    "idle_size": self.pool.get_idle_size(),
                    "holders": len(self.pool._holders),
                    "in_use": len(holders),
                },
                "pg_stat_activity": await asyncio.wait_for(
                    self.inspect_pg(), self.operation_timeout_seconds
                ),
                "journal_events": await asyncio.wait_for(
                    self.load_journal(owner_id), self.operation_timeout_seconds
                ),
            }
            self.specimen_dir.mkdir(parents=True, exist_ok=True)
            path = self.specimen_dir / f"held-connection-{time.time_ns()}.json"
            rendered = json.dumps(specimen, default=str, indent=2)
            await asyncio.wait_for(
                asyncio.to_thread(path.write_text, rendered), self.operation_timeout_seconds
            )
            # Bound forensic disk retention; this observer must not fill /tmp.
            old = sorted(
                self.specimen_dir.glob("held-connection-*.json"),
                key=lambda item: item.stat().st_mtime_ns,
                reverse=True,
            )
            for stale in old[self.max_specimens :]:
                with contextlib.suppress(OSError):
                    stale.unlink()
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
    operation_timeout_seconds: float = 5.0,
    activity_limit: int = 100,
    max_specimens: int = 20,
) -> None:
    """Run fail-open observers on a reconnecting, dedicated connection."""
    inspector: Any = None
    backoff = min(1.0, interval_seconds)
    watchdog: HeldConnectionWatchdog | None = None
    dead_man = ThroughputDeadMan(
        threshold_seconds=dead_man_threshold_seconds, rate_limit_seconds=rate_limit_seconds
    )

    while True:
        try:
            if inspector is None or inspector.is_closed():
                inspector = await asyncio.wait_for(
                    asyncpg.connect(
                        normalize_dsn(db_url),
                        server_settings=LISTENER_TCP_KEEPALIVE_SETTINGS,
                    ),
                    operation_timeout_seconds,
                )

                connected = inspector

                async def inspect_pg(connected: Any = connected) -> list[dict[str, Any]]:
                    rows = await connected.fetch(
                        "SELECT pid, application_name, state, wait_event_type, wait_event, "
                        "query_start, xact_start, left(query, 2000) AS query "
                        "FROM pg_stat_activity WHERE datname = current_database() "
                        "AND pid <> pg_backend_pid() ORDER BY query_start NULLS LAST LIMIT $1",
                        activity_limit,
                        timeout=operation_timeout_seconds,
                    )
                    return [dict(row) for row in rows]

                async def load_journal(
                    owner_id: str | None, connected: Any = connected
                ) -> list[dict[str, Any]]:
                    if owner_id is None:
                        return []
                    rows = await connected.fetch(
                        "SELECT seq, kind, data, created_at FROM ("
                        "SELECT seq, kind, left(data::text, 4000) AS data, created_at "
                        "FROM events WHERE session_id = $1 "
                        "UNION ALL SELECT seq, type::text AS kind, "
                        "left(payload::text, 4000) AS data, created_at "
                        "FROM wf_run_events WHERE run_id = $1) journal "
                        "ORDER BY created_at DESC LIMIT $2",
                        owner_id,
                        journal_limit,
                        timeout=operation_timeout_seconds,
                    )
                    return [dict(row) for row in rows]

                watchdog = HeldConnectionWatchdog(
                    pool,
                    threshold_seconds=held_threshold_seconds,
                    rate_limit_seconds=rate_limit_seconds,
                    specimen_dir=specimen_dir,
                    inspect_pg=inspect_pg,
                    load_journal=load_journal,
                    operation_timeout_seconds=operation_timeout_seconds,
                    max_specimens=max_specimens,
                )
                backoff = min(1.0, interval_seconds)

            await asyncio.sleep(interval_seconds)
            assert watchdog is not None
            await asyncio.wait_for(watchdog.check_once(), operation_timeout_seconds * 3)
            row = await inspector.fetchrow(
                "SELECT (SELECT count(*) FROM procrastinate_jobs "
                "WHERE status = 'doing' AND task_name IN "
                "('harness.wake_session', 'harness.wake_workflow')) AS claimed, "
                "((SELECT count(*) FROM events WHERE kind = 'span' "
                "AND data->>'event' = 'step_end' "
                "AND created_at >= now() - make_interval(secs => $1)) + "
                "(SELECT count(*) FROM wf_run_events WHERE type = 'run_completed' "
                "AND created_at >= now() - make_interval(secs => $1))) AS completed",
                interval_seconds,
                timeout=operation_timeout_seconds,
            )
            dead_man.observe(claimed=int(row["claimed"]), completed=int(row["completed"]))
        except asyncio.CancelledError:
            if inspector is not None:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(inspector.close(), operation_timeout_seconds)
            raise
        except Exception:
            # Pure telemetry: failure is never allowed onto worker fatal supervision.
            log.exception("worker.watchdog_tick_failed", retry_seconds=backoff)
            if inspector is not None:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(inspector.close(), operation_timeout_seconds)
            inspector = None
            await asyncio.sleep(backoff)
            backoff = min(max(interval_seconds, 1.0), backoff * 2)
