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
from aios.harness.throughput import WAKE_COMPLETIONS, CompletionCounter
from aios.logging import get_logger

log = get_logger("aios.worker.watchdogs")

try:
    from prometheus_client import Counter

    _ALARMS = Counter("aios_worker_watchdog_alarms_total", "Worker watchdog alarms", ["kind"])
except Exception:  # pragma: no cover
    _ALARMS = None


try:
    from prometheus_client import Gauge

    _GC_CONSECUTIVE_FAILURES = Gauge(
        "aios_sandbox_gc_consecutive_failures",
        "Consecutive aborted snapshot GC ticks (0 after a successful tick).",
    )
    _GC_LAST_SUCCESS = Gauge(
        "aios_sandbox_gc_last_success_timestamp_seconds",
        "Unix time of the last successful snapshot GC tick (0 = none yet).",
    )
except Exception:  # pragma: no cover
    _GC_CONSECUTIVE_FAILURES = None
    _GC_LAST_SUCCESS = None


# Consecutive aborted GC ticks at which the snapshot GC counts as STALLED. One
# aborted tick is a hiccup; two in a row (two hours at the hourly cadence) is a
# disk reclaimer that is not running (aios#2343).
GC_STALLED_AFTER_FAILURES = 2


class SnapshotGcHealth:
    """Worker-side liveness state of the snapshot GC reconciler (aios#2343).

    The GC loop (``SandboxRegistry._gc_loop``) publishes here; the production
    watchdog loop reads it and alarms while the GC is stalled. It is
    deliberately in-process worker state: the API's ``/health`` runs in a
    different process and cannot see it.
    """

    def __init__(self) -> None:
        self.last_success_at: datetime | None = None
        self.consecutive_failures = 0
        self.last_error_class: str | None = None

    def record_success(self, *, at: datetime | None = None) -> None:
        self.last_success_at = at or datetime.now(UTC)
        self.consecutive_failures = 0
        self.last_error_class = None
        self._export()

    def record_failure(self, error: BaseException) -> int:
        self.consecutive_failures += 1
        self.last_error_class = type(error).__name__
        self._export()
        return self.consecutive_failures

    @property
    def stalled(self) -> bool:
        return self.consecutive_failures >= GC_STALLED_AFTER_FAILURES

    def snapshot(self) -> dict[str, Any]:
        return {
            "gc_last_success_at": (
                self.last_success_at.isoformat() if self.last_success_at else None
            ),
            "gc_consecutive_failures": self.consecutive_failures,
            "gc_last_error_class": self.last_error_class,
        }

    def _export(self) -> None:
        # Pure telemetry: a metrics failure must never reach the GC loop.
        with contextlib.suppress(Exception):
            if _GC_CONSECUTIVE_FAILURES is not None:
                _GC_CONSECUTIVE_FAILURES.set(self.consecutive_failures)
            if _GC_LAST_SUCCESS is not None and self.last_success_at is not None:
                _GC_LAST_SUCCESS.set(self.last_success_at.timestamp())


# The single worker-process instance the GC loop publishes to.
SNAPSHOT_GC_HEALTH = SnapshotGcHealth()


class SnapshotGcStallWatch:
    """Alarm (rate-limited) while the snapshot GC is stalled."""

    def __init__(
        self,
        health: SnapshotGcHealth,
        *,
        rate_limit_seconds: float,
        alarm: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.health = health
        self.rate_limit_seconds = rate_limit_seconds
        self.alarm = alarm if alarm is not None else _emit_alarm
        self._last_alarm = float("-inf")

    def check(self, *, now: float | None = None) -> bool:
        stamp = time.monotonic() if now is None else now
        if not self.health.stalled:
            self._last_alarm = float("-inf")
            return False
        if stamp - self._last_alarm < self.rate_limit_seconds:
            return False
        self.alarm("snapshot_gc_stalled", self.health.snapshot())
        self._last_alarm = stamp
        return True


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


WAKE_TASK_NAMES = ("harness.wake_session", "harness.wake_workflow")

CLAIMED_WAKES_SQL = (
    "SELECT count(*) FROM procrastinate_jobs WHERE status = 'doing' AND task_name = ANY($1::text[])"
)


class WakeThroughputProbe:
    """Sample ``(claimed, completed)`` for :class:`ThroughputDeadMan`.

    ``claimed`` is the live count of ``doing`` wake jobs. ``completed`` is the
    number of wake jobs this worker finished since the previous sample, read
    from the in-process :data:`~aios.harness.throughput.WAKE_COMPLETIONS`
    counter. It is NOT read from ``procrastinate_events``: under
    ``delete_jobs="successful"`` a successful job is deleted (events cascade)
    and never writes a ``succeeded`` event, so that count is structurally zero
    and cannot distinguish a stall from a busy worker (#2541).
    """

    def __init__(self, counter: CompletionCounter = WAKE_COMPLETIONS) -> None:
        self._counter = counter
        self._last_total = counter.total

    async def sample(
        self, conn: Any, *, query_timeout_seconds: float | None = None
    ) -> tuple[int, int]:
        claimed = await conn.fetchval(
            CLAIMED_WAKES_SQL, list(WAKE_TASK_NAMES), timeout=query_timeout_seconds
        )
        total = self._counter.total
        completed, self._last_total = total - self._last_total, total
        return int(claimed or 0), completed


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
    gc_health: SnapshotGcHealth | None = None,
) -> None:
    """Run fail-open observers.

    The snapshot-GC stall check reads in-process state and runs in its own
    loop; the DB-backed observers run on a reconnecting, dedicated connection.
    """
    dead_man = ThroughputDeadMan(
        threshold_seconds=dead_man_threshold_seconds, rate_limit_seconds=rate_limit_seconds
    )
    gc_stall_watch = SnapshotGcStallWatch(
        gc_health if gc_health is not None else SNAPSHOT_GC_HEALTH,
        rate_limit_seconds=rate_limit_seconds,
    )

    async def gc_stall_loop() -> None:
        # In-process state only, so it runs on its own cadence, fully
        # independent of the DB path below: a PostgreSQL outage, a hung
        # reconnect, or a slow DB observer must never mask a stalled disk
        # reclaimer (aios#2343).
        while True:
            try:
                gc_stall_watch.check()
            except Exception:
                log.exception("worker.gc_stall_check_failed")
            await asyncio.sleep(interval_seconds)

    gc_stall_task = asyncio.create_task(gc_stall_loop(), name="watchdog-gc-stall")
    try:
        await _run_db_watchdogs(
            pool,
            db_url,
            held_threshold_seconds=held_threshold_seconds,
            interval_seconds=interval_seconds,
            rate_limit_seconds=rate_limit_seconds,
            specimen_dir=specimen_dir,
            journal_limit=journal_limit,
            operation_timeout_seconds=operation_timeout_seconds,
            activity_limit=activity_limit,
            max_specimens=max_specimens,
            dead_man=dead_man,
        )
    finally:
        gc_stall_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await gc_stall_task


async def _run_db_watchdogs(
    pool: Any,
    db_url: str,
    *,
    held_threshold_seconds: float,
    interval_seconds: float,
    rate_limit_seconds: float,
    specimen_dir: Path,
    journal_limit: int,
    operation_timeout_seconds: float,
    activity_limit: int,
    max_specimens: int,
    dead_man: ThroughputDeadMan,
) -> None:
    """DB-backed observers on a reconnecting, dedicated connection."""
    inspector: Any = None
    backoff = min(1.0, interval_seconds)
    watchdog: HeldConnectionWatchdog | None = None
    probe = WakeThroughputProbe()

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
            claimed, completed = await probe.sample(
                inspector, query_timeout_seconds=operation_timeout_seconds
            )
            dead_man.observe(claimed=claimed, completed=completed)
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
