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


def _build_filesystem_probe_command(
    *,
    repo_sentinel: str | None = None,
    memory_sentinel: str | None = None,
) -> str:
    """Build the probe shell script with capability-focused semantics.

    Core assertions (always):
    - workspace write: create a temp file under /workspace, write, read back, delete
    - workspace read: verify readback matches

    Optional (only when configured sentinel is provided):
    - repo_sentinel: read a specific file.  Use ``.git/HEAD`` for a normal
      repository (a regular file inside the ``.git/`` directory) or ``.git``
      alone for a worktree checkout (where ``.git`` is itself a regular file
      containing ``gitdir: <path>``).  ``head -c 64`` works for both shapes.
    - memory_sentinel: read a specific memory mount file
    """
    lines = [
        "set -eu",
        "probe=$(mktemp /workspace/.aios-fs-probe.XXXXXX)",
        "trap 'rm -f \"$probe\"' EXIT",
        'printf aios-fs-probe > "$probe"',
        'test "$(cat "$probe")" = aios-fs-probe',
    ]
    if repo_sentinel:
        # Sentinel clarification: use ``.git/HEAD`` for a normal repo (a
        # regular file inside the ``.git/`` directory); use ``.git`` alone
        # for a worktree checkout (where ``.git`` is itself a regular file
        # containing ``gitdir: <path>``).  ``head -c 64`` reads the first
        # bytes and works for both shapes.
        lines.append(f"test -e {_sh_quote(repo_sentinel)}")
        lines.append(f"head -c 64 {_sh_quote(repo_sentinel)} >/dev/null")
    if memory_sentinel:
        lines.append(f"test -r {_sh_quote(memory_sentinel)}")
        lines.append(f"head -c 1 {_sh_quote(memory_sentinel)} >/dev/null")
    return "\n".join(lines) + "\n"


def _sh_quote(s: str) -> str:
    """Single-quote a string for safe shell interpolation."""
    return "'" + s.replace("'", "'\\''") + "'"


# Bounded grace period for cleanup operations (release, cancel-settle)
# after the main operation deadline expires.  This is the ONLY post-deadline
# allowance; sub-operations that compute ``remaining`` get zero budget once
# the deadline passes, and the probe fails immediately.
_CLEANUP_GRACE_SECONDS: float = 5.0


class _DeadlineExceeded(Exception):
    """The probe's hard overall deadline has passed — fail immediately."""


class StandingSessionFilesystemProbe:
    """Exercise the real sandbox and mounts of one configured standing session.

    Uses one hard overall deadline so the total wall-clock cost is bounded
    regardless of how many sub-operations run.  When the remaining budget
    hits zero, the probe raises ``_DeadlineExceeded`` immediately — there
    is no undocumented minimum grace on the operation path.  Cleanup
    (cancel-settle, release) runs under a separate, bounded
    ``_CLEANUP_GRACE_SECONDS`` cap that is explicitly tested.

    Ownership-aware lifecycle — generation token
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The probe must never release a sandbox that a concurrent consumer
    provisioned (or re-provisioned after the probe's peek).  Ownership is
    tracked via a **generation token**: the object identity of the handle
    returned by ``peek()`` before provisioning (``pre_handle``) vs. the
    handle returned by ``get_or_provision`` (``provision_handle``).

    * **Warm hit** (``pre_handle is not None``): the sandbox was already
      resident.  The probe uses it but does NOT release it.
    * **Cold provision** (``pre_handle is None``): the probe cold-started
      a sandbox.  After success/failure/timeout the probe re-peeks and
      releases **only if the current handle is the same object** it
      received from ``get_or_provision`` (``post_handle is provision_handle``).
      If a concurrent consumer re-provisioned between the probe's provision
      and its release, ``post_handle`` will be a *different* object and the
      release is skipped — the concurrent consumer owns that handle.

    Cancellation / timeout safety
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    On timeout or external cancellation the probe:
    1. Cancels the in-flight provision/exec task.
    2. Awaits (settles) it under ``_CLEANUP_GRACE_SECONDS`` so no
       provision/exec overlaps with a subsequent release.
    3. Releases only if the generation token still matches (see above).
    4. Re-raises ``CancelledError`` so the caller's cancellation
       propagates.
    """

    def __init__(
        self,
        registry: Any,
        pool: Any,
        session_id: str,
        *,
        rate_limit_seconds: float,
        operation_timeout_seconds: float,
        repo_sentinel: str | None = None,
        memory_sentinel: str | None = None,
        alarm: Callable[[str, dict[str, Any]], None] = _emit_alarm,
    ) -> None:
        self.registry = registry
        self.pool = pool
        self.session_id = session_id
        self.rate_limit_seconds = rate_limit_seconds
        self.operation_timeout_seconds = operation_timeout_seconds
        self.repo_sentinel = repo_sentinel
        self.memory_sentinel = memory_sentinel
        self.alarm = alarm
        self._last_alarm = float("-inf")
        self._command = _build_filesystem_probe_command(
            repo_sentinel=repo_sentinel,
            memory_sentinel=memory_sentinel,
        )

    async def _release_owned(self, provision_handle: object | None) -> None:
        """Release the sandbox only if the probe still owns it.

        Ownership is verified by comparing the current peek() handle's
        identity against ``provision_handle``.  If a concurrent consumer
        re-provisioned, peek() returns a different object and the release
        is skipped — the concurrent consumer owns that handle.

        Best-effort: a release failure is logged but never propagated — the
        idle reaper will converge.
        """
        if provision_handle is None:
            return
        current = self.registry.peek(self.session_id)
        if current is not provision_handle:
            # A concurrent consumer replaced our handle — do not release.
            return
        try:
            await self.registry.release(self.session_id)
        except Exception as err:
            log.warning(
                "standing_session_probe.release_failed",
                session_id=self.session_id,
                error=str(err),
            )

    @staticmethod
    def _remaining_or_fail(deadline: float) -> float:
        """Return seconds until *deadline*; raise immediately if ≤ 0.

        Uses ``time.monotonic()`` — *deadline* must be on the same clock.
        """
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _DeadlineExceeded("probe deadline exceeded")
        return remaining

    async def _settle_task(
        self, task: asyncio.Task[Any], grace: float = _CLEANUP_GRACE_SECONDS
    ) -> None:
        """Cancel *task* and await it within *grace* seconds."""
        task.cancel()
        with contextlib.suppress(BaseException):
            await asyncio.wait_for(asyncio.shield(task), timeout=grace)

    async def check_once(self, *, now: float | None = None) -> bool:
        mono = time.monotonic()
        stamp = mono if now is None else now
        # One hard overall deadline for the entire probe (provision + exec).
        # When remaining hits zero the probe fails immediately — no implicit
        # minimum grace.  Cleanup runs under _CLEANUP_GRACE_SECONDS.
        # The deadline always uses real monotonic time so _remaining_or_fail
        # comparisons are consistent; ``stamp`` is only for rate-limit bookkeeping.
        deadline = mono + self.operation_timeout_seconds

        # Ownership probe: is the sandbox already warm (owned by another
        # consumer)?  peek() is synchronous and lock-free.
        pre_handle = self.registry.peek(self.session_id)
        was_warm = pre_handle is not None

        # Track the in-flight coroutine so cancellation can settle it before
        # cleanup (no overlap / orphan).
        provision_task: asyncio.Task[Any] | None = None
        exec_task: asyncio.Task[Any] | None = None
        provision_handle: object | None = None
        try:
            remaining = self._remaining_or_fail(deadline)
            provision_coro = self.registry.get_or_provision(self.session_id, pool=self.pool)
            provision_task = asyncio.ensure_future(provision_coro)
            try:
                handle = await asyncio.wait_for(asyncio.shield(provision_task), remaining)
            except (TimeoutError, _DeadlineExceeded):
                # Settle the in-flight provision before cleanup so no
                # overlap with a subsequent release.
                await self._settle_task(provision_task)
                provision_task = None
                raise
            provision_task = None  # completed normally

            if not was_warm:
                # We cold-provisioned — record the handle for ownership checks.
                provision_handle = handle

            remaining = self._remaining_or_fail(deadline)
            exec_coro = self.registry.exec(
                handle,
                self._command,
                timeout_seconds=max(1, int(remaining)),
                max_output_bytes=4096,
            )
            exec_task = asyncio.ensure_future(exec_coro)
            try:
                result = await asyncio.wait_for(asyncio.shield(exec_task), remaining)
            except (TimeoutError, _DeadlineExceeded):
                await self._settle_task(exec_task)
                exec_task = None
                raise
            exec_task = None  # completed normally

            if result.exit_code == 0 and not result.timed_out:
                # Success: release if we cold-provisioned solely for monitoring.
                await self._release_owned(provision_handle)
                return True
            detail = {
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "stderr": result.stderr[-1024:],
            }
        except asyncio.CancelledError:
            # External cancellation: settle any in-flight operation before
            # cleanup so no provision/exec overlaps with the release.
            for task in (provision_task, exec_task):
                if task is not None and not task.done():
                    await self._settle_task(task)
            # If we cold-provisioned (or provision completed during cancel),
            # release only if we still own the handle.
            if provision_handle is None and not was_warm:
                # Provision may have completed before cancel settled;
                # check if a handle now exists that we created.
                post = self.registry.peek(self.session_id)
                if post is not None:
                    provision_handle = post
            await self._release_owned(provision_handle)
            raise
        except _DeadlineExceeded as exc:
            detail = {"error_type": "DeadlineExceeded", "error": str(exc)[-1024:]}
        except Exception as exc:
            detail = {"error_type": type(exc).__name__, "error": str(exc)[-1024:]}

        # Failure / timeout: release if we cold-provisioned solely for monitoring.
        await self._release_owned(provision_handle)

        if stamp - self._last_alarm >= self.rate_limit_seconds:
            self.alarm(
                "standing_session_filesystem_probe",
                {"session_id": self.session_id, **detail},
            )
            self._last_alarm = stamp
        return False


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
    sandbox_registry: Any | None = None,
    standing_session_id: str | None = None,
    filesystem_probe_interval_seconds: float = 300.0,
    filesystem_probe_timeout_seconds: float = 120.0,
    filesystem_probe_repo_sentinel: str | None = None,
    filesystem_probe_memory_sentinel: str | None = None,
) -> None:
    """Run fail-open observers on a reconnecting, dedicated connection."""
    inspector: Any = None
    backoff = min(1.0, interval_seconds)
    watchdog: HeldConnectionWatchdog | None = None
    dead_man = ThroughputDeadMan(
        threshold_seconds=dead_man_threshold_seconds, rate_limit_seconds=rate_limit_seconds
    )
    filesystem_probe = (
        StandingSessionFilesystemProbe(
            sandbox_registry,
            pool,
            standing_session_id,
            rate_limit_seconds=rate_limit_seconds,
            operation_timeout_seconds=filesystem_probe_timeout_seconds,
            repo_sentinel=filesystem_probe_repo_sentinel,
            memory_sentinel=filesystem_probe_memory_sentinel,
        )
        if sandbox_registry is not None and standing_session_id
        else None
    )

    next_filesystem_probe_at = time.monotonic()

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
            now = time.monotonic()
            if filesystem_probe is not None and now >= next_filesystem_probe_at:
                await filesystem_probe.check_once(now=now)
                next_filesystem_probe_at = now + filesystem_probe_interval_seconds
            row = await inspector.fetchrow(
                "SELECT (SELECT count(*) FROM procrastinate_jobs "
                "WHERE status = 'doing' AND task_name IN "
                "('harness.wake_session', 'harness.wake_workflow')) AS claimed, "
                "(SELECT count(*) FROM procrastinate_events e "
                "JOIN procrastinate_jobs j ON j.id = e.job_id "
                "WHERE e.type = 'succeeded' AND j.task_name IN "
                "('harness.wake_session', 'harness.wake_workflow') "
                "AND e.at >= now() - make_interval(secs => $1)) AS completed",
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
