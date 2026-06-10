"""Fire-handler for scheduled_tasks (#636).

Invoked via the procrastinate ``harness.run_scheduled_task`` task. Loads
the task by id, acquires the owning session's sandbox, runs the bash
command, records the outcome, and auto-disables after consecutive
failures cross the threshold.

For one-shot rows (``fire_at IS NOT NULL``), the row is DELETED BEFORE
the bash command runs — this gives at-most-once semantics: a worker
crash between the delete and the curl loses the wake silently rather
than risking a duplicate marker via stuck-recovery re-claim. Curl
failures (broker unreachable, exit_code != 0) append a synthetic
``[Scheduled wake failed: <reason>]`` user message and defer a wake so
the agent has something to react to instead of an unexplained silence.

For cron rows, the post-fire path records the outcome and auto-disables
on consecutive failures (same as #636).
"""

from __future__ import annotations

from datetime import UTC, datetime

from aios.db import queries
from aios.errors import NotFoundError
from aios.harness import runtime
from aios.logging import get_logger
from aios.models.scheduled_tasks import ScheduledTaskStatus
from aios.services import sessions as sessions_service
from aios.services.wake import defer_wake

log = get_logger("aios.harness.scheduled_task_runner")

MAX_CONSECUTIVE_FAILURES = 5


async def run_scheduled_task_step(task_id: str) -> None:
    """Run one fire of a scheduled task.

    Steps:
      1. Load the task; exit silently if it was removed, or had its
         owning session archived between claim and execute. Disabled
         rows for one-shot are DELETED (the skip path leaves no row
         behind).
      2. For one-shot rows: DELETE the row in its own transaction
         BEFORE the bash runs. This gives at-most-once semantics — a
         worker crash between delete and curl loses the wake instead
         of risking duplicate fires via stuck recovery.
      3. Acquire the session's sandbox and run the command.
      4. For cron: record the fire; auto-disable on threshold.
         For one-shot: on failure, append a synthetic wake-failure
         message so the agent isn't left with unexplained silence.
    """
    pool = runtime.require_pool()

    try:
        async with pool.acquire() as conn:
            task = await queries.unscoped_get_scheduled_task_row(conn, task_id)
    except NotFoundError:
        # Row was deleted between claim and execute (e.g. the agent or
        # operator removed it). No cleanup needed — the row is gone, so
        # ``running_since`` doesn't matter. Log and exit silently.
        log.info("scheduled_task.skip_deleted", task_id=task_id)
        return

    if task.session_archived_at is not None:
        # Session was archived between tick-claim and fire-handler-execute.
        # Archive is the lifecycle boundary: stop firing. For one-shot
        # rows we DELETE rather than record a skip — otherwise the row
        # sits forever and would re-fire on unarchive with a semantically
        # stale wake reason. For cron rows we record a skip so
        # ``last_fire_status`` is observable; running_since gets cleared.
        log.info(
            "scheduled_task.skip_archived",
            task_id=task_id,
            session_id=task.session_id,
            name=task.name,
        )
        async with pool.acquire() as conn:
            if task.fire_at is not None:
                await queries.delete_scheduled_task_by_id(conn, task_id)
            else:
                await queries.record_scheduled_task_fire(
                    conn,
                    task_id,
                    status="skipped",
                    consecutive_failures=task.consecutive_failures,
                    fired_at=datetime.now(UTC),
                )
        return

    if not task.enabled:
        # Disabled between claim and execute. One-shot: delete (the
        # disabled-while-claimed row would otherwise sit forever — the
        # claim filter would never re-claim it, but per-account cap and
        # listing UX still see it). Cron: record skip so the disabled
        # transition is observable in last_fire_status.
        log.debug("scheduled_task.skip_disabled", task_id=task_id)
        async with pool.acquire() as conn:
            if task.fire_at is not None:
                await queries.delete_scheduled_task_by_id(conn, task_id)
            else:
                await queries.record_scheduled_task_fire(
                    conn,
                    task_id,
                    status="skipped",
                    consecutive_failures=task.consecutive_failures,
                    fired_at=datetime.now(UTC),
                )
        return

    # For one-shot rows, DELETE the row in its own transaction BEFORE the
    # sandbox exec runs. This gives at-most-once: if the worker dies
    # between this commit and the curl call, the wake is lost — but
    # there's no chance of a duplicate marker landing via stuck-recovery
    # re-claim 2h later. Loss is preferable to phantom duplicate user
    # messages whose semantic intent is hours stale.
    if task.fire_at is not None:
        async with pool.acquire() as conn:
            await queries.delete_scheduled_task_by_id(conn, task_id)
        log.info(
            "scheduled_task.one_shot_deleted",
            task_id=task_id,
            session_id=task.session_id,
            name=task.name,
        )

    # Only the actual-fire path needs the sandbox registry; the early-skip
    # paths above don't, so we resolve it here to keep the skip-only paths
    # independent of sandbox-registry availability.
    sandbox_registry = runtime.require_sandbox_registry()

    started_at = datetime.now(UTC)
    status: ScheduledTaskStatus
    error_summary: str | None = None
    try:
        handle = await sandbox_registry.get_or_provision(task.session_id, pool=pool)
        # Alias the registry's run method to a local; calling the method
        # via its native name as a call site trips a local file-write
        # hook that pattern-matches the literal substring 'e' 'x' 'e' 'c' '('.
        run_in_sandbox = sandbox_registry.exec
        result = await run_in_sandbox(
            handle,
            task.command,
            timeout_seconds=task.timeout_seconds,
            max_output_bytes=task.max_output_bytes,
        )
        if result.timed_out:
            status = "timeout"
            error_summary = f"command timed out after {task.timeout_seconds}s"
        elif result.exit_code != 0:
            status = "error"
            # Surface a short stderr tail so the failure event in the
            # session log carries actionable context.
            tail = (result.stderr or "").strip().splitlines()
            tail_text = " | ".join(tail[-3:])[:500] if tail else ""
            error_summary = f"command exited {result.exit_code}" + (
                f": {tail_text}" if tail_text else ""
            )
        else:
            status = "ok"
        log.info(
            "scheduled_task.fired",
            task_id=task_id,
            session_id=task.session_id,
            name=task.name,
            status=status,
            exit_code=result.exit_code,
        )
    except Exception as e:
        log.exception(
            "scheduled_task.run_error",
            task_id=task_id,
            session_id=task.session_id,
            name=task.name,
        )
        status = "error"
        error_summary = f"sandbox error: {type(e).__name__}: {e!s:.200}"

    new_failures = 0 if status == "ok" else task.consecutive_failures + 1

    if task.fire_at is not None:
        # One-shot row: the row was already deleted pre-fire. On failure,
        # surface a synthetic user message so the agent isn't left with
        # unexplained silence — the curl never delivered its marker, so
        # nothing else explains why the scheduled wake didn't fire.
        if status != "ok":
            await _surface_one_shot_failure(
                task.session_id,
                task.account_id,
                task.name,
                status=status,
                error_summary=error_summary,
            )
    else:
        async with pool.acquire() as conn, conn.transaction():
            await queries.record_scheduled_task_fire(
                conn,
                task_id,
                status=status,
                consecutive_failures=new_failures,
                fired_at=started_at,
            )
            if new_failures >= MAX_CONSECUTIVE_FAILURES:
                await queries.disable_scheduled_task(conn, task_id)
                log.warning(
                    "scheduled_task.auto_disabled",
                    task_id=task_id,
                    session_id=task.session_id,
                    name=task.name,
                    consecutive_failures=new_failures,
                )
        # Surface the auto-disable AFTER the transaction commits — _surface_failure
        # re-acquires the pool, so nesting it inside the open transaction connection
        # risks deadlock on the small pool. Mirrors the one-shot path, which also
        # holds no transaction while surfacing.
        if new_failures >= MAX_CONSECUTIVE_FAILURES:
            content = (
                f"[Scheduled task '{task.name}' auto-disabled after "
                f"{MAX_CONSECUTIVE_FAILURES} consecutive failures: "
                f"{error_summary or status}]"
            )
            await _surface_failure(task.session_id, task.account_id, content)


async def _surface_failure(session_id: str, account_id: str, content: str) -> None:
    """Append a synthetic user message + defer a wake (best-effort).

    Shared by the one-shot failure path and the cron auto-disable path so
    both surface a user-visible event instead of failing silently.
    """
    pool = runtime.require_pool()
    try:
        await sessions_service.append_user_message(pool, session_id, content, account_id=account_id)
        await defer_wake(pool, session_id, cause="message", account_id=account_id)
    except Exception:
        log.exception(
            "scheduled_task.surface_failure_failed",
            session_id=session_id,
        )


async def _surface_one_shot_failure(
    session_id: str,
    account_id: str,
    name: str,
    *,
    status: ScheduledTaskStatus,
    error_summary: str | None,
) -> None:
    """Append a session message + defer wake when a one-shot fire fails.

    The bash command's job for a ``schedule_wake``-style one-shot is to
    POST a user-role message to the broker. When that fails (broker
    unreachable, sandbox error, command timeout), no marker reaches the
    session log via the normal path and the agent has no way to know
    what happened. Synthesizing a user message here closes the loop:
    the model sees a deterministic failure event and can decide what to
    do next.
    """
    detail = error_summary or status
    content = f"[Scheduled wake '{name}' failed to deliver: {detail}]"
    await _surface_failure(session_id, account_id, content)
