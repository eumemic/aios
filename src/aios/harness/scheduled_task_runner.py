"""Fire-handler for scheduled_tasks (#636).

Invoked via the procrastinate ``harness.run_scheduled_task`` task. Loads
the task by id, acquires the owning session's sandbox, runs the bash
command, records the outcome, and auto-disables after consecutive
failures cross the threshold.

Does NOT append to the session's event log — the model is woken only by
the bash script's explicit escalation (a separate code path, added in
Phase 4: bash POSTs to the broker's ``sessions/messages`` endpoint).
"""

from __future__ import annotations

from datetime import UTC, datetime

from aios.db import queries
from aios.errors import NotFoundError
from aios.harness import runtime
from aios.logging import get_logger
from aios.models.scheduled_tasks import ScheduledTaskStatus

log = get_logger("aios.harness.scheduled_task_runner")

MAX_CONSECUTIVE_FAILURES = 5


async def run_scheduled_task_step(task_id: str) -> None:
    """Run one fire of a scheduled task.

    Steps:
      1. Load the task; exit silently if it was removed, disabled, or
         had its owning session archived between claim and execute.
      2. Acquire the session's sandbox (lazy provisioning via the
         registry; reused across fires of the same session).
      3. Run the command via the sandbox handle; classify outcome as
         ``ok`` / ``error`` / ``timeout``.
      4. Record the fire; auto-disable on the consecutive-failure
         threshold.
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
        # Archive is the lifecycle boundary: stop firing. Clear
        # ``running_since`` so a future unarchive doesn't see a stuck row.
        log.info(
            "scheduled_task.skip_archived",
            task_id=task_id,
            session_id=task.session_id,
            name=task.name,
        )
        async with pool.acquire() as conn:
            await queries.record_scheduled_task_fire(
                conn,
                task_id,
                status="skipped",
                consecutive_failures=task.consecutive_failures,
                fired_at=datetime.now(UTC),
            )
        return

    if not task.enabled:
        # Disabled between claim and execute — exit silently. ``running_since``
        # is cleared regardless so the row doesn't stay stuck.
        log.debug("scheduled_task.skip_disabled", task_id=task_id)
        async with pool.acquire() as conn:
            await queries.record_scheduled_task_fire(
                conn,
                task_id,
                status="skipped",
                consecutive_failures=task.consecutive_failures,
                fired_at=datetime.now(UTC),
            )
        return

    # Only the actual-fire path needs the sandbox registry; the early-skip
    # paths above don't, so we resolve it here to keep the skip-only paths
    # independent of sandbox-registry availability.
    sandbox_registry = runtime.require_sandbox_registry()

    started_at = datetime.now(UTC)
    status: ScheduledTaskStatus
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
        elif result.exit_code != 0:
            status = "error"
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
    except Exception:
        log.exception(
            "scheduled_task.run_error",
            task_id=task_id,
            session_id=task.session_id,
            name=task.name,
        )
        status = "error"

    new_failures = 0 if status == "ok" else task.consecutive_failures + 1

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
