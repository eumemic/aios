"""Fire-handler for triggers (#818).

Invoked via the procrastinate ``harness.run_trigger`` task. Loads the
trigger by id, then dispatches on two orthogonal axes:

- **Lifecycle on ``source``**: one-shot rows (``source == "one_shot"``) are
  DELETED BEFORE the action runs — at-most-once semantics: a worker crash
  between the delete and the action loses the fire silently rather than
  risking a duplicate via stuck-recovery re-claim. Cron rows record the
  outcome after and auto-disable once consecutive failures cross the
  threshold.
- **Execution on ``action.kind``**: ``sandbox_command`` runs a bash command
  in the session's sandbox (verbatim today's behavior); ``wake_owner``
  delivers ``content`` as a user-role message to the OWNING session via the
  in-worker self-delivery path (append + ``defer_wake``) — the
  ``wake_self`` primitive on a timer, NOT the capped cross-session
  ``wake_session`` tool.

A failed one-shot ``sandbox_command`` appends a synthetic
``[Scheduled wake '<name>' failed to deliver: <reason>]`` user message and
defers a wake so the agent has something to react to instead of an
unexplained silence. A failed one-shot ``wake_owner`` means the append
ITSELF failed (DB-level) — surfacing through the same append path would
fail identically, so we log and exit.
"""

from __future__ import annotations

from datetime import UTC, datetime

from aios.db import queries
from aios.errors import NotFoundError
from aios.harness import runtime
from aios.logging import get_logger
from aios.models.triggers import SandboxCommandAction, TriggerFireStatus, WakeOwnerAction
from aios.services import sessions as sessions_service
from aios.services.wake import defer_wake

log = get_logger("aios.harness.trigger_runner")

MAX_CONSECUTIVE_FAILURES = 5


async def run_trigger_step(trigger_id: str) -> None:
    """Run one fire of a trigger.

    Steps:
      1. Load the trigger; exit silently if it was removed, or had its
         owning session archived between claim and execute. Disabled
         one-shot rows are DELETED (the skip path leaves no row behind).
      2. For one-shot rows: DELETE the row in its own transaction BEFORE
         the action runs (at-most-once — a worker crash between delete and
         action loses the fire rather than risking a duplicate).
      3. Execute the action (sandbox_command or wake_owner).
      4. For cron: record the fire; auto-disable on threshold. For
         one-shot: on a sandbox_command failure, surface a synthetic
         wake-failure message.
    """
    pool = runtime.require_pool()

    try:
        async with pool.acquire() as conn:
            trigger = await queries.unscoped_get_trigger_row(conn, trigger_id)
    except NotFoundError:
        # Row was deleted between claim and execute (e.g. the agent or
        # operator removed it). No cleanup needed — the row is gone, so
        # ``running_since`` doesn't matter. Log and exit silently.
        log.info("trigger.skip_deleted", trigger_id=trigger_id)
        return

    is_one_shot = trigger.source == "one_shot"

    if trigger.session_archived_at is not None:
        # Session was archived between tick-claim and fire-handler-execute.
        # Archive is the lifecycle boundary: stop firing.
        log.info(
            "trigger.skip_archived",
            trigger_id=trigger_id,
            session_id=trigger.owner_session_id,
            name=trigger.name,
        )
        await _skip_claimed_fire(trigger, is_one_shot=is_one_shot)
        return

    if not trigger.enabled:
        # Disabled between claim and execute (the claim filter would never
        # re-claim it, but per-account cap and listing UX still see it).
        log.debug("trigger.skip_disabled", trigger_id=trigger_id)
        await _skip_claimed_fire(trigger, is_one_shot=is_one_shot)
        return

    # For one-shot rows, DELETE the row in its own transaction BEFORE the
    # action runs. This gives at-most-once: if the worker dies between this
    # commit and the action, the fire is lost — but there's no chance of a
    # duplicate landing via stuck-recovery re-claim 2h later.
    if is_one_shot:
        async with pool.acquire() as conn:
            await queries.delete_trigger_by_id(conn, trigger_id)
        log.info(
            "trigger.one_shot_deleted",
            trigger_id=trigger_id,
            session_id=trigger.owner_session_id,
            name=trigger.name,
        )

    started_at = datetime.now(UTC)
    action = trigger.action
    if isinstance(action, SandboxCommandAction):
        status, error_summary = await _run_sandbox_command(trigger, action)
    else:
        status, error_summary = await _run_wake_owner(trigger, action)

    new_failures = 0 if status == "ok" else trigger.consecutive_failures + 1

    if is_one_shot:
        # One-shot row: already deleted pre-fire. On a sandbox_command
        # failure, surface a synthetic user message so the agent isn't left
        # with unexplained silence (the command never delivered its marker).
        # A wake_owner failure means the append ITSELF failed (DB-level) —
        # the surface path appends through the same channel and would fail
        # identically, so it's skipped (already logged in _run_wake_owner).
        if status != "ok" and isinstance(action, SandboxCommandAction):
            await _surface_one_shot_failure(
                trigger.owner_session_id,
                trigger.account_id,
                trigger.name,
                status=status,
                error_summary=error_summary,
            )
    else:
        async with pool.acquire() as conn, conn.transaction():
            await queries.record_trigger_fire(
                conn,
                trigger_id,
                status=status,
                consecutive_failures=new_failures,
                fired_at=started_at,
            )
            if new_failures >= MAX_CONSECUTIVE_FAILURES:
                await queries.disable_trigger(conn, trigger_id)
                log.warning(
                    "trigger.auto_disabled",
                    trigger_id=trigger_id,
                    session_id=trigger.owner_session_id,
                    name=trigger.name,
                    consecutive_failures=new_failures,
                )
        # Surface the auto-disable AFTER the transaction commits — _surface_failure
        # re-acquires the pool, so nesting it inside the open transaction connection
        # risks deadlock on the small pool. Mirrors the one-shot path, which also
        # holds no transaction while surfacing.
        if new_failures >= MAX_CONSECUTIVE_FAILURES:
            content = (
                f"[Scheduled task '{trigger.name}' auto-disabled after "
                f"{MAX_CONSECUTIVE_FAILURES} consecutive failures: "
                f"{error_summary or status}]"
            )
            await _surface_failure(trigger.owner_session_id, trigger.account_id, content)


async def _skip_claimed_fire(trigger: queries.TriggerRow, *, is_one_shot: bool) -> None:
    """Resolve a claimed fire we've decided to skip (archived or disabled).

    One-shot rows are DELETED (otherwise the row sits forever and would
    re-fire on unarchive/re-enable with a stale intent); cron rows record a
    ``skipped`` outcome so ``last_fire_status`` is observable and
    ``running_since`` clears. The caller logs the distinct skip reason.
    """
    pool = runtime.require_pool()
    async with pool.acquire() as conn:
        if is_one_shot:
            await queries.delete_trigger_by_id(conn, trigger.id)
        else:
            await queries.record_trigger_fire(
                conn,
                trigger.id,
                status="skipped",
                consecutive_failures=trigger.consecutive_failures,
                fired_at=datetime.now(UTC),
            )


async def _run_sandbox_command(
    trigger: queries.TriggerRow, action: SandboxCommandAction
) -> tuple[TriggerFireStatus, str | None]:
    """Run a ``sandbox_command`` action — bash in the session's sandbox.

    Byte-identical to today's scheduled-task fire: statuses ok/error/timeout,
    a stderr-tail summary on non-zero exit.
    """
    pool = runtime.require_pool()
    sandbox_registry = runtime.require_sandbox_registry()
    status: TriggerFireStatus
    error_summary: str | None = None
    try:
        handle = await sandbox_registry.get_or_provision(trigger.owner_session_id, pool=pool)
        # Bind the sandbox-run method to a clearer local name.
        run_in_sandbox = sandbox_registry.exec
        result = await run_in_sandbox(
            handle,
            action.command,
            timeout_seconds=action.timeout_seconds,
            max_output_bytes=action.max_output_bytes,
        )
        if result.timed_out:
            status = "timeout"
            error_summary = f"command timed out after {action.timeout_seconds}s"
        elif result.exit_code != 0:
            status = "error"
            # Surface a short stderr tail so the failure event in the session
            # log carries actionable context.
            tail = (result.stderr or "").strip().splitlines()
            tail_text = " | ".join(tail[-3:])[:500] if tail else ""
            error_summary = f"command exited {result.exit_code}" + (
                f": {tail_text}" if tail_text else ""
            )
        else:
            status = "ok"
        log.info(
            "trigger.fired",
            trigger_id=trigger.id,
            session_id=trigger.owner_session_id,
            name=trigger.name,
            kind="sandbox_command",
            status=status,
            exit_code=result.exit_code,
        )
    except Exception as e:
        log.exception(
            "trigger.run_error",
            trigger_id=trigger.id,
            session_id=trigger.owner_session_id,
            name=trigger.name,
        )
        status = "error"
        error_summary = f"sandbox error: {type(e).__name__}: {e!s:.200}"
    return status, error_summary


async def _run_wake_owner(
    trigger: queries.TriggerRow, action: WakeOwnerAction
) -> tuple[TriggerFireStatus, str | None]:
    """Run a ``wake_owner`` action — self-delivery to the OWNING session.

    Appends ``content`` as a user-role message and defers a wake: the
    ``wake_self`` handler body executed in-worker. Explicitly NOT the
    ``wake_session`` tool handler — no depth counter, no per-pair rate
    limit, no cross-session reach. A trigger waking its own owner is the
    ``wake_self``-class primitive (already uncapped today via cron bash);
    no capability delta, only cheaper/more reliable. No sandbox, no broker.
    No wake_depth/wake_source metadata (this is self-delivery, not a chain
    link). Statuses: ok/error (timeout N/A).
    """
    pool = runtime.require_pool()
    try:
        await sessions_service.append_user_message(
            pool, trigger.owner_session_id, action.content, account_id=trigger.account_id
        )
        await defer_wake(
            pool, trigger.owner_session_id, cause="message", account_id=trigger.account_id
        )
        log.info(
            "trigger.fired",
            trigger_id=trigger.id,
            session_id=trigger.owner_session_id,
            name=trigger.name,
            kind="wake_owner",
            status="ok",
        )
        return "ok", None
    except Exception as e:
        log.exception(
            "trigger.wake_owner_error",
            trigger_id=trigger.id,
            session_id=trigger.owner_session_id,
            name=trigger.name,
        )
        return "error", f"wake delivery failed: {type(e).__name__}: {e!s:.200}"


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
            "trigger.surface_failure_failed",
            session_id=session_id,
            content=content,
        )


async def _surface_one_shot_failure(
    session_id: str,
    account_id: str,
    name: str,
    *,
    status: TriggerFireStatus,
    error_summary: str | None,
) -> None:
    """Append a session message + defer wake when a one-shot fire fails.

    The sandbox command's job for a ``schedule_wake``-style one-shot is to
    deliver a user-role message. When that fails (broker unreachable,
    sandbox error, command timeout), no marker reaches the session log and
    the agent has no way to know what happened. Synthesizing a user message
    here closes the loop: the model sees a deterministic failure event and
    can decide what to do next.
    """
    detail = error_summary or status
    content = f"[Scheduled wake '{name}' failed to deliver: {detail}]"
    await _surface_failure(session_id, account_id, content)
