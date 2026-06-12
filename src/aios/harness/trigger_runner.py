"""Fire-handler for triggers (#818, #819).

Invoked via the procrastinate ``harness.run_trigger`` task. Loads the
trigger by id, then dispatches on two orthogonal axes:

- **Lifecycle on the FIRE'S ORIGIN** (never the reloaded row's ``source`` —
  the row is user-mutable between claim and fire, and a mid-flight
  ``run_completion → one_shot`` conversion must not route an event fire into
  the one-shot DELETE arm and destroy the user's fresh trigger):

  * Tick fires (``trigger_run_id is None``): one-shot rows are DELETED BEFORE
    the action runs — at-most-once semantics — and write a terminal
    ``trigger_runs`` audit row after (by then the only persistent record the
    fire ever happened); cron rows record the outcome + a ``cron`` audit row
    in one transaction and auto-disable once consecutive failures hit the
    threshold.
  * Event fires (``trigger_run_id`` set — a ``run_completion`` dispatch): the
    pre-inserted ``trigger_runs`` carrier row is claimed ``pending→running``
    INSTEAD of the tick's ``running_since`` gate (``None`` = a duplicate job
    lost the claim race — exit without firing), and finalized terminal with
    the outcome. Event fires are unserialized by design: distinct completions
    of a watched workflow fire concurrently; the failure counter stays
    coherent via ``record_trigger_fire``'s SQL-side CASE.

- **Execution on ``action.kind``**: ``sandbox_command`` runs bash in the
  session's sandbox; ``wake_owner`` delivers ``content`` as a user-role
  message to the OWNING session (the ``wake_self`` primitive — content is
  delivered VERBATIM even for event fires; the workflow action is the
  event-consuming action); ``workflow`` launches a run of a workflow via the
  unmodified ``create_run`` with ``launcher_session_id = owner`` — so EVERY
  fire re-clamps the run's surface to the owner's current agent, re-checks
  ``vault_ids`` against the owner's current vaults, and counts against the
  owner's outstanding-run cap. ``parent_run_id`` threads the completing run's
  id (event fires) or the owner session's own lineage (timer fires), so the
  existing depth cap bounds reactive cascades and self-fire loops by
  construction.

One ``started_at`` per fire: the carrier claim stamp, the composed input's
``fired_at``, and ``record_trigger_fire``'s stamp are the same value, so the
audit and the delivered input can never disagree.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

import asyncpg

from aios.config import get_settings
from aios.db import queries
from aios.db.queries import workflows as wf_queries
from aios.errors import NotFoundError
from aios.harness import runtime
from aios.logging import get_logger
from aios.models.triggers import (
    SandboxCommandAction,
    TriggerFireStatus,
    WakeOwnerAction,
    WorkflowAction,
)
from aios.models.workflows import WfRun
from aios.services import sessions as sessions_service
from aios.services.wake import defer_trigger_fire, defer_wake

log = get_logger("aios.harness.trigger_runner")

MAX_CONSECUTIVE_FAILURES = 5


def compose_workflow_run_input(
    *,
    trigger_id: str,
    trigger_name: str,
    source: str,
    fired_at: datetime,
    input_template: Any,
    completed_run: WfRun | None = None,
    completed_error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The fired run's input — deterministic envelope composition.

    ALWAYS ``{"trigger": <firing context>, "input": <input_template
    verbatim>}``: the author's template is never parsed or mutated (a template
    carrying its own ``"trigger"`` key nests harmlessly under ``"input"``),
    and there is no placeholder language to fail at fire time. For
    run_completion fires the completing run rides BY VALUE under
    ``trigger.run`` — a workflow script has no capability to read another run,
    so by-reference would strand the data. ``trigger.run.error`` mirrors the
    ``WfRunWaitResponse`` shape: the ``{'kind': …}`` from the run_completed
    journal event, ``None`` unless the watched run errored.
    """
    trigger: dict[str, Any] = {
        "id": trigger_id,
        "name": trigger_name,
        "source": source,
        "fired_at": fired_at.isoformat(),
    }
    if completed_run is not None:
        trigger["run"] = {
            "id": completed_run.id,
            "workflow_id": completed_run.workflow_id,
            "status": completed_run.status,  # completed | errored | cancelled
            "output": completed_run.output,  # row value: non-null only on completed
            "error": completed_error,
        }
    return {"trigger": trigger, "input": input_template}


async def run_trigger_step(trigger_id: str, trigger_run_id: str | None = None) -> None:
    """Run one fire of a trigger.

    Steps:
      1. For event fires: claim the ``trigger_runs`` carrier row
         (``pending → running``); a lost claim means a duplicate job — exit.
      2. Load the trigger; a row deleted between claim and execute finalizes
         the carrier as ``skipped`` (event) or exits silently (tick).
      3. Archived-owner / disabled rows resolve the claimed fire as a skip
         (see :func:`_skip_claimed_fire` for the per-origin matrix).
      4. For one-shot tick fires: DELETE the row BEFORE the action runs
         (at-most-once — a worker crash between delete and action loses the
         fire rather than risking a duplicate via stuck-recovery re-claim).
      5. Execute the action; record the outcome (echo columns + audit row /
         carrier finalize, one transaction for standing rows) and
         auto-disable at the failure threshold.
    """
    pool = runtime.require_pool()
    started_at = datetime.now(UTC)

    event: dict[str, Any] | None = None
    async with pool.acquire() as conn:
        if trigger_run_id is not None:
            event = await queries.claim_trigger_run(conn, trigger_run_id, started_at=started_at)
            if event is None:
                # Already claimed: a sweep re-defer raced the live job. The
                # claim, not queue dedup, is what makes that race safe.
                log.info("trigger.fire_already_claimed", trigger_run_id=trigger_run_id)
                return
        try:
            trigger = await queries.unscoped_get_trigger_row(conn, trigger_id)
        except NotFoundError:
            # Row was deleted between claim and execute (e.g. the agent or
            # operator removed it). For an event fire the carrier row outlives
            # the trigger by design and MUST be finalized — an unfinished
            # claim row would be sweep-re-deferred until retention pruned it.
            log.info("trigger.skip_deleted", trigger_id=trigger_id)
            if trigger_run_id is not None:
                await queries.finalize_trigger_run(
                    conn, trigger_run_id, status="skipped", error_summary="trigger deleted"
                )
            return

    # Lifecycle derives from the fire's ORIGIN: an event fire never takes the
    # one-shot arm, no matter what the row's source says at reload (the fire's
    # source identity was fixed at match time, stamped on the carrier row).
    is_one_shot = trigger_run_id is None and trigger.source == "one_shot"

    if trigger.session_archived_at is not None:
        # Session was archived between claim and execute. Archive is the
        # lifecycle boundary: stop firing.
        log.info(
            "trigger.skip_archived",
            trigger_id=trigger_id,
            session_id=trigger.owner_session_id,
            name=trigger.name,
        )
        await _skip_claimed_fire(
            trigger,
            trigger_run_id=trigger_run_id,
            reason="owner session archived",
            fired_at=started_at,
        )
        return

    if not trigger.enabled:
        # Disabled between claim and execute (the claim filter would never
        # re-claim it, but per-account cap and listing UX still see it).
        log.debug("trigger.skip_disabled", trigger_id=trigger_id)
        await _skip_claimed_fire(
            trigger,
            trigger_run_id=trigger_run_id,
            reason="trigger disabled",
            fired_at=started_at,
        )
        return

    # For one-shot tick fires, DELETE the row in its own transaction BEFORE
    # the action runs. This gives at-most-once: if the worker dies between
    # this commit and the action, the fire is lost — but there's no chance of
    # a duplicate landing via stuck-recovery re-claim 2h later.
    if is_one_shot:
        async with pool.acquire() as conn:
            await queries.delete_trigger_by_id(conn, trigger_id)
        log.info(
            "trigger.one_shot_deleted",
            trigger_id=trigger_id,
            session_id=trigger.owner_session_id,
            name=trigger.name,
        )

    action = trigger.action
    if isinstance(action, SandboxCommandAction):
        status, error_summary, result_id = await _run_sandbox_command(trigger, action)
    elif isinstance(action, WakeOwnerAction):
        status, error_summary, result_id = await _run_wake_owner(trigger, action)
    else:
        status, error_summary, result_id = await _run_workflow(
            trigger, action, event=event, started_at=started_at
        )

    if is_one_shot:
        # Surface a one-shot failure FIRST (slice-1 ordering: the agent must
        # hear about it even if the audit insert below then fails) — the
        # sandbox command's job was to deliver a marker that never arrived
        # (the string is byte-frozen — backfilled rows keep producing it); a
        # workflow launch failure gets its own runtime-truthful string
        # (nothing was a "wake" and nothing was being "delivered"). A failed
        # wake_owner means the append ITSELF failed (DB-level) — surfacing
        # through the same append path would fail identically; logged in
        # _run_wake_owner. _surface_failure is best-effort (swallows).
        if status != "ok":
            if isinstance(action, SandboxCommandAction):
                await _surface_failure(
                    trigger.owner_session_id,
                    trigger.account_id,
                    f"[Scheduled wake '{trigger.name}' failed to deliver: "
                    f"{error_summary or status}]",
                )
            elif isinstance(action, WorkflowAction):
                await _surface_failure(
                    trigger.owner_session_id,
                    trigger.account_id,
                    f"[Trigger '{trigger.name}' failed to launch its workflow run: "
                    f"{error_summary or status}]",
                )
        # Row already deleted pre-fire; the terminal audit row is now the only
        # persistent record the fire ever happened. (A crash between the
        # action and this insert loses the record — the at-most-once spirit;
        # journaling before the action is forbidden for tick fires.)
        async with pool.acquire() as conn:
            await _record_timer_audit(
                conn,
                trigger,
                trigger_context="one_shot",
                status=status,
                error_summary=error_summary,
                result_id=result_id,
                started_at=started_at,
            )
        return

    # Standing rows (cron tick fires + run_completion event fires): echo
    # columns, the audit record, and the auto-disable decision commit in ONE
    # transaction, so the echo cache and the audit can never disagree.
    async with pool.acquire() as conn, conn.transaction():
        failures = await queries.record_trigger_fire(
            conn,
            trigger_id,
            status=status,
            fired_at=started_at,
            # Event fires never held the tick's running_since claim; clearing
            # it could release a CONCURRENT tick fire's overlap lease (see
            # record_trigger_fire).
            clear_running_since=trigger_run_id is None,
        )
        if trigger_run_id is not None:
            await queries.finalize_trigger_run(
                conn,
                trigger_run_id,
                status=status,
                error_summary=error_summary,
                result_id=result_id,
            )
        else:
            await _record_timer_audit(
                conn,
                trigger,
                trigger_context="cron",
                status=status,
                error_summary=error_summary,
                result_id=result_id,
                started_at=started_at,
            )
        # ``failures is None`` = the trigger row vanished mid-fire (an API
        # DELETE raced us) — benign; the carrier finalize above still landed.
        # The gate is ``==``, not ``>=``: under concurrent event fires a
        # straggler that increments past the threshold after the disable must
        # not re-spam the owner with a second surfaced message.
        auto_disable = failures is not None and failures == MAX_CONSECUTIVE_FAILURES
        if auto_disable:
            await queries.disable_trigger(conn, trigger_id)
            log.warning(
                "trigger.auto_disabled",
                trigger_id=trigger_id,
                session_id=trigger.owner_session_id,
                name=trigger.name,
                consecutive_failures=failures,
            )
    # Surface the auto-disable AFTER the transaction commits — _surface_failure
    # re-acquires the pool, so nesting it inside the open transaction connection
    # risks deadlock on the small pool.
    if auto_disable:
        content = (
            f"[Scheduled task '{trigger.name}' auto-disabled after "
            f"{MAX_CONSECUTIVE_FAILURES} consecutive failures: "
            f"{error_summary or status}]"
        )
        await _surface_failure(trigger.owner_session_id, trigger.account_id, content)


async def _skip_claimed_fire(
    trigger: queries.TriggerRow,
    *,
    trigger_run_id: str | None,
    reason: str,
    fired_at: datetime,
) -> None:
    """Resolve a claimed fire we've decided to skip (archived or disabled).

    Per-origin matrix (origin derived the same way as the caller's
    ``is_one_shot``):

    - EVENT fire: record ``skipped`` on the trigger row (counter unchanged)
      AND finalize the carrier row — an unfinished claim row would be
      sweep-re-deferred until retention pruned it.
    - ONE-SHOT tick fire: DELETE the row (otherwise it sits forever and would
      re-fire on unarchive/re-enable with a stale intent) and write a
      ``skipped`` tombstone audit row — without it the skip leaves zero
      record anywhere (the row deletes and takes ``last_fire_*`` with it).
    - CRON tick fire: record ``skipped`` so ``last_fire_status`` is observable
      and ``running_since`` clears; no audit row (the action never executed).
    """
    pool = runtime.require_pool()
    if trigger_run_id is not None:
        async with pool.acquire() as conn, conn.transaction():
            await queries.record_trigger_fire(
                conn,
                trigger.id,
                status="skipped",
                fired_at=fired_at,
                clear_running_since=False,  # event fires never held the tick claim
            )
            await queries.finalize_trigger_run(
                conn, trigger_run_id, status="skipped", error_summary=reason
            )
    elif trigger.source == "one_shot":
        # DELETE first, tombstone second, deliberately NOT one transaction:
        # the delete invariant ("no stale intent left behind to re-fire on
        # unarchive/re-enable") must win — coupling them would let a failed
        # audit INSERT roll back the delete and resurrect the stale row. A
        # crash between the two statements is slice-1 behavior exactly (row
        # gone, no record).
        async with pool.acquire() as conn:
            await queries.delete_trigger_by_id(conn, trigger.id)
            await _record_timer_audit(
                conn,
                trigger,
                trigger_context="one_shot",
                status="skipped",
                error_summary=reason,
                result_id=None,
                started_at=fired_at,
            )
    else:
        async with pool.acquire() as conn:
            await queries.record_trigger_fire(conn, trigger.id, status="skipped", fired_at=fired_at)


async def _record_timer_audit(
    conn: asyncpg.Connection[Any],
    trigger: queries.TriggerRow,
    *,
    trigger_context: str,
    status: TriggerFireStatus,
    error_summary: str | None,
    result_id: str | None,
    started_at: datetime,
) -> None:
    """Write a timer fire's audit row — owns the TriggerRow → kwargs projection."""
    await queries.record_trigger_run(
        conn,
        trigger_id=trigger.id,
        account_id=trigger.account_id,
        owner_session_id=trigger.owner_session_id,
        trigger_name=trigger.name,
        trigger_context=trigger_context,
        status=status,
        error_summary=error_summary,
        result_id=result_id,
        started_at=started_at,
    )


async def _run_sandbox_command(
    trigger: queries.TriggerRow, action: SandboxCommandAction
) -> tuple[TriggerFireStatus, str | None, str | None]:
    """Run a ``sandbox_command`` action — bash in the session's sandbox.

    Byte-identical to today's scheduled-task fire: statuses ok/error/timeout,
    a stderr-tail summary on non-zero exit. Produces no resource
    (``result_id`` is always ``None``).
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
    return status, error_summary, None


async def _run_wake_owner(
    trigger: queries.TriggerRow, action: WakeOwnerAction
) -> tuple[TriggerFireStatus, str | None, str | None]:
    """Run a ``wake_owner`` action — self-delivery to the OWNING session.

    Appends ``content`` as a user-role message and defers a wake: the
    ``wake_self`` handler body executed in-worker. Explicitly NOT the
    ``wake_session`` tool handler — no depth counter, no per-pair rate
    limit, no cross-session reach. ``content`` is delivered VERBATIM for
    event fires too (no event interpolation — a wake is a "something
    completed, go look" ping; the model can list runs, and the workflow
    action is the event-consuming action). Statuses: ok/error (timeout N/A);
    no resource produced.
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
        return "ok", None, None
    except Exception as e:
        log.exception(
            "trigger.wake_owner_error",
            trigger_id=trigger.id,
            session_id=trigger.owner_session_id,
            name=trigger.name,
        )
        return "error", f"wake delivery failed: {type(e).__name__}: {e!s:.200}", None


async def _run_workflow(
    trigger: queries.TriggerRow,
    action: WorkflowAction,
    *,
    event: dict[str, Any] | None,
    started_at: datetime,
) -> tuple[TriggerFireStatus, str | None, str | None]:
    """Run a ``workflow`` action — launch a run, deterministic, no model wake.

    ``'ok'`` means the run was CREATED (launch semantics): the run executes
    asynchronously and its own outcome is its own audit trail, reachable via
    the returned ``result_id`` — observing run outcomes is what a
    ``run_completion`` watcher is for. Statuses: ok/error (timeout N/A — the
    launch is one DB transaction).

    All owner authority flows from ``launcher_session_id``: every fire
    re-clamps the surface, re-checks the vault subset against the owner's
    CURRENT vaults, asserts the version pin at the script-snapshot consistency
    point, and counts against the owner's outstanding-run cap. Every launch
    error is loud and feeds the failure counter — ``ConflictError`` (pin
    drift), ``ForbiddenError`` (vault breach), ``RateLimitedError``
    (outstanding caps; deliberately an error, never ``skipped`` — a
    cap-saturated trigger must trip auto-disable, not drop fires silently),
    ``WorkflowRunDepthExceededError`` (a reactive cascade or self-fire loop
    hit the depth cap — the loop's structural bound), ``ConflictError``
    (workflow version drift or an archived target workflow), and ``NotFoundError``
    (a vault in ``action.vault_ids`` was deleted, or the owner session was
    deleted mid-fire; workflows and environments have no hard-delete path).
    """
    # Lazy: aios.workflows.service transitively imports the tools package,
    # which imports services.workflows, which imports back into
    # aios.workflows.service — a cycle when trigger_runner is the entry
    # module. Fire-time import, the tasks.py idiom.
    from aios.workflows import service as wf_run_service

    pool = runtime.require_pool()
    # The iff CHECK guarantees workflow-kind rows carry the column.
    assert trigger.environment_id is not None
    try:
        completed_run: WfRun | None = None
        completed_error: dict[str, Any] | None = None
        parent_run_id: str | None
        if event is not None:
            # Read the watched run at fire time, ACCOUNT-SCOPED (never the
            # unscoped step-getter): if the carrier's run id were ever
            # mismatched to a foreign run, the fire fails NotFound instead of
            # embedding another tenant's output. Safe to read late: terminal
            # statuses are monotonic and runs are never deleted.
            async with pool.acquire() as conn:
                completed_run = await wf_queries.get_wf_run(
                    conn, event["run_id"], account_id=trigger.account_id
                )
                if completed_run.status == "errored":
                    completed_error = await wf_queries.resolve_run_error(conn, completed_run.id)
            # The completing run is the lineage parent: same-account by the
            # matcher's account-equality conjunct, and the existing depth cap
            # then bounds completion→fire→run→completion cycles at
            # WORKFLOW_RUN_MAX_DEPTH by construction.
            parent_run_id = completed_run.id
        else:
            # Timer fires inherit the owner session's own (immutable) lineage
            # — exactly what the create_run builtin threads, projected onto
            # the TriggerRow off its sessions JOIN. None for normal sessions
            # (root run); for a workflow-child owner this closes the
            # depth-laundering bypass (a past-fire_at one-shot is create_run
            # with a 0s delay).
            parent_run_id = trigger.session_parent_run_id
        composed = compose_workflow_run_input(
            trigger_id=trigger.id,
            trigger_name=trigger.name,
            # The envelope's source derives from the FIRE'S ORIGIN, like the
            # lifecycle arm — the reloaded row's source is user-mutable in the
            # match→fire window, and the delivered context must agree with the
            # carrier audit row about what kind of fire this was.
            source="run_completion" if event is not None else trigger.source,
            fired_at=started_at,
            input_template=action.input_template,
            completed_run=completed_run,
            completed_error=completed_error,
        )
        run = await wf_run_service.create_run(
            pool,
            account_id=trigger.account_id,
            workflow_id=action.workflow_id,
            environment_id=trigger.environment_id,
            input=composed,
            vault_ids=action.vault_ids,
            launcher_session_id=trigger.owner_session_id,
            parent_run_id=parent_run_id,
            expected_version=action.workflow_version,
        )
        log.info(
            "trigger.fired",
            trigger_id=trigger.id,
            session_id=trigger.owner_session_id,
            name=trigger.name,
            kind="workflow",
            status="ok",
            run_id=run.id,
        )
        return "ok", None, run.id
    except Exception as e:
        log.exception(
            "trigger.workflow_error",
            trigger_id=trigger.id,
            session_id=trigger.owner_session_id,
            name=trigger.name,
        )
        return "error", f"run launch failed: {type(e).__name__}: {e!s:.200}", None


async def _surface_failure(session_id: str, account_id: str, content: str) -> None:
    """Append a synthetic user message + defer a wake (best-effort).

    Shared by the one-shot failure paths and the auto-disable path so all
    surface a user-visible event instead of failing silently — the model sees
    a deterministic failure event instead of unexplained silence.
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


# Sweep thresholds — the single source (the query functions take them as
# required kwargs, so a default can't silently drift from the sweep).
_PENDING_REDEFER_SECONDS = 60.0  # a lost post-commit defer re-defers after this
_STUCK_RUNNING_SECONDS = 7200.0  # crashed mid-fire: surfaced, deliberately never retried
# Maintenance (stuck-running count + retention prune) runs hourly, not per
# 30s tick: the prune enforces a DAYS-scale window (2,880 DELETEs/day bought
# nothing), and a stuck row warned every tick is ~86k log lines per incident.
_MAINTENANCE_INTERVAL_SECONDS = 3600.0
_last_maintenance: float = 0.0


async def sweep_trigger_fires(pool: asyncpg.Pool[Any]) -> None:
    """Periodic-sweep arm for run_completion fire intents + audit retention.

    Re-defers ``pending`` carrier rows whose post-commit defer was lost (a
    worker crash between the completion commit and ``defer_async`` — the
    durable row is the recovery anchor; the ``pending → running`` claim makes
    a re-defer racing a live job safe). ``running`` rows past the stale
    threshold are a worker crash mid-fire: counted and warned, deliberately
    NEVER retried (re-firing could double-launch a run — at-most-once after
    claim). The pending re-defer runs every tick (it IS the recovery
    latency); the maintenance pair (stuck count + prune) is gated hourly.
    """
    global _last_maintenance
    now = time.monotonic()
    maintain = now - _last_maintenance >= _MAINTENANCE_INTERVAL_SECONDS
    if maintain:
        _last_maintenance = now

    async with pool.acquire() as conn:
        pending = await queries.list_pending_trigger_run_refs(
            conn, older_than_seconds=_PENDING_REDEFER_SECONDS
        )
        if maintain:
            stuck = await queries.count_stuck_running_trigger_runs(
                conn, older_than_seconds=_STUCK_RUNNING_SECONDS
            )
            pruned = await queries.prune_trigger_runs(
                conn, retention_days=get_settings().trigger_runs_retention_days
            )
    for ref in pending:
        await defer_trigger_fire(ref.trigger_id, ref.trigger_run_id)
    if pending:
        log.info("sweep.trigger_fires_redeferred", count=len(pending))
    if maintain:
        if stuck:
            log.warning("trigger.fires_stuck_running", count=stuck)
        if pruned:
            log.info("sweep.trigger_runs_pruned", count=pruned)
