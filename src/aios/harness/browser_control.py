"""Worker-side browser control plane: call executor + the plane reaper.

The consumer half of :mod:`aios.services.browser_calls` (jarbot#106 §5.7).
The worker's browser-call listener hands each ``call_id`` to
:func:`execute_browser_call`, which loads the row, runs the op — through
:func:`aios.sandbox.browser.driver_call` where the driver is involved — and
resolves the row + result NOTIFY. Every op is safe to re-execute (the
listener's redrive re-runs pending rows after a reconnect or worker
restart): ``takeover_open`` is idempotent per grant id BY DRIVER CONTRACT,
``takeover_close``/grant updates are conditional on ``status='open'``, and
``status``/``revoke_site``/``clear_state`` are naturally re-runnable.

:func:`browser_reaper_tick` is the plane's periodic upkeep, one tick per
``sandbox_browser_reaper_interval_seconds``: grant-TTL expiry (with the
driver handback and the model-visible lifecycle notice), the
container-keepalive bump for fresh open grants, ``browser_calls`` row
retention, and the plane byte quotas — shots/frames/downloads oldest-first,
NEVER the profile (real logins; only explicit clear-state deletes it), and
the input spool truncated once no grant is open.
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import asyncpg

from aios.config import get_settings
from aios.db import queries
from aios.logging import get_logger
from aios.sandbox.browser import EXEC_KILL_MARGIN_S, BrowserUnavailableError, driver_call
from aios.sandbox.browser_protocol import BrowserRequest, BrowserResponse
from aios.sandbox.spec import BrowserImageUnconfiguredError
from aios.sandbox.volumes import (
    BROWSER_PLANE_SUBDIRS,
    browser_plane_dir,
    browser_plane_root,
    ensure_browser_plane_dir,
)
from aios.services import sessions as sessions_service

if TYPE_CHECKING:
    from aios.sandbox.registry import SandboxRegistry

log = get_logger("aios.harness.browser_control")

# Content the plane reaper never touches: the profile holds real logins
# (non-reconstructible; deleted only by explicit clear-state), and the input
# spool has its own no-open-grant truncation rule.
_QUOTA_SUBDIRS = ("shots", "frames", "downloads")


async def execute_browser_call(
    registry: SandboxRegistry, pool: asyncpg.Pool[asyncpg.Record], call_id: str
) -> None:
    """Load, execute, and resolve one browser control call.

    Never raises on op-level failure — the row resolves ``is_error`` with a
    ``{code, message}`` result the API maps to its HTTP currency. A missing
    or already-resolved row is a no-op (redrive racing the live dispatch).
    """
    async with pool.acquire() as conn:
        row = await queries.get_browser_call_unscoped(conn, call_id)
    if row is None or row["status"] != "pending":
        return
    if row["expires_at"] <= datetime.now(UTC):
        # The submitter's wait already timed out; execution now would act
        # with nobody listening. Resolve as expired instead of running.
        await _resolve(pool, call_id, {"code": "expired", "message": "call expired"}, True)
        return

    account_id = str(row["account_id"])
    params: dict[str, Any] = row["params"] or {}
    try:
        result = await _dispatch(registry, pool, str(row["method"]), account_id, params)
        await _resolve(pool, call_id, result, False)
    except BrowserImageUnconfiguredError as err:
        await _resolve(pool, call_id, {"code": "browser_unconfigured", "message": str(err)}, True)
    except BrowserUnavailableError as err:
        await _resolve(pool, call_id, {"code": "browser_unavailable", "message": str(err)}, True)
    except _OpError as err:
        await _resolve(pool, call_id, {"code": err.code, "message": err.message}, True)
    except Exception as err:
        # The plane's analog of tool-always-appends-result (invariant #4): an
        # unexpected executor fault must still resolve the row, or the blocked
        # submitter sees a 504 masking a deterministic bug and the redrive
        # re-crashes on the same row forever.
        log.exception("browser_call.execute_failed", call_id=call_id)
        await _resolve(pool, call_id, {"code": "internal", "message": str(err)}, True)


class _OpError(Exception):
    """An op-level failure the row resolves as ``is_error`` (driver ok:false,
    an open-grant conflict, ...)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


async def _resolve(
    pool: asyncpg.Pool[asyncpg.Record], call_id: str, result: Any, is_error: bool
) -> None:
    async with pool.acquire() as conn:
        # Autocommit: the result NOTIFY must fire only after the UPDATE is
        # visible to the waiter's re-fetch. Conditional resolve makes the
        # redrive race a silent no-op (no double NOTIFY).
        moved = await queries.resolve_browser_call(
            conn, call_id=call_id, result=result, is_error=is_error
        )
        if moved:
            await queries.notify_browser_call_result(conn, call_id=call_id)


async def _driver(
    registry: SandboxRegistry,
    account_id: str,
    op: str,
    args: dict[str, Any],
    *,
    timeout_s: int,
    session_id: str | None = None,
) -> BrowserResponse:
    response = await driver_call(
        registry,
        account_id,
        BrowserRequest(op=op, args=args, timeout_ms=timeout_s * 1000, session_id=session_id),
        timeout_s=timeout_s + EXEC_KILL_MARGIN_S,
    )
    if not response.ok:
        code = response.error.code if response.error else "internal"
        message = response.error.message if response.error else "unknown driver failure"
        raise _OpError(code, message)
    return response


async def _dispatch(
    registry: SandboxRegistry,
    pool: asyncpg.Pool[asyncpg.Record],
    method: str,
    account_id: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    settings = get_settings()
    action_timeout = settings.sandbox_browser_action_timeout_seconds

    if method == "open":
        # The grant_id is minted at submit time by the API (open_takeover) and
        # threaded through params — read it fail-hard, NO fallback mint. A
        # per-execution mint would make a lost-NOTIFY REDRIVE issue a SECOND,
        # DISTINCT takeover_open (fresh id) that the driver's per-grant-id
        # idempotency can't collapse — orphaning the first takeover.
        grant_id = str(params["grant_id"])
        session_id = str(params["session_id"])
        reason = str(params.get("reason") or "")
        ttl_seconds = int(settings.sandbox_browser_grant_ttl_seconds)
        response = await _driver(
            registry,
            account_id,
            "takeover_open",
            {"grant_id": grant_id, "reason": reason},
            # Blocks through the drain of an in-flight action — its own budget.
            timeout_s=settings.sandbox_browser_takeover_open_timeout_seconds,
            # The driver takes over THIS session's page (jarbot#106 §5.6);
            # threaded via the request's session_id field, not args.
            session_id=session_id,
        )
        async with pool.acquire() as conn:
            try:
                # AFTER the driver ack — the driver is the enforcement
                # authority; the row is the control-plane record.
                await queries.insert_browser_grant(
                    conn,
                    grant_id=grant_id,
                    account_id=account_id,
                    session_id=session_id,
                    reason=reason,
                    boot=response.boot,
                    epoch=response.epoch,
                    target=response.data.get("target") or {},
                    ttl_seconds=ttl_seconds,
                )
            except asyncpg.UniqueViolationError as err:
                raise _OpError("takeover_in_progress", "a takeover is already in progress") from err
        return {
            "grant_id": grant_id,
            "target": response.data.get("target") or {},
            "boot": response.boot,
            "epoch": response.epoch,
            "ttl_seconds": ttl_seconds,
        }

    if method == "close":
        grant_id = str(params["grant_id"])
        outcome = str(params.get("outcome") or "done")
        # Load scoped FIRST: ``close_browser_grant`` is unscoped by contract
        # ("callers act from rows they already loaded"), and grant_id here is
        # caller-supplied — a cross-tenant or unknown id must fail, never
        # touch another account's row.
        async with pool.acquire() as conn:
            grant = await queries.get_browser_grant(conn, grant_id, account_id=account_id)
        if grant is None:
            raise _OpError("unknown_grant", f"no takeover grant {grant_id}")
        return await _finalize_takeover(registry, pool, grant, status="closed", outcome=outcome)

    if method == "status":
        if registry.peek(account_id) is None:
            # Never provision for a status read — a cold computer is simply
            # not running.
            return {"running": False}
        response = await _driver(registry, account_id, "status", {}, timeout_s=action_timeout)
        return {
            "running": True,
            "url": response.url,
            "title": response.title,
            "boot": response.boot,
            "epoch": response.epoch,
            **response.data,
        }

    if method == "revoke_site":
        host = str(params["host"])
        response = await _driver(
            registry, account_id, "revoke_site", {"host": host}, timeout_s=action_timeout
        )
        return dict(response.data)

    if method == "clear_state":
        async with pool.acquire() as conn:
            open_grant = await queries.get_open_browser_grant_for_account(conn, account_id)
        if open_grant is not None:
            raise _OpError("takeover_open", "a takeover is in progress; close it first")
        # No driver op: destroy the container, then recreate the plane subdirs
        # empty — the one sanctioned way login state is deleted. Hold the
        # account's owner lock across destroy + wipe + recreate so a concurrent
        # (re)provision can't cold-mount the plane mid-wipe and have its profile
        # deleted out from under a live container. release_browser is lock-free,
        # so this cannot self-deadlock.
        async with registry.owner_lock(account_id):
            await registry.release_browser(account_id)
            plane = browser_plane_dir(account_id)
            for sub in BROWSER_PLANE_SUBDIRS:
                target = plane / sub
                if target.exists():
                    # NO ignore_errors: a partial wipe of login state must
                    # surface (execute_browser_call -> is_error -> API 5xx),
                    # never a false 204-success while cookies persist on disk.
                    shutil.rmtree(target)
            ensure_browser_plane_dir(account_id)
        session_id_param = params.get("session_id")
        if session_id_param:
            await _append_lifecycle(
                pool,
                str(session_id_param),
                account_id,
                {"event": "browser_state_lost", "cause": "cleared"},
            )
        return {}

    raise _OpError("unknown_method", f"unknown browser call method {method!r}")


async def _finalize_takeover(
    registry: SandboxRegistry,
    pool: asyncpg.Pool[asyncpg.Record],
    grant: dict[str, Any],
    *,
    status: str,
    outcome: str,
) -> dict[str, Any]:
    """Move an open grant to its terminal state, CLAIM-FIRST.

    The conditional ``open -> terminal`` UPDATE is the serialization point:
    only the claim-winner then runs the driver handback, so an explicit close
    racing the TTL reaper can never issue two ``takeover_close`` calls — and,
    crucially, the losing call (which finds no active takeover and returns an
    error-handback) can never overwrite the winner's real handback (the
    signed-in-hosts delta). Idempotent under redrive: a re-run finds the row
    already terminal (``moved`` False) and no-ops.
    """
    grant_id = str(grant["id"])
    account_id = str(grant["account_id"])
    async with pool.acquire() as conn:
        moved = await queries.close_browser_grant(
            conn, grant_id=grant_id, status=status, outcome=outcome, handback=None
        )
    if not moved:
        # A concurrent close/expiry (or a redrive) already finalized it; its
        # handback is on the row. Do NOT touch the driver or re-notify.
        return {"handback": None}
    handback = await _close_takeover(registry, account_id, grant_id, outcome=outcome)
    if handback is not None:
        async with pool.acquire() as conn:
            await queries.set_browser_grant_handback(conn, grant_id=grant_id, handback=handback)
    log.info(
        "browser.grant_expired" if status == "expired" else "browser.grant_closed",
        account_id=account_id,
        grant_id=grant_id,
    )
    await _append_takeover_ended(pool, grant, outcome=outcome)
    return {"handback": handback}


async def _close_takeover(
    registry: SandboxRegistry, account_id: str, grant_id: str, *, outcome: str
) -> dict[str, Any] | None:
    """Capture the driver handback for a takeover being finalized.

    Returns ``None`` when there is no live container to hand back from — and
    deliberately does NOT cold-provision one: a fresh container has no takeover
    (so ``takeover_close`` would fail anyway), and a cold provision under
    snapshot-pool pressure would fault, starving the reaper tick. Returns a
    handback dict on success, or a dict carrying an explicit ``error`` when the
    driver was REACHED but could not capture the handback — never a silent
    empty payload reported as a clean close (the signed-in-hosts delta is the
    whole point of an auth takeover).
    """
    if registry.peek(account_id) is None:
        return None
    settings = get_settings()
    try:
        response = await _driver(
            registry,
            account_id,
            "takeover_close",
            {"grant_id": grant_id, "outcome": outcome},
            timeout_s=settings.sandbox_browser_action_timeout_seconds,
        )
    except (BrowserUnavailableError, BrowserImageUnconfiguredError) as err:
        # The container went away between the peek and the exec — genuinely no
        # handback to capture.
        log.warning(
            "browser.takeover_close_handback_unavailable",
            account_id=account_id,
            grant_id=grant_id,
            error=str(err),
        )
        return None
    except _OpError as err:
        # The driver was reached and reported it could not capture the handback
        # (snapshot / profile / signed-in extraction failed). Record the
        # failure honestly rather than a silent empty "done".
        log.warning(
            "browser.takeover_close_handback_failed",
            account_id=account_id,
            grant_id=grant_id,
            code=err.code,
            message=err.message,
        )
        return {"error": f"handback_failed: {err.code}: {err.message}"}
    return {
        "snapshot": response.snapshot,
        "shot_path": response.shot_path,
        "signed_in_hosts": response.data.get("signed_in_hosts") or [],
        "url": response.url,
    }


async def _append_takeover_ended(
    pool: asyncpg.Pool[asyncpg.Record], grant: dict[str, Any], *, outcome: str
) -> None:
    await _append_lifecycle(
        pool,
        str(grant["session_id"]),
        str(grant["account_id"]),
        {
            "event": "browser_takeover_ended",
            "grant_id": str(grant["id"]),
            "outcome": outcome,
            "url": (grant.get("target") or {}).get("url"),
        },
    )


async def _append_lifecycle(
    pool: asyncpg.Pool[asyncpg.Record],
    session_id: str,
    account_id: str,
    data: dict[str, Any],
) -> None:
    """Append a model-visible, structurally non-waking lifecycle notice.

    Best-effort by design: a lifecycle notice is non-waking (the model reads it
    at its next genuine wake, never sooner), so a failed append must not fail —
    nor, in the reaper, ABORT — the control op that produced it. The bound is
    that on the rare append failure the requesting session simply learns of the
    handback/state-loss at its next stimulus rather than proactively; it is not
    lost from any waking path. (Atomic co-write with the grant close was
    considered and deferred: it would couple the lifecycle append into the
    grant-close transaction for a non-waking notice — not worth the coupling.)
    """
    try:
        await sessions_service.append_event(
            pool, session_id, "lifecycle", data, account_id=account_id
        )
    except Exception as err:
        log.warning(
            "browser.lifecycle_append_failed",
            session_id=session_id,
            lifecycle_event=data.get("event"),
            error=str(err),
        )


# ── the reaper tick ───────────────────────────────────────────────────────


async def browser_reaper_tick(
    registry: SandboxRegistry, pool: asyncpg.Pool[asyncpg.Record]
) -> None:
    """One pass of the browser plane's periodic upkeep.

    Each step is isolated in its own try/except: a failure in one (a wedged
    grant, a bad file) is logged and the remaining steps still run. The quota
    sweep (step 4) is the backstop against unbounded plane disk growth, so it
    must NEVER be starved by an earlier step failing — precisely the state
    (host pressure + a dead-container grant) where the sweep matters most.
    """
    settings = get_settings()

    # 1. Expire stale open grants: claim-first terminal move, driver handback,
    #    model-visible notice to the requesting session. Per-grant isolation so
    #    one wedged grant cannot starve the others or the later steps.
    try:
        async with pool.acquire() as conn:
            stale = await queries.list_stale_open_browser_grants(conn)
        for grant in stale:
            try:
                await _finalize_takeover(registry, pool, grant, status="expired", outcome="expired")
            except Exception:
                log.exception("browser.grant_expiry_failed", grant_id=str(grant["id"]))
    except Exception:
        log.exception("browser.reaper_step_failed", step="expire")

    # 2. Container keepalive: a fresh-heartbeat open grant means a human is
    #    driving — the idle reaper must not take Chromium out from under
    #    them (touch never provisions).
    try:
        async with pool.acquire() as conn:
            fresh_accounts = await queries.list_fresh_open_browser_grant_accounts(conn)
        for account_id in fresh_accounts:
            registry.touch_browser(account_id)
    except Exception:
        log.exception("browser.reaper_step_failed", step="keepalive")

    # 3. Call-row retention (the sweep pending_management_calls never got).
    try:
        cutoff = datetime.now(UTC) - timedelta(
            seconds=settings.sandbox_browser_calls_retention_seconds
        )
        async with pool.acquire() as conn:
            reaped = await queries.delete_finished_browser_calls_before(conn, cutoff)
        if reaped:
            log.info("browser.calls_reaped", count=reaped)
    except Exception:
        log.exception("browser.reaper_step_failed", step="retention")

    # 4. Plane byte quotas. Spool preservation keys on "an open grant exists
    #    RIGHT NOW" — not the fresh-heartbeat set (a grant mid-lapse is still
    #    open until closed, and its human's input must survive until then).
    try:
        async with pool.acquire() as conn:
            open_accounts = await queries.list_open_browser_grant_accounts(conn)
        _enforce_plane_quotas(set(open_accounts))
    except Exception:
        log.exception("browser.reaper_step_failed", step="quotas")


def _enforce_plane_quotas(accounts_with_open_grants: set[str]) -> None:
    """Bound every account plane's shots/frames/downloads; truncate idle spools.

    Oldest-first deletion within each capped subdir; the profile is never
    touched. Purely local filesystem work — errors are logged, never raised
    (the reaper's other duties must not stop for one bad file).
    """
    settings = get_settings()
    caps = {
        "shots": settings.sandbox_browser_shots_max_bytes,
        "frames": settings.sandbox_browser_frames_max_bytes,
        "downloads": settings.sandbox_browser_downloads_max_bytes,
    }
    plane_root = browser_plane_root()
    if not plane_root.exists():
        return
    for account_dir in plane_root.iterdir():
        if not account_dir.is_dir() or account_dir.is_symlink():
            continue
        for sub in _QUOTA_SUBDIRS:
            try:
                _trim_dir_to_cap(account_dir / sub, caps[sub])
            except OSError as err:
                log.warning(
                    "browser.plane_quota_sweep_failed",
                    path=str(account_dir / sub),
                    error=str(err),
                )
        spool = account_dir / "input" / "spool.jsonl"
        if account_dir.name not in accounts_with_open_grants and spool.exists():
            try:
                spool.unlink()
            except OSError as err:
                log.warning("browser.spool_truncate_failed", path=str(spool), error=str(err))


def _trim_dir_to_cap(directory: Path, cap_bytes: int) -> None:
    if not directory.is_dir():
        return
    entries = [(f, f.stat()) for f in directory.iterdir() if f.is_file()]
    total = sum(st.st_size for _, st in entries)
    if total <= cap_bytes:
        return
    for f, st in sorted(entries, key=lambda pair: pair[1].st_mtime):
        f.unlink(missing_ok=True)
        total -= st.st_size
        log.info("browser.plane_file_reaped", path=str(f), size=st.st_size)
        if total <= cap_bytes:
            return
