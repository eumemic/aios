"""The product-facing browser takeover surface (jarbot#106 §5.7).

jarbot (or any operator client) drives a human takeover of an account's
shared computer through these routes: open a grant, stream frames, post the
human's input, heartbeat while the viewer is attached, close with a handback.
Plus the account-computer operations: status, per-site sign-out, clear-state.

Trust boundary: every grant-scoped route resolves the grant with the
standard account-scoped query FIRST (cross-tenant ids surface as 404) —
before any listener, stream, or filesystem access. The control ops ride the
worker-consumed RPC (:func:`aios.services.browser_calls.submit_browser_call`);
the frame stream tails the shared-filesystem ring and the input POST appends
to the shared-filesystem spool, so neither touches the worker.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Response
from sse_starlette import EventSourceResponse, ServerSentEvent

from aios.api.deps import AccountIdDep, DbUrlDep, PoolDep
from aios.api.sse import make_sse_response
from aios.config import get_settings
from aios.db import queries
from aios.errors import (
    AiosError,
    ConflictError,
    NotFoundError,
    PayloadTooLargeError,
    ServiceUnavailableError,
)
from aios.ids import BROWSER_GRANT, make_id
from aios.logging import get_logger
from aios.models.browser import (
    BrowserStatusResponse,
    BrowserTakeoverStatus,
    HandbackPayload,
    InputBatch,
    TakeoverCloseRequest,
    TakeoverCloseResponse,
    TakeoverOpenRequest,
    TakeoverOpenResponse,
)
from aios.sandbox.browser_protocol import TAKEOVER_HEARTBEAT_MARKER
from aios.sandbox.volumes import browser_plane_dir
from aios.services import sessions as sessions_service
from aios.services.browser_calls import submit_browser_call

router = APIRouter(prefix="/v1/browser", tags=["browser"])
log = get_logger("aios.api.browser")

_FRAME_POLL_SECONDS = 0.2
_GRANT_RECHECK_SECONDS = 2.0


def _control_timeout() -> float:
    return float(get_settings().sandbox_browser_call_timeout_seconds)


async def _call(
    db_url: str,
    pool: Any,
    account_id: str,
    method: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Submit a control op and translate its error currency to HTTP.

    ``browser_unavailable``/``browser_unconfigured`` → 503; ``internal`` (the
    executor's generic backstop — a deterministic worker-side bug) → 500;
    everything else the executor reported as an error → 409 (an op-level refusal
    the caller can act on: ``takeover_in_progress``, ``unknown_grant``, a driver
    error code, ...). A raw success result passes through.
    """
    result, is_error = await submit_browser_call(
        db_url,
        pool,
        account_id=account_id,
        method=method,
        params=params,
        timeout_s=_control_timeout(),
    )
    if is_error:
        code = (result or {}).get("code", "internal")
        message = (result or {}).get("message", "browser control op failed")
        if code in ("browser_unavailable", "browser_unconfigured"):
            raise ServiceUnavailableError(message, detail={"code": code})
        if code == "internal":
            # A worker-side fault, not a caller-actionable refusal — 500, not a
            # 409 that reads as "your request conflicts with state".
            raise AiosError(message, detail={"code": code})
        raise ConflictError(message, detail={"code": code})
    return result or {}


async def _require_grant(pool: Any, grant_id: str, account_id: str) -> dict[str, Any]:
    async with pool.acquire() as conn:
        grant = await queries.get_browser_grant(conn, grant_id, account_id=account_id)
    if grant is None:
        raise NotFoundError(f"takeover grant {grant_id} not found", detail={"id": grant_id})
    return grant


@router.post("/takeover")
async def open_takeover(
    body: TakeoverOpenRequest,
    db_url: DbUrlDep,
    pool: PoolDep,
    account_id: AccountIdDep,
) -> TakeoverOpenResponse:
    """Open a takeover of the account's computer for one session's page.

    409 if a takeover is already in progress (the one-open-per-account
    invariant); 503 if the computer is unavailable.
    """
    # Scope-check the requesting session before opening anything.
    await sessions_service.get_session_basic(pool, body.session_id, account_id=account_id)
    # Mint the grant_id HERE, caller-side, and thread it through: the executor
    # reads it fail-hard, so a lost-NOTIFY redrive re-presents the SAME id and
    # the driver's per-grant-id idempotency collapses the retry rather than
    # opening a second, orphaned takeover.
    grant_id = make_id(BROWSER_GRANT)
    result = await _call(
        db_url,
        pool,
        account_id,
        "open",
        {"session_id": body.session_id, "reason": body.reason, "grant_id": grant_id},
    )
    return TakeoverOpenResponse(**result)


@router.post("/takeover/{grant_id}/heartbeat", status_code=204)
async def heartbeat_takeover(
    grant_id: str,
    pool: PoolDep,
    account_id: AccountIdDep,
) -> Response:
    """Bump the grant's heartbeat — the viewer calls this while it holds the
    frame stream. 404 if unknown, 409 if the grant is no longer open."""
    async with pool.acquire() as conn:
        ok = await queries.touch_browser_grant_heartbeat(conn, grant_id, account_id=account_id)
        if not ok:
            grant = await queries.get_browser_grant(conn, grant_id, account_id=account_id)
    if not ok:
        if grant is None:
            raise NotFoundError(f"takeover grant {grant_id} not found", detail={"id": grant_id})
        raise ConflictError(
            f"takeover grant {grant_id} is not open", detail={"status": grant["status"]}
        )
    # Touch the plane liveness marker the driver's idle watchdog folds into its
    # clock, so a human who is watching (heartbeating) but not typing does not
    # trip the driver's idle auto-close. The DB heartbeat above is the reaper's
    # authoritative signal; this marker is only the driver's watch-detection
    # input, so a write failure degrades (the driver may idle-close a passive
    # viewer) but must not fail the heartbeat — log it as the plane bug it is.
    marker = browser_plane_dir(account_id) / TAKEOVER_HEARTBEAT_MARKER
    try:
        marker.touch()
    except OSError as exc:
        log.warning("browser.heartbeat_marker_touch_failed", grant_id=grant_id, error=str(exc))
    return Response(status_code=204)


@router.post("/takeover/{grant_id}/input", status_code=204)
async def post_input(
    grant_id: str,
    body: InputBatch,
    pool: PoolDep,
    account_id: AccountIdDep,
) -> Response:
    """Append one epoch-stamped input batch to the shared-filesystem spool.

    No worker involvement. The check-then-append race (grant closes between
    the epoch check and the write) is harmless — the driver drops stale-epoch
    lines, being the enforcement authority. 409 on a closed grant or stale
    epoch; 413 when the spool would exceed its byte cap.
    """
    grant = await _require_grant(pool, grant_id, account_id)
    if grant["status"] != "open":
        raise ConflictError(
            f"takeover grant {grant_id} is not open", detail={"status": grant["status"]}
        )
    if body.epoch != grant["epoch"]:
        raise ConflictError(
            "input epoch does not match the open grant",
            detail={"code": "stale_epoch", "expected": grant["epoch"], "got": body.epoch},
        )

    plane = browser_plane_dir(account_id)
    spool = plane / "input" / "spool.jsonl"
    line = (
        json.dumps(
            {
                "grant_id": grant_id,
                "epoch": body.epoch,
                "seq": body.seq,
                "ts_ms": int(time.time() * 1000),
                "events": [e.model_dump(exclude_none=True) for e in body.events],
            }
        )
        + "\n"
    ).encode("utf-8")

    cap = get_settings().sandbox_browser_input_spool_max_bytes
    existing = spool.stat().st_size if spool.exists() else 0
    # Soft cap: the check-then-write is per-request, so N concurrent posts can
    # each pass here and overshoot by up to (N-1) lines. That is bounded (one
    # viewer per grant is the norm; a line is a few hundred bytes) and the
    # reaper truncates the spool once the grant closes — a hard atomic cap isn't
    # worth an flock on the hot input path.
    if existing + len(line) > cap:
        raise PayloadTooLargeError(
            "input spool is full", detail={"cap_bytes": cap, "size_bytes": existing}
        )
    # O_APPEND makes each write atomic against concurrent appenders (the kernel
    # holds the inode offset lock), so lines never interleave — but os.write can
    # still return a SHORT count, which would leave a truncated, unparseable
    # JSONL line. Loop until the whole line lands, and fail hard if it can't.
    fd = os.open(spool, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        view = memoryview(line)
        written = 0
        while written < len(view):
            n = os.write(fd, view[written:])
            if n <= 0:
                raise OSError("short write to input spool")
            written += n
    finally:
        os.close(fd)
    return Response(status_code=204)


@router.get(
    "/takeover/{grant_id}/frames",
    openapi_extra={"x-codegen": {"targets": []}},
)
async def stream_frames(
    grant_id: str,
    pool: PoolDep,
    account_id: AccountIdDep,
) -> EventSourceResponse:
    """Stream the takeover screencast as SSE ``frame`` events, ending on close.

    Novel among aios SSE routes: it tails a shared-filesystem ring, not a
    LISTEN channel (the driver has no route to Postgres). The frames dir is
    derived server-side from the scoped grant's account — the client supplies
    only ``grant_id`` — so path containment holds by construction.
    """
    grant = await _require_grant(pool, grant_id, account_id)  # scope + 404 gate
    if grant["status"] != "open":
        # Don't open a screencast on an already-terminal grant — mirror
        # post_input rather than stream one stale frame before the recheck ends
        # it.
        raise ConflictError(
            f"takeover grant {grant_id} is not open", detail={"status": grant["status"]}
        )
    plane = browser_plane_dir(account_id)
    plane_root = plane.resolve()
    frames_dir = plane / "frames"
    if not plane.exists():
        # The plane dir is created at provision; its absence while a grant is
        # open is a deployment/mount fault, not a quiet driver — surface it
        # loudly rather than streaming silence forever.
        raise ServiceUnavailableError(
            "browser plane directory is missing", detail={"grant_id": grant_id}
        )

    async def _frames() -> Any:
        last_seq = -1
        last_boot: str | None = None
        last_grant_check = 0.0
        while True:
            manifest = _read_manifest(frames_dir, plane_root, account_id)
            if manifest is not None:
                boot = manifest.get("boot")
                if last_boot is not None and boot != last_boot:
                    # The driver rebooted mid-takeover (frame seq resets to a low
                    # value, so a seq-only test would freeze the screencast
                    # forever). End the stream — the viewer pins boot and must
                    # re-attach against the new one.
                    yield ServerSentEvent(
                        data=json.dumps({"outcome": "browser_restarted"}), event="end"
                    )
                    return
                if manifest["seq"] > last_seq:
                    frame = _load_frame(frames_dir, manifest, plane_root, account_id)
                    if frame is not None:
                        last_seq = manifest["seq"]
                        last_boot = boot
                        yield ServerSentEvent(data=json.dumps(frame), event="frame")

            now = time.monotonic()
            if now - last_grant_check >= _GRANT_RECHECK_SECONDS:
                last_grant_check = now
                async with pool.acquire() as conn:
                    fresh = await queries.get_browser_grant(conn, grant_id, account_id=account_id)
                if fresh is None or fresh["status"] != "open":
                    outcome = (fresh or {}).get("outcome") or "closed"
                    yield ServerSentEvent(data=json.dumps({"outcome": outcome}), event="end")
                    return
            await asyncio.sleep(_FRAME_POLL_SECONDS)

    # No LISTEN subscription — a no-op terminate satisfies make_sse_response's
    # un-invoked-generator cleanup (the poll loop owns no external resource).
    return make_sse_response(lambda: None, _frames())


def _frames_dir_in_plane(frames_dir: Path, plane_root: Path, account_id: str) -> Path | None:
    """Resolve ``frames_dir`` and confirm it is still inside the account's plane
    ROOT (the bind-mount SOURCE, which a compromised container cannot symlink
    away). Anchoring on the plane root — not on ``frames_dir`` itself — is what
    defeats a hostile ``frames`` symlink pointing at ANOTHER account's plane:
    resolving both sides of a self-referential symlink check would pass it.
    Returns the resolved dir, or ``None`` if it escaped (logged)."""
    resolved = frames_dir.resolve()
    if not resolved.is_relative_to(plane_root):
        log.warning("browser.frames_dir_escape", account_id=account_id, resolved=str(resolved))
        return None
    return resolved


def _read_manifest(frames_dir: Path, plane_root: Path, account_id: str) -> dict[str, Any] | None:
    if _frames_dir_in_plane(frames_dir, plane_root, account_id) is None:
        return None
    manifest_path = frames_dir / "manifest.json"
    try:
        raw = manifest_path.read_bytes()
    except OSError:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    # ``seq`` drives a ``> last_seq`` comparison against an int; a manifest with
    # a null/placeholder or non-int seq is "no frame yet", not a TypeError that
    # tears down the stream.
    if not isinstance(data.get("seq"), int) or isinstance(data.get("seq"), bool):
        return None
    return data


def _load_frame(
    frames_dir: Path, manifest: dict[str, Any], plane_root: Path, account_id: str
) -> dict[str, Any] | None:
    """Build a frame SSE payload from the manifest, or ``None`` if unreadable.

    Forwards the §5.6 trusted-chrome envelope (minus the account, which never
    crosses to the client) with the JPEG inlined. Containment is TWO checks,
    both anchored so a compromised container cannot exfiltrate cross-tenant:
    the frames dir must resolve inside the account's plane root (blocks a
    symlinked ``frames`` -> another account's plane), and the manifest ``file``
    must resolve inside that frames dir (blocks a ``../`` file ref).
    """
    resolved_frames = _frames_dir_in_plane(frames_dir, plane_root, account_id)
    if resolved_frames is None:
        return None
    file_ref = manifest.get("file")
    if not isinstance(file_ref, str):
        return None
    frame_path = (frames_dir / file_ref).resolve()
    if not frame_path.is_relative_to(resolved_frames):
        log.warning("browser.frame_path_escape", account_id=account_id, file=file_ref)
        return None
    try:
        jpeg = frame_path.read_bytes()
    except OSError:
        return None
    return {
        "seq": manifest.get("seq"),
        "ts_ms": manifest.get("ts_ms"),
        "epoch": manifest.get("epoch"),
        "boot": manifest.get("boot"),
        "origin": manifest.get("origin"),
        "security": manifest.get("security"),
        "w": manifest.get("w"),
        "h": manifest.get("h"),
        "jpeg_b64": base64.b64encode(jpeg).decode("ascii"),
    }


@router.delete("/takeover/{grant_id}")
async def close_takeover(
    grant_id: str,
    body: TakeoverCloseRequest,
    db_url: DbUrlDep,
    pool: PoolDep,
    account_id: AccountIdDep,
) -> TakeoverCloseResponse:
    """Close a takeover; return the handback (post-human snapshot + inlined
    screenshot + signed-in delta). A browser-dead close still closes → null
    handback fields."""
    await _require_grant(pool, grant_id, account_id)
    result = await _call(
        db_url, pool, account_id, "close", {"grant_id": grant_id, "outcome": body.outcome}
    )
    handback = _handback_payload(result.get("handback"), account_id)
    return TakeoverCloseResponse(handback=handback)


def _handback_payload(raw: dict[str, Any] | None, account_id: str) -> HandbackPayload:
    if not raw:
        return HandbackPayload()
    screenshot_data_url = None
    shot_path = raw.get("shot_path")
    if isinstance(shot_path, str):
        plane = browser_plane_dir(account_id)
        resolved = (plane / shot_path).resolve()
        if resolved.is_relative_to(plane.resolve()):
            with contextlib.suppress(OSError):
                screenshot_data_url = "data:image/png;base64," + base64.b64encode(
                    resolved.read_bytes()
                ).decode("ascii")
    return HandbackPayload(
        snapshot=raw.get("snapshot"),
        screenshot_data_url=screenshot_data_url,
        signed_in_hosts=raw.get("signed_in_hosts") or [],
        url=raw.get("url"),
    )


@router.get("/status")
async def browser_status(
    db_url: DbUrlDep,
    pool: PoolDep,
    account_id: AccountIdDep,
) -> BrowserStatusResponse:
    """The account computer's state — never provisions it."""
    result = await _call(db_url, pool, account_id, "status", {})
    async with pool.acquire() as conn:
        open_grant = await queries.get_open_browser_grant_for_account(conn, account_id)
    takeover = None
    if open_grant is not None:
        takeover = BrowserTakeoverStatus(
            grant_id=open_grant["id"],
            session_id=open_grant["session_id"],
            reason=open_grant["reason"],
            epoch=open_grant["epoch"],
            boot=open_grant["boot"],
            created_at=open_grant["created_at"].isoformat(),
        )
    return BrowserStatusResponse(
        running=bool(result.get("running")),
        url=result.get("url"),
        title=result.get("title"),
        signed_in_hosts=result.get("signed_in_hosts") or [],
        takeover=takeover,
    )


@router.delete("/sites/{host}", status_code=204)
async def revoke_site(
    host: str,
    db_url: DbUrlDep,
    pool: PoolDep,
    account_id: AccountIdDep,
) -> Response:
    """Delete one host's cookies + storage from the account's profile."""
    await _call(db_url, pool, account_id, "revoke_site", {"host": host})
    return Response(status_code=204)


@router.post("/clear", status_code=204)
async def clear_state(
    db_url: DbUrlDep,
    pool: PoolDep,
    account_id: AccountIdDep,
    session_id: str | None = None,
) -> Response:
    """Clear the account computer's state (profile, downloads, ...). 409 if a
    takeover is open. ``session_id``, when given, receives the
    ``browser_state_lost`` model notice."""
    params: dict[str, Any] = {}
    if session_id:
        # Scope-check the notice recipient, mirroring open_takeover — the worker
        # append is account-scoped too, but the API should not accept a
        # cross-tenant session id here either.
        await sessions_service.get_session_basic(pool, session_id, account_id=account_id)
        params["session_id"] = session_id
    await _call(db_url, pool, account_id, "clear_state", params)
    return Response(status_code=204)
