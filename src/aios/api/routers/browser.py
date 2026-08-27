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
    ConflictError,
    NotFoundError,
    PayloadTooLargeError,
    ServiceUnavailableError,
)
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

    ``browser_unavailable`` → 503, ``takeover_in_progress`` → 409, everything
    else the executor reported as an error → 409 (an op-level refusal the
    caller can act on). A raw success result passes through.
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
    result = await _call(
        db_url,
        pool,
        account_id,
        "open",
        {"session_id": body.session_id, "reason": body.reason},
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
    if existing + len(line) > cap:
        raise PayloadTooLargeError(
            "input spool is full", detail={"cap_bytes": cap, "size_bytes": existing}
        )
    # O_APPEND + a single write: concurrent posts interleave whole lines,
    # never partial ones.
    fd = os.open(spool, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, line)
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
    await _require_grant(pool, grant_id, account_id)  # scope + 404 gate
    plane = browser_plane_dir(account_id)
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
        last_grant_check = 0.0
        try:
            while True:
                manifest = _read_manifest(frames_dir)
                if manifest is not None and manifest.get("seq", -1) > last_seq:
                    frame = _load_frame(frames_dir, manifest, account_id)
                    if frame is not None:
                        last_seq = manifest["seq"]
                        yield ServerSentEvent(data=json.dumps(frame), event="frame")

                now = time.monotonic()
                if now - last_grant_check >= _GRANT_RECHECK_SECONDS:
                    last_grant_check = now
                    async with pool.acquire() as conn:
                        fresh = await queries.get_browser_grant(
                            conn, grant_id, account_id=account_id
                        )
                    if fresh is None or fresh["status"] != "open":
                        outcome = (fresh or {}).get("outcome") or "closed"
                        yield ServerSentEvent(data=json.dumps({"outcome": outcome}), event="end")
                        return
                await asyncio.sleep(_FRAME_POLL_SECONDS)
        finally:
            pass

    # No LISTEN subscription — a no-op terminate satisfies make_sse_response's
    # un-invoked-generator cleanup (the poll loop owns no external resource).
    return make_sse_response(lambda: None, _frames())


def _read_manifest(frames_dir: Path) -> dict[str, Any] | None:
    manifest_path = frames_dir / "manifest.json"
    try:
        raw = manifest_path.read_bytes()
    except OSError:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _load_frame(
    frames_dir: Path, manifest: dict[str, Any], account_id: str
) -> dict[str, Any] | None:
    """Build a frame SSE payload from the manifest, or ``None`` if unreadable.

    Forwards the §5.6 trusted-chrome envelope (minus the account, which never
    crosses to the client) with the JPEG inlined. The manifest's ``file`` is
    containment-checked against the frames dir — a hostile manifest cannot
    read outside the plane.
    """
    file_ref = manifest.get("file")
    if not isinstance(file_ref, str):
        return None
    frame_path = (frames_dir / file_ref).resolve()
    if not frame_path.is_relative_to(frames_dir.resolve()):
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
        params["session_id"] = session_id
    await _call(db_url, pool, account_id, "clear_state", params)
    return Response(status_code=204)
