"""External-event trigger ingress (#1281) — the ONE account-key-free route.

A per-trigger bearer secret in the URL path (``aios_evt_…``, the runtime_tokens
precedent) authenticates an inbound webhook to exactly one enabled
``external_event`` trigger on a non-archived session. This is the
uncorrelated-substrate intake edge: external ground truth (a GitHub
``issues.labeled``, a monitor alarm) pushes a workflow run directly instead of
having everything reactive poll on cron.

Auth / threat model (CEO disposition, ratified): the token IS the tenant proof.
The resolver matches the SHA-256 of the path token to one row and uses THAT
row's ``account_id`` as the authenticated scope — no caller-supplied account or
session. Unknown / disabled / revoked / archived all collapse to a uniform 404
(the ``lookup_account_by_key_hash`` no-oracle stance). Blast radius is bounded
to that one trigger's owner-clamped action: every fire re-clamps to
``launcher_session_id`` authority (vault subset re-check, version-pin assert,
outstanding-run cap, down-counting depth), so a forged event at worst launches
that one workflow to the owner's run cap, then trips ``MAX_CONSECUTIVE_FAILURES``
auto-disable. Rotate via ``update_trigger`` re-mint.

Check ordering is cheapest-first (security review fold #1): enforce the byte
cap (413) and reject non-object JSON (422) BEFORE the token-hash lookup, and the
hash lookup before the carrier INSERT — a malformed or unauthenticated probe
must never reach the DB.

aios has NO application-level rate-limiting (pre-existing posture); when this
ingress is internet-exposed the fleet edge (proxy) must carry IP-level
rate-limiting on ``/v1/triggers/ingest/*``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from fastapi import APIRouter, Request, status

from aios.api.deps import PoolDep
from aios.db import queries
from aios.errors import NotFoundError, PayloadTooLargeError, ValidationError
from aios.jobs.app import defer_trigger_fire
from aios.logging import get_logger
from aios.models.triggers import MAX_INGEST_EVENT_BYTES

log = get_logger("aios.api.triggers_ingest")

router = APIRouter(prefix="/v1/triggers", tags=["triggers"])


@router.post(
    "/ingest/{ingest_token}",
    operation_id="ingest_external_event",
    status_code=status.HTTP_202_ACCEPTED,
)
async def ingest_external_event(
    ingest_token: str,
    request: Request,
    pool: PoolDep,
) -> dict[str, str]:
    """Resolve a per-trigger ingest token, record a pending fire, and dispatch.

    ``PoolDep`` only — deliberately NO ``AccountIdDep``: this is the sole
    account-key-free route; the path token authenticates and scopes the call.

    Returns ``202 {"trigger_run_id": …}``. A lost post-commit defer is recovered
    by the existing ``list_pending_trigger_run_refs`` sweep — identical
    resilience to run_completion, for free.
    """
    # 1. Byte cap FIRST — cheapest check, before any parse or DB touch. Read the
    #    raw body and reject over-cap with 413; an oversized probe never parses.
    raw = await request.body()
    if len(raw) > MAX_INGEST_EVENT_BYTES:
        raise PayloadTooLargeError(
            f"event body exceeds the {MAX_INGEST_EVENT_BYTES}-byte ingest cap",
            detail={"max_bytes": MAX_INGEST_EVENT_BYTES, "received_bytes": len(raw)},
        )

    # 2. Parse + require a JSON OBJECT. The carrier ``event`` column and the
    #    ``claim_trigger_run`` ``isinstance(event, dict)`` assert both require a
    #    dict, so a non-object (or unparseable) body is a 422 BEFORE the lookup.
    try:
        event: Any = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValidationError(
            "ingest body must be a JSON object",
            detail={"reason": "unparseable JSON"},
        ) from exc
    if not isinstance(event, dict):
        raise ValidationError(
            "ingest body must be a JSON object (got "
            f"{type(event).__name__}) — the event is the object, not a bare scalar/array",
            detail={"reason": "non-object body"},
        )

    # 3. Token hash lookup — uniform 404 on any miss (no oracle): unknown,
    #    disabled, revoked, or archived-owner all look identical to a caller.
    token_hash = hashlib.sha256(ingest_token.encode("utf-8")).hexdigest()

    async with pool.acquire() as conn, conn.transaction():
        resolved = await queries.resolve_external_event_trigger(conn, ingest_token_hash=token_hash)
        if resolved is None:
            raise NotFoundError("not found", detail={})
        # 4. One pending carrier row, account-scoped via the resolved row's own
        #    account_id (the token IS the tenant proof). Same txn as the lookup.
        trigger_run_id = await queries.insert_external_event_fire(
            conn,
            trigger_id=resolved.trigger_id,
            account_id=resolved.account_id,
            owner_session_id=resolved.owner_session_id,
            trigger_name=resolved.trigger_name,
            event=event,
        )

    # 5. Post-commit defer (the fire is durably owed by the committed carrier; a
    #    lost defer is recovered by the pending-row sweep). Mirrors the
    #    run_completion dispatch edge exactly.
    await defer_trigger_fire(resolved.trigger_id, trigger_run_id)
    log.info(
        "trigger.ingest_accepted",
        trigger_id=resolved.trigger_id,
        trigger_run_id=trigger_run_id,
    )
    return {"trigger_run_id": trigger_run_id}
