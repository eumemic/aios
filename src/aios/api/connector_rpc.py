"""Shared LISTEN-then-defer-then-await helper for connector RPC round-trips.

Both ``routers/connectors.py`` (admin endpoints) and ``routers/connections.py``
(attach drift check) talk to the worker via procrastinate-job + Postgres
NOTIFY.  This helper centralises the LISTEN-before-defer invariant and the
timeout_s-to-408 mapping so a future change to ordering / timeout_s / queue
plumbing lands in one place.

Each caller does its own error-envelope mapping — the connectors router maps
granular ``code`` values onto specific HTTP statuses, the drift check folds
all error envelopes to 503 — so this helper returns the parsed envelope and
leaves error handling to the call site.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import HTTPException, status
from ulid import ULID

from aios.db.listen import listen_for_connector_result


async def connector_rpc(
    db_url: str,
    defer: Callable[[str], Awaitable[None]],
    *,
    timeout_s: float,
) -> dict[str, Any]:
    """LISTEN-then-defer-then-await one connector RPC round-trip.

    ``defer`` is the call-site closure that enqueues the matching
    procrastinate task with the minted ``call_id``.  Order matters: LISTEN
    must be live before ``defer`` runs, otherwise a fast worker could NOTIFY
    before the listener is attached and we'd hang waiting for a payload
    that's already been dropped.

    Returns the parsed envelope dict.  Raises ``HTTPException(408)`` on
    timeout_s.  Error envelopes are returned as-is for the caller to map
    onto the right HTTP status.
    """
    call_id = str(ULID())
    async with listen_for_connector_result(db_url, call_id) as queue:
        await defer(call_id)
        try:
            payload = await asyncio.wait_for(queue.get(), timeout=timeout_s)
        except TimeoutError as exc:
            raise HTTPException(
                status_code=status.HTTP_408_REQUEST_TIMEOUT,
                detail=f"worker did not respond within {timeout_s:g}s",
            ) from exc
    envelope: dict[str, Any] = json.loads(payload)
    return envelope
