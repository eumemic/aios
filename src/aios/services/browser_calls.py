"""API→worker blocking RPC for browser control ops (jarbot#106 §5.7).

The structural copy of :mod:`aios.services.management_calls`, retargeted at
the WORKER (the browser-call listener) instead of a connector runtime. The
ordering invariant is identical and load-bearing:

1. LISTEN on the per-call result channel BEFORE inserting the row, so the
   resolve NOTIFY can never race past an unopened subscription.
2. INSERT + dispatch-NOTIFY on an autocommit connection — never inside a
   transaction, or the NOTIFY would fire before the row is visible.

Control ops are rare and human-paced (takeover open/close, status, per-site
revocation, clear-state) — the per-call result LISTEN briefly consuming an
SSE subscriber slot is acceptable here, unlike the takeover INPUT path,
which deliberately rides the shared filesystem instead of this plane.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg

from aios.db import listen, queries
from aios.errors import BrowserCallTimeoutError
from aios.ids import BROWSER_CALL, make_id

# Slack past the caller timeout so a deadline-edge resolve still finds a
# live row (mirrors management_calls._EXPIRY_SLACK_S).
_EXPIRY_SLACK_S = 5.0


async def submit_browser_call(
    db_url: str,
    pool: asyncpg.Pool[asyncpg.Record],
    *,
    account_id: str,
    method: str,
    params: dict[str, Any],
    timeout_s: float,
) -> tuple[Any, bool]:
    """Submit one browser control op and block for its result.

    Returns ``(result, is_error)``. Raises
    :class:`~aios.errors.BrowserCallTimeoutError` (504) when the worker
    does not resolve the row within ``timeout_s`` — the pending row is
    deliberately left in place for the listener's redrive (idempotent driver
    ops make a late execution safe).
    """
    call_id = make_id(BROWSER_CALL)
    expires_at = datetime.now(UTC) + timedelta(seconds=timeout_s + _EXPIRY_SLACK_S)

    async with listen.listen_for_browser_result(db_url, call_id) as queue:
        async with pool.acquire() as conn:
            # Autocommit on purpose: the dispatch NOTIFY must fire only after
            # the row is visible to the listener's re-fetch.
            await queries.insert_browser_call(
                conn,
                account_id=account_id,
                call_id=call_id,
                method=method,
                params=params,
                expires_at=expires_at,
            )
            await queries.notify_browser_call_dispatch(conn, call_id=call_id)

        try:
            await asyncio.wait_for(queue.get(), timeout=timeout_s)
        except TimeoutError as err:
            raise BrowserCallTimeoutError(
                f"browser call {method!r} timed out after {timeout_s:.0f}s",
                detail={"call_id": call_id, "method": method},
            ) from err

    async with pool.acquire() as conn:
        row = await queries.get_browser_call(conn, call_id, account_id=account_id)
    assert row is not None and row["status"] != "pending"
    return row["result"], row["is_error"]
