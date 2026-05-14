"""Operator→connector RPC plane: insert pending row, NOTIFY, await result."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg

from aios.db import listen, queries
from aios.errors import ManagementCallTimeoutError
from aios.ids import make_id

# Row outlives the request slightly so a deadline-edge resolve still
# UPDATEs a row that hasn't been GC'd.
_EXPIRY_SLACK_S: float = 5.0


async def submit_call(
    db_url: str,
    pool: asyncpg.Pool[asyncpg.Record],
    *,
    connector: str,
    method: str,
    params: dict[str, Any],
    timeout_s: float,
) -> tuple[Any, bool]:
    """Submit a management call and block until the connector resolves it.

    Returns ``(result, is_error)``; raises :class:`ManagementCallTimeoutError`
    if the connector doesn't POST within ``timeout_s``.

    LISTEN-before-INSERT: flipping the order would race the NOTIFY past
    the queue (same invariant as the SSE handlers in :mod:`aios.api.sse`).
    """
    call_id = make_id("mgmt")
    expires_at = datetime.now(UTC) + timedelta(seconds=timeout_s + _EXPIRY_SLACK_S)

    async with listen.listen_for_connector_result(db_url, call_id) as queue:
        async with pool.acquire() as conn:
            await queries.insert_management_call(
                conn,
                call_id=call_id,
                connector=connector,
                method=method,
                params=params,
                expires_at=expires_at,
            )
            await queries.notify_management_call_dispatch(
                conn, connector=connector, call_id=call_id
            )

        try:
            await asyncio.wait_for(queue.get(), timeout=timeout_s)
        except TimeoutError as exc:
            raise ManagementCallTimeoutError(
                f"connector {connector!r} did not resolve {method!r} within {timeout_s}s",
                detail={"call_id": call_id, "connector": connector, "method": method},
            ) from exc

    async with pool.acquire() as conn:
        row = await queries.get_management_call(conn, call_id)
    assert row is not None and row["status"] != "pending"
    return row["result"], row["is_error"]
