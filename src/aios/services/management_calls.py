"""Operator→connector RPC plane: insert pending row, NOTIFY, await result."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg

from aios.db import listen, queries
from aios.errors import ConflictError, ManagementCallTimeoutError
from aios.ids import make_id

# Row outlives the request slightly so a deadline-edge resolve still
# UPDATEs a row that hasn't been GC'd.
_EXPIRY_SLACK_S: float = 5.0


async def submit_call(
    db_url: str,
    pool: asyncpg.Pool[asyncpg.Record],
    *,
    account_id: str,
    connector: str,
    method: str,
    params: dict[str, Any],
    timeout_s: float,
    refuse_if_number_bound: str | None = None,
) -> tuple[Any, bool]:
    """Submit a management call and block until the connector resolves it.

    Returns ``(result, is_error)``; raises :class:`ManagementCallTimeoutError`
    if the connector doesn't POST within ``timeout_s``.

    ``refuse_if_number_bound`` (an ``external_account_id``) makes the
    dispatch EXCLUSIVE with respect to that number's bindings — used by
    irreversible calls such as signal ``unregister`` (#2322).  The
    "no live binding" check and the INSERT of the call row then run in
    ONE transaction under :func:`queries.acquire_connection_number_lock`,
    the same lock every binding-creating path takes.  The committed INSERT
    is the point of no return (the connector may pick the row up via
    NOTIFY or its SSE backfill at any moment after), so it must be inside
    the critical section; the NOTIFY is just a wake-up and stays after
    commit.  A binding that wins the lock first makes this raise
    :class:`ConflictError` with nothing inserted; a binding that arrives
    after the INSERT commits sees the pending call under the same lock
    and is refused (see ``services.connections._lock_number_and_refuse_if_releasing``).

    LISTEN-before-INSERT: flipping the order would race the NOTIFY past
    the queue (same invariant as the SSE handlers in :mod:`aios.api.sse`).
    """
    call_id = make_id("mgmt")
    expires_at = datetime.now(UTC) + timedelta(seconds=timeout_s + _EXPIRY_SLACK_S)

    async with listen.listen_for_connector_result(db_url, call_id) as queue:
        # pool.acquire() yields an autocommit connection; the INSERT
        # commits before the NOTIFY fires.  Do NOT wrap these in
        # ``async with conn.transaction()`` — see db/listen.py for why
        # NOTIFY-after-commit is load-bearing for subscribers.
        async with pool.acquire() as conn:
            if refuse_if_number_bound is None:
                await queries.insert_management_call(
                    conn,
                    call_id=call_id,
                    connector=connector,
                    method=method,
                    params=params,
                    expires_at=expires_at,
                    account_id=account_id,
                )
            else:
                # The INSERT commits at the end of this block (before the
                # NOTIFY below), so NOTIFY-after-commit still holds.
                async with conn.transaction():
                    await queries.acquire_connection_number_lock(
                        conn,
                        account_id=account_id,
                        connector=connector,
                        external_account_id=refuse_if_number_bound,
                    )
                    attached = await queries.list_attached_connections_for_phone(
                        conn,
                        connector,
                        queries.phone_digits(refuse_if_number_bound),
                        account_id=account_id,
                    )
                    if attached:
                        raise ConflictError(
                            f"{connector} number {refuse_if_number_bound} is still attached; "
                            "detach (or archive) the connection before unregistering",
                            detail={
                                "external_account_id": refuse_if_number_bound,
                                "reason": "connection_still_attached",
                                "connection_ids": [c.id for c in attached],
                            },
                        )
                    await queries.insert_management_call(
                        conn,
                        call_id=call_id,
                        connector=connector,
                        method=method,
                        params=params,
                        expires_at=expires_at,
                        account_id=account_id,
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
        row = await queries.get_management_call(conn, call_id, account_id=account_id)
    assert row is not None and row["status"] != "pending"
    return row["result"], row["is_error"]
