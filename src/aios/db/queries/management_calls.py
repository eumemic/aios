"""Pending management-call queries for the operator-to-connector RPC plane."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import asyncpg


async def insert_management_call(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    call_id: str,
    connector: str,
    method: str,
    params: dict[str, Any],
    expires_at: datetime,
) -> None:
    """Insert a fresh ``pending`` row for ``call_id``."""
    await conn.execute(
        """
        INSERT INTO pending_management_calls
            (id, connector, method, params, expires_at, account_id)
        VALUES ($1, $2, $3, $4::jsonb, $5, $6)
        """,
        call_id,
        connector,
        method,
        json.dumps(params),
        expires_at,
        account_id,
    )


async def list_pending_management_calls_for_connector(
    conn: asyncpg.Connection[Any],
    connector: str,
    *,
    account_id: str,
) -> list[dict[str, Any]]:
    """Pending, unexpired management calls for ``connector`` scoped to ``account_id``.

    Used by the runtime SSE backfill on connector reconnect.  Output dict
    shape::

        {"call_id": "mgmt_...", "method": "register", "params": {...}}

    Filtered by ``account_id`` so a runtime container authenticated for
    one tenant never sees another tenant's pending calls. The partial
    index ``pending_management_calls_connector_account_pending_idx``
    (migration 0049) backs this query directly.
    """
    rows = await conn.fetch(
        """
        SELECT id, method, params
          FROM pending_management_calls
         WHERE connector = $1
           AND account_id = $2
           AND status = 'pending'
           AND expires_at > now()
         ORDER BY created_at ASC
        """,
        connector,
        account_id,
    )
    return [
        {
            "call_id": row["id"],
            "method": row["method"],
            "params": row["params"],
        }
        for row in rows
    ]


async def get_management_call(
    conn: asyncpg.Connection[Any], call_id: str, *, account_id: str
) -> dict[str, Any] | None:
    """Fetch one management call by id, or ``None`` if missing.

    Used by both the runtime SSE NOTIFY tail (to assemble the emit
    payload from the freshly-inserted row), the runtime result-intake
    route (to authorise the caller's bearer scope before the conditional
    UPDATE), and the operator-side wake to fetch the resolved row.
    """
    row = await conn.fetchrow(
        """
        SELECT id, connector, method, params, status, result, is_error
          FROM pending_management_calls
         WHERE id = $1 AND account_id = $2
        """,
        call_id,
        account_id,
    )
    if row is None:
        return None
    return {
        "id": row["id"],
        "connector": row["connector"],
        "method": row["method"],
        "params": row["params"],
        "status": row["status"],
        "result": row["result"] if row["result"] is not None else None,
        "is_error": row["is_error"],
    }


async def mark_management_call_resolved(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    call_id: str,
    result: Any,
    is_error: bool,
) -> bool:
    """Conditional UPDATE: only resolves a still-``pending`` row.

    Returns ``True`` iff this call moved the row from ``pending`` to a
    terminal state.  A second POST from a race / retry gets ``False`` —
    the caller no-ops the NOTIFY so the operator never sees a double wake.
    """
    new_status = "failed" if is_error else "succeeded"
    row = await conn.fetchrow(
        """
        UPDATE pending_management_calls
           SET status      = $2,
               result      = $3::jsonb,
               is_error    = $4,
               resolved_at = now()
         WHERE id = $1
           AND status = 'pending'
           AND account_id = $5
         RETURNING id
        """,
        call_id,
        new_status,
        json.dumps(result),
        is_error,
        account_id,
    )
    return row is not None


async def notify_management_call_dispatch(
    conn: asyncpg.Connection[Any],
    *,
    connector: str,
    call_id: str,
) -> None:
    """NOTIFY the per-connector dispatch channel after inserting a pending row.

    Payload is just ``call_id`` so subscribers re-fetch full details from
    the row; keeps the NOTIFY well under Postgres' 8000-byte cap and
    means an in-flight payload can't desync from a later UPDATE.

    Carries no tenancy info — subscribers fetch the row via
    :func:`get_management_call`, which enforces ``WHERE account_id = $N``.
    """
    await conn.execute(
        "SELECT pg_notify($1, $2)",
        f"connector_management_calls_{connector}",
        call_id,
    )


async def notify_management_call_result(
    conn: asyncpg.Connection[Any],
    *,
    call_id: str,
) -> None:
    """NOTIFY the per-call result channel after resolving the row.

    Payload is empty — listeners re-fetch the resolved row via
    :func:`get_management_call`, mirroring the dispatch-side convention
    (which also lets the fetch enforce tenancy).
    """
    await conn.execute(
        "SELECT pg_notify($1, $2)",
        f"connector_result_{call_id}",
        "",
    )
