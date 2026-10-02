"""Pending management-call queries for the operator-to-connector RPC plane."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import asyncpg

# Management-call methods that hold a guard on their number until the row is
# TERMINAL (#2322): the binding-creating paths refuse while one is pending,
# whatever its ``expires_at``.  The SSE backfill therefore keeps redelivering
# them past expiry so a connector restart still drives them to a terminal
# status.
REDELIVER_UNTIL_TERMINAL_METHODS: frozenset[str] = frozenset({"unregister"})


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


async def has_pending_management_call_for_number(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    connector: str,
    method: str,
    phone_digits: str,
) -> bool:
    """Whether a NON-TERMINAL ``method`` call targets this number.

    Matches ``params->>'external_account_id'`` digits-only, the same
    normal form :func:`acquire_connection_number_lock` keys on.

    Deliberately ignores ``expires_at`` (#2322 F3): ``expires_at`` only
    bounds how long the OPERATOR waits.  A connector that has already
    received the call executes it whenever its serial management loop
    reaches it, expired or not, and its result POST is still accepted.  So
    the call can run until the row reaches a terminal status
    (``succeeded`` / ``failed``), and the guard holds until then — no clock
    is involved.  A non-terminal ``unregister`` row is redelivered on every
    connector (re)connect regardless of expiry
    (:func:`list_pending_management_calls_for_connector`), and the operator
    can terminalise one explicitly
    (:func:`cancel_pending_management_calls_for_number`).
    """
    return bool(
        await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1
                  FROM pending_management_calls
                 WHERE connector = $1
                   AND account_id = $2
                   AND method = $3
                   AND status = 'pending'
                   AND regexp_replace(params->>'external_account_id', '[^0-9]', '', 'g') = $4
            )
            """,
            connector,
            account_id,
            method,
            phone_digits,
        )
    )


async def cancel_pending_management_calls_for_number(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    connector: str,
    method: str,
    phone_digits: str,
) -> list[str]:
    """Terminalise (``failed``, ``cancelled_by_operator``) every still-pending
    ``method`` call for this number; return the ids moved.

    The operator escape hatch for a guard that would otherwise never clear
    (connector gone for good, or a result POST that was lost after the
    connector executed the call).  Conditional on ``status = 'pending'``
    exactly like :func:`mark_management_call_resolved`, so it never
    overwrites a real result.  Callers must hold the number lock.
    """
    rows = await conn.fetch(
        """
        UPDATE pending_management_calls
           SET status      = 'failed',
               result      = $5::jsonb,
               is_error    = true,
               resolved_at = now()
         WHERE connector = $1
           AND account_id = $2
           AND method = $3
           AND status = 'pending'
           AND regexp_replace(params->>'external_account_id', '[^0-9]', '', 'g') = $4
         RETURNING id
        """,
        connector,
        account_id,
        method,
        phone_digits,
        json.dumps({"error": "cancelled by operator", "code": "cancelled_by_operator"}),
    )
    return sorted(r["id"] for r in rows)


async def list_pending_management_calls_for_connector(
    conn: asyncpg.Connection[Any],
    connector: str,
    *,
    account_id: str,
) -> list[dict[str, Any]]:
    """Pending management calls for ``connector`` scoped to ``account_id``.

    Unexpired calls of every method, plus every still-pending call of a
    method in :data:`REDELIVER_UNTIL_TERMINAL_METHODS` regardless of
    ``expires_at`` (#2322 F3).  Such a call blocks binding creation on its
    number until it is terminal; redelivering it means a connector that
    crashed (or restarted) before executing it picks it up again and
    resolves it, instead of the number staying un-bindable forever.  It is
    safe to run late precisely because no binding can exist meanwhile.

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
           AND (expires_at > now() OR method = ANY($3::text[]))
         ORDER BY created_at ASC
        """,
        connector,
        account_id,
        list(REDELIVER_UNTIL_TERMINAL_METHODS),
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
