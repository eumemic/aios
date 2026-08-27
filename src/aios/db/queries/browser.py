"""Browser control-plane queries: takeover grants + worker-consumed calls.

Two row families (migration 0175, jarbot#106):

* ``browser_calls`` — the API→worker blocking-RPC rows, the structural twin of
  ``pending_management_calls`` minus the ``connector`` column: their consumer
  is the WORKER's browser-call listener (one channel deployment-wide), never a
  connector runtime, so they get their own table + channel family rather than
  a connector value another runtime token could subscribe to.
* ``browser_grants`` — the control-plane record of a human takeover.
  Worker-written after the driver acks the epoch barrier; API-read with the
  standard scoped pattern (``WHERE account_id = $N`` → cross-tenant ids
  surface as not-found). The partial unique index
  ``browser_grants_one_open_per_account`` makes "at most one open grant per
  computer" true by construction. The driver keeps its own in-memory
  grant/epoch as the ENFORCEMENT authority; these rows are the control-plane
  bookkeeping (scoping, TTL, handback pickup).
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import asyncpg

from aios.db.listen import BROWSER_CALLS_CHANNEL

# ── browser_calls ─────────────────────────────────────────────────────────


async def insert_browser_call(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    call_id: str,
    method: str,
    params: dict[str, Any],
    expires_at: datetime,
) -> None:
    """Insert a fresh ``pending`` row for ``call_id``."""
    await conn.execute(
        """
        INSERT INTO browser_calls (id, account_id, method, params, expires_at)
        VALUES ($1, $2, $3, $4::jsonb, $5)
        """,
        call_id,
        account_id,
        method,
        json.dumps(params),
        expires_at,
    )


async def get_browser_call(
    conn: asyncpg.Connection[Any], call_id: str, *, account_id: str
) -> dict[str, Any] | None:
    """Fetch one browser call by id (account-scoped), or ``None``."""
    row = await conn.fetchrow(
        """
        SELECT id, account_id, method, params, status, result, is_error, expires_at
          FROM browser_calls
         WHERE id = $1 AND account_id = $2
        """,
        call_id,
        account_id,
    )
    return dict(row) if row is not None else None


async def get_browser_call_unscoped(
    conn: asyncpg.Connection[Any], call_id: str
) -> dict[str, Any] | None:
    """Fetch one browser call by id WITHOUT tenant scoping.

    Worker-side only: the listener receives a bare ``call_id`` NOTIFY payload
    and must load the row (including its ``account_id``) to execute it — the
    worker acts for every tenant. API paths use :func:`get_browser_call`.
    """
    row = await conn.fetchrow(
        """
        SELECT id, account_id, method, params, status, result, is_error, expires_at
          FROM browser_calls
         WHERE id = $1
        """,
        call_id,
    )
    return dict(row) if row is not None else None


async def list_pending_browser_calls(conn: asyncpg.Connection[Any]) -> list[dict[str, Any]]:
    """Every pending, unexpired browser call — the listener's redrive sweep.

    NOTIFY is fire-and-forget: a call submitted while the worker's LISTEN
    connection was down (the reconnect-backoff window) is lost unless a
    durable row is re-read on every successful (re)connect. This is that
    re-read; the partial pending index backs it.
    """
    rows = await conn.fetch(
        """
        SELECT id, account_id, method, params, status, result, is_error, expires_at
          FROM browser_calls
         WHERE status = 'pending' AND expires_at > now()
         ORDER BY created_at ASC
        """
    )
    return [dict(row) for row in rows]


async def resolve_browser_call(
    conn: asyncpg.Connection[Any],
    *,
    call_id: str,
    result: Any,
    is_error: bool,
) -> bool:
    """Conditional UPDATE: only resolves a still-``pending`` row.

    Returns ``True`` iff this call moved the row to a terminal state — a
    redrive racing the live dispatch gets ``False`` and skips its NOTIFY.
    Unscoped: the resolver is the worker executor, which loaded the row
    (and its tenant) via :func:`get_browser_call_unscoped`.
    """
    new_status = "failed" if is_error else "succeeded"
    row = await conn.fetchrow(
        """
        UPDATE browser_calls
           SET status = $2, result = $3::jsonb, is_error = $4, resolved_at = now()
         WHERE id = $1 AND status = 'pending'
         RETURNING id
        """,
        call_id,
        new_status,
        json.dumps(result),
        is_error,
    )
    return row is not None


async def delete_finished_browser_calls_before(
    conn: asyncpg.Connection[Any], cutoff: datetime
) -> int:
    """Reap resolved/expired call rows older than ``cutoff``; return the count.

    (``pending_management_calls`` never got this sweep — a known gap; the
    browser plane polices its own rows from day one.)
    """
    status = await conn.execute(
        """
        DELETE FROM browser_calls
         WHERE (status <> 'pending' AND resolved_at < $1)
            OR (status = 'pending' AND expires_at < $1)
        """,
        cutoff,
    )
    return int(status.split()[-1])


async def notify_browser_call_dispatch(conn: asyncpg.Connection[Any], *, call_id: str) -> None:
    """NOTIFY the worker's dispatch channel after inserting a pending row.

    Payload is just ``call_id`` (well under the 8000-byte NOTIFY cap); the
    listener re-fetches the row, so an in-flight payload can't desync from a
    later UPDATE.
    """
    await conn.execute("SELECT pg_notify($1, $2)", BROWSER_CALLS_CHANNEL, call_id)


async def notify_browser_call_result(conn: asyncpg.Connection[Any], *, call_id: str) -> None:
    """NOTIFY the per-call result channel after resolving the row."""
    await conn.execute("SELECT pg_notify($1, $2)", f"browser_result_{call_id}", "")


# ── browser_grants ────────────────────────────────────────────────────────


async def insert_browser_grant(
    conn: asyncpg.Connection[Any],
    *,
    grant_id: str,
    account_id: str,
    session_id: str,
    reason: str,
    boot: str,
    epoch: int,
    target: dict[str, Any],
    ttl_seconds: int,
) -> None:
    """Record an opened takeover — AFTER the driver acked the epoch barrier.

    Raises ``UniqueViolationError`` via the one-open-per-account partial
    index if a concurrent open won; the caller surfaces "a takeover is
    already in progress".
    """
    await conn.execute(
        """
        INSERT INTO browser_grants
            (id, account_id, session_id, reason, boot, epoch, target, ttl_seconds)
        VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8)
        """,
        grant_id,
        account_id,
        session_id,
        reason,
        boot,
        epoch,
        json.dumps(target),
        ttl_seconds,
    )


async def get_browser_grant(
    conn: asyncpg.Connection[Any], grant_id: str, *, account_id: str
) -> dict[str, Any] | None:
    """Fetch one grant by id (account-scoped), or ``None``."""
    row = await conn.fetchrow(
        "SELECT * FROM browser_grants WHERE id = $1 AND account_id = $2",
        grant_id,
        account_id,
    )
    return dict(row) if row is not None else None


async def get_open_browser_grant_for_account(
    conn: asyncpg.Connection[Any], account_id: str
) -> dict[str, Any] | None:
    """The account's open grant, if any (at most one by construction)."""
    row = await conn.fetchrow(
        "SELECT * FROM browser_grants WHERE account_id = $1 AND status = 'open'",
        account_id,
    )
    return dict(row) if row is not None else None


async def touch_browser_grant_heartbeat(
    conn: asyncpg.Connection[Any], grant_id: str, *, account_id: str
) -> bool:
    """Bump ``heartbeat_at`` on an OPEN grant; ``False`` if unknown or closed.

    The rowcount split lets the route distinguish 404 (no such grant for
    this account) from 409 (grant exists but is no longer open).
    """
    status = await conn.execute(
        """
        UPDATE browser_grants
           SET heartbeat_at = now()
         WHERE id = $1 AND account_id = $2 AND status = 'open'
        """,
        grant_id,
        account_id,
    )
    return bool(status != "UPDATE 0")


async def close_browser_grant(
    conn: asyncpg.Connection[Any],
    *,
    grant_id: str,
    status: str,
    outcome: str,
    handback: dict[str, Any] | None,
) -> bool:
    """Move an OPEN grant to ``closed``/``expired`` with its handback payload.

    Conditional on ``status = 'open'`` so a TTL expiry racing an explicit
    close resolves to exactly one winner. Unscoped: callers are the worker
    executor and the grant reaper, both acting from rows they already
    loaded.
    """
    result = await conn.execute(
        """
        UPDATE browser_grants
           SET status = $2, outcome = $3, handback = $4::jsonb, closed_at = now()
         WHERE id = $1 AND status = 'open'
        """,
        grant_id,
        status,
        outcome,
        json.dumps(handback) if handback is not None else None,
    )
    return bool(result != "UPDATE 0")


async def list_stale_open_browser_grants(
    conn: asyncpg.Connection[Any],
) -> list[dict[str, Any]]:
    """Open grants whose heartbeat lapsed past their TTL — the reaper's scan.

    The partial ``(heartbeat_at) WHERE status='open'`` index backs it.
    """
    rows = await conn.fetch(
        """
        SELECT * FROM browser_grants
         WHERE status = 'open'
           AND heartbeat_at < now() - make_interval(secs => ttl_seconds)
        """
    )
    return [dict(row) for row in rows]


async def list_open_browser_grant_accounts(
    conn: asyncpg.Connection[Any],
) -> list[str]:
    """Accounts with an open grant RIGHT NOW — the spool-preservation predicate.

    (Distinct from the fresh-heartbeat set: a grant mid-lapse is still open
    until the reaper closes it, and its human's input spool must survive
    until then.)
    """
    rows = await conn.fetch("SELECT account_id FROM browser_grants WHERE status = 'open'")
    return [row["account_id"] for row in rows]


async def list_fresh_open_browser_grant_accounts(
    conn: asyncpg.Connection[Any],
) -> list[str]:
    """Accounts holding a fresh-heartbeat open grant — the container keepalive.

    The reaper bumps each account's browser-container idle timer so a long
    human session never loses Chromium to the idle reaper.
    """
    rows = await conn.fetch(
        """
        SELECT account_id FROM browser_grants
         WHERE status = 'open'
           AND heartbeat_at >= now() - make_interval(secs => ttl_seconds)
        """
    )
    return [row["account_id"] for row in rows]
