"""Rolling per-session, per-verb quotas for outbound tool dispatch.

Redesign (#1903): admission is a **durable reservation**, not a count over
``events``.  Each admitted dispatch inserts one row into
``outbound_tool_reservations`` (migration 0156) keyed on
``(session_id, canonical verb)`` inside a single short, DB-only transaction:

* **Atomic count+insert.** The transaction takes a per-key
  ``pg_advisory_xact_lock``, purges rows older than the rolling window,
  counts the survivors, and inserts the reservation only when the count is
  under the cap.  Concurrent same-key admissions serialize on the lock, so a
  simultaneous burst can never collectively exceed ``max_per_window`` — an
  in-flight dispatch is visible to every later admission as its durable row.
* **No lock or pooled connection across external I/O.** The transaction is
  DB-only (purge + count + insert); the connection is released before the
  caller performs connector I/O or publishes results, and the transaction
  never acquires a second pooled connection.  A one-slot pool cannot
  deadlock, and same-key waiters block only for the milliseconds of a
  sibling's count+insert.
* **Refusals consume nothing.** A refused admission inserts no row, so
  denied retries can never extend the rolling lockout — capacity returns
  exactly when the *admitted* rows age past ``window_seconds``.
* **Admitted capacity is conservative.** Dispatchers admit at the last
  moment before connector invocation (everything earlier is pure or
  read-only), so once a row exists the external side effect may have begun.
  The row therefore keeps counting through crash, cancellation, and local
  publication failure — a retry after an externally-successful send cannot
  exceed the cap.  ``mark_outbound_dispatch_completed`` flips the row to
  ``completed`` for observability only; both states count.
* **Expiry and recovery.** Rows age out of the count at ``window_seconds``
  and are physically purged per key at the next admission; session deletion
  cascades.  A worker crash needs no repair step — the orphaned ``admitted``
  row simply counts until the window rolls past it.

The verb key is canonical: MCP-qualified names (``mcp__<server>__<verb>``)
are matched and counted by their bare connector verb, so sibling effectors
on different servers share one quota.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import asyncpg

from aios.config import get_settings
from aios.logging import get_logger

log = get_logger("aios.services.outbound_tool_quota")


def _display_window(seconds: int) -> str:
    if seconds == 3600:
        return "hour"
    if seconds == 60:
        return "minute"
    if seconds == 86400:
        return "day"
    return f"{seconds} seconds"


def _quota_key(dispatched_name: str) -> str:
    """Return the canonical connector verb from an MCP-qualified dispatch name."""
    parts = dispatched_name.split("__", 2)
    if len(parts) == 3 and parts[0] == "mcp" and parts[2]:
        return parts[2]
    return dispatched_name


_LOCK_SQL = "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))"

_PURGE_SQL = """
    DELETE FROM outbound_tool_reservations
    WHERE session_id = $1
      AND verb = $2
      AND created_at <= now() - make_interval(secs => $3::bigint)
"""

_COUNT_SQL = """
    SELECT count(*)
    FROM outbound_tool_reservations
    WHERE session_id = $1
      AND verb = $2
      AND created_at > now() - make_interval(secs => $3::bigint)
"""

_INSERT_SQL = """
    INSERT INTO outbound_tool_reservations (session_id, verb)
    VALUES ($1, $2)
    RETURNING id
"""

_COMPLETE_SQL = """
    UPDATE outbound_tool_reservations
    SET state = 'completed', updated_at = now()
    WHERE id = $1
"""


@dataclass(frozen=True, slots=True)
class QuotaAdmission:
    """Outcome of one quota admission attempt.

    Exactly one of the shapes below:

    * quota disabled for the verb — ``refusal is None``,
      ``reservation_id is None`` (no query was issued);
    * admitted — ``reservation_id`` set: one unit of capacity is durably
      consumed and the caller may invoke the connector;
    * refused — ``refusal`` is the model-visible ``quota_exceeded`` message;
      nothing was inserted.
    """

    refusal: str | None = None
    reservation_id: str | None = None


_DISABLED = QuotaAdmission()


async def reserve_outbound_tool_quota(
    pool: asyncpg.Pool[Any], session_id: str, dispatched_name: str
) -> QuotaAdmission:
    """Atomically admit one outbound dispatch for ``(session, verb)``.

    Runs the purge + count + insert in one short transaction serialized by a
    per-key advisory xact lock (see module docstring).  The pooled connection
    is acquired and released entirely inside this call — callers must invoke
    it BEFORE connector I/O, never around it.

    Raises whatever the DB raises: the dispatcher runs admission inside
    ``_tool_lifecycle``, so an admission-store failure fails closed as a
    typed, model-visible tool error rather than an unresolved call.
    """
    verb = _quota_key(dispatched_name)
    quota = get_settings().outbound_tool_quotas.get(verb)
    if quota is None:
        return _DISABLED
    window_seconds, maximum = quota
    if window_seconds <= 0 or maximum <= 0:
        return _DISABLED

    async with pool.acquire() as conn, conn.transaction():
        await conn.execute(_LOCK_SQL, f"outbound-tool-quota:{session_id}:{verb}")
        await conn.execute(_PURGE_SQL, session_id, verb, window_seconds)
        count = int(await conn.fetchval(_COUNT_SQL, session_id, verb, window_seconds) or 0)
        if count >= maximum:
            return QuotaAdmission(
                refusal=(
                    f"quota_exceeded: {verb} {count}/{maximum} "
                    f"per {_display_window(window_seconds)}"
                )
            )
        reservation_id = await conn.fetchval(_INSERT_SQL, session_id, verb)
    return QuotaAdmission(reservation_id=str(reservation_id))


async def mark_outbound_dispatch_completed(pool: asyncpg.Pool[Any], reservation_id: str) -> None:
    """Best-effort observability mark: the connector invocation returned.

    Deliberately NOT an accounting operation — ``completed`` rows count
    against the window exactly like ``admitted`` ones (capacity was consumed
    at admission; see module docstring).  A failure here is swallowed and
    logged: the row simply stays ``admitted``, which is the conservative
    state, and the tool result still publishes.
    """
    try:
        async with pool.acquire() as conn:
            await conn.execute(_COMPLETE_SQL, reservation_id)
    except Exception:
        log.warning("outbound_tool_quota.completion_mark_failed", reservation_id=reservation_id)
