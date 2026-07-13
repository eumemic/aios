"""Rolling per-session, per-verb quotas for outbound tool dispatch."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import asyncpg

from aios.config import get_settings


def _display_window(seconds: int) -> str:
    if seconds == 3600:
        return "hour"
    if seconds == 60:
        return "minute"
    if seconds == 86400:
        return "day"
    return f"{seconds} seconds"


def _quota_key(dispatched_name: str) -> str:
    """Return the connector verb from an MCP-qualified dispatch name."""
    parts = dispatched_name.split("__", 2)
    if len(parts) == 3 and parts[0] == "mcp" and parts[2]:
        return parts[2]
    return dispatched_name


_COUNT_SQL = """
    SELECT count(*)
    FROM events
    WHERE session_id = $1
      AND tool_name = $2
      AND kind = 'span'
      AND data->>'event' = 'tool_execute_end'
      AND data->>'is_error' = 'false'
      AND created_at > now() - make_interval(secs => $3::bigint)
"""


def _refusal(name: str, recent: int, maximum: int, window_seconds: int) -> str | None:
    if recent < maximum:
        return None
    return f"quota_exceeded: {name} {recent}/{maximum} per {_display_window(window_seconds)}"


async def check_outbound_tool_quota(
    pool: asyncpg.Pool[Any], session_id: str, dispatched_name: str
) -> str | None:
    """Read the rolling cap without reserving it (diagnostics/tests only).

    Dispatchers must use :func:`outbound_tool_quota_reservation`, which makes
    admission and publication atomic for a ``(session, verb)`` key.
    """
    quotas = get_settings().outbound_tool_quotas
    quota_name = _quota_key(dispatched_name)
    quota = quotas.get(quota_name)
    if quota is None:
        return None
    window_seconds, maximum = quota
    if window_seconds <= 0 or maximum <= 0:
        return None

    async with pool.acquire() as conn:
        count = await conn.fetchval(_COUNT_SQL, session_id, dispatched_name, window_seconds)
    return _refusal(quota_name, int(count or 0), maximum, window_seconds)


@asynccontextmanager
async def outbound_tool_quota_reservation(
    pool: asyncpg.Pool[Any], session_id: str, dispatched_name: str
) -> AsyncIterator[str | None]:
    """Serialize quota admission through the result publication boundary.

    The transaction-scoped advisory lock is shared by every worker for the
    ``(session_id, configured verb)`` key.  It remains held while the caller
    publishes the tool result and its ``tool_execute_end`` span.  Therefore the
    next waiter observes a successful dispatch before counting, while a failed
    publication rolls back/releases the lock and contributes no successful span.
    Refusals are yielded under the lock and their error spans are deliberately
    excluded by ``_COUNT_SQL``.
    """
    quota_name = _quota_key(dispatched_name)
    quota = get_settings().outbound_tool_quotas.get(quota_name)
    if quota is None:
        yield None
        return
    window_seconds, maximum = quota
    if window_seconds <= 0 or maximum <= 0:
        yield None
        return

    # A transaction lock cannot leak into a pooled connection, including when
    # publication raises or the task is cancelled.
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
            f"outbound-tool-quota\x00{session_id}\x00{quota_name}",
        )
        count = await conn.fetchval(_COUNT_SQL, session_id, dispatched_name, window_seconds)
        yield _refusal(quota_name, int(count or 0), maximum, window_seconds)
