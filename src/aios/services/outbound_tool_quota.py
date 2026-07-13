"""Rolling per-session, per-verb quotas for outbound tool dispatch."""

from __future__ import annotations

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


async def check_outbound_tool_quota(
    pool: asyncpg.Pool[Any], session_id: str, dispatched_name: str
) -> str | None:
    """Return a model-visible refusal when the configured rolling cap is full.

    No configured entry means no pool acquisition and no query. The query counts
    persisted tool results, which are one-for-one with completed dispatches.
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
        count = await conn.fetchval(
            """
            SELECT count(*)
            FROM events
            WHERE session_id = $1
              AND tool_name = $2
              AND role = 'tool'
              AND created_at > now() - make_interval(secs => $3::bigint)
            """,
            session_id,
            dispatched_name,
            window_seconds,
        )
    recent = int(count or 0)
    if recent < maximum:
        return None
    return f"quota_exceeded: {quota_name} {recent}/{maximum} per {_display_window(window_seconds)}"
