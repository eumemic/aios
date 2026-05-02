"""Channel-state queries for the harness.

In the connector redesign (#200) the explicit ``channel_bindings`` table
is gone — a "binding" is now derived from the event log: a session is
"bound to a channel" if it has any message events stamped with that
channel address.  This module exposes the small surface the harness
needs: list channel addresses for a session, plus a constant for the
focal-channel ``_meta`` key carried on outbound MCP requests.
"""

from __future__ import annotations

from typing import Any

import asyncpg

from aios.db import queries


async def list_session_channels(pool: asyncpg.Pool[Any], session_id: str) -> list[str]:
    """Channel addresses the session has interacted with, sorted.

    Thin wrapper over :func:`aios.db.queries.list_session_channels`.
    Used by the loop to feed the focal-paradigm prose, the channels tail
    block, and the unread-derivation helpers in
    :mod:`aios.harness.channels`.
    """
    async with pool.acquire() as conn:
        return await queries.list_session_channels(conn, session_id)
