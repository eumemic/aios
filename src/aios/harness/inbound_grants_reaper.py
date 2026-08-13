"""TTL garbage collection for denied strangers awaiting approval."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import asyncpg

from aios.config import get_settings
from aios.db import queries


async def sweep_inbound_grants(pool: asyncpg.Pool[Any]) -> int:
    settings = get_settings()
    if not settings.inbound_grants_reaper_enabled:
        return 0
    async with pool.acquire() as conn:
        return await queries.reap_pending_inbound_grants(
            conn, ttl=timedelta(seconds=settings.inbound_grants_pending_ttl_seconds)
        )
