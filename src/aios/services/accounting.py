"""Public inference-accounting service."""

from __future__ import annotations

from typing import Any

import asyncpg

from aios.db.queries import accounting as accounting_queries
from aios.models.accounting import UsageConsumersResponse, UsageMetric

USAGE_CONSUMERS_STATEMENT_TIMEOUT_MS = 5_000


async def ranked_consumers(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    window_seconds: int,
    metric: UsageMetric,
    limit: int,
) -> UsageConsumersResponse:
    """Return additive account roots ranked by live subtree rate."""
    async with pool.acquire() as conn, conn.transaction(isolation="repeatable_read", readonly=True):
        await conn.execute(
            f"SET LOCAL statement_timeout = '{USAGE_CONSUMERS_STATEMENT_TIMEOUT_MS}ms'"
        )
        coverage_started_at, total_rate, items = await accounting_queries.ranked_consumers(
            conn,
            account_id=account_id,
            window_seconds=window_seconds,
            metric=metric,
            limit=limit,
        )
    return UsageConsumersResponse(
        metric=metric,
        window_seconds=window_seconds,
        coverage_started_at=coverage_started_at,
        total_rate_per_hour=total_rate,
        items=items,
    )
