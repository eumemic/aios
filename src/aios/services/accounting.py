"""Public inference-accounting service."""

from __future__ import annotations

from typing import Any

import asyncpg

from aios.db.queries import accounting as accounting_queries
from aios.db.queries.accounting import USAGE_STATEMENT_TIMEOUT_MS
from aios.models.accounting import UsageConsumersResponse, UsageMetric


async def ranked_consumers(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    window_seconds: int,
    metric: UsageMetric,
    limit: int,
) -> UsageConsumersResponse:
    """Return additive account roots ranked by live subtree rate."""
    # One shared bound for every account-scale usage statement (#2246): the
    # previous 5s bound chronically failed the largest production account
    # (measured 11.6s with JIT, 4-4.3s with the pool's jit=off).
    async with pool.acquire() as conn, conn.transaction(isolation="repeatable_read", readonly=True):
        await conn.execute(f"SET LOCAL statement_timeout = '{USAGE_STATEMENT_TIMEOUT_MS}ms'")
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
