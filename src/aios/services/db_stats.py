"""Cached collection of bounded database storage statistics."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from time import monotonic
from typing import Any

from aios.db.queries import db_stats as db_stats_queries
from aios.models.db_stats import DatabaseStats

CACHE_SECONDS = 300
FAILURE_CACHE_SECONDS = 10
COLLECTION_TIMEOUT_SECONDS = 10.0
_cache: tuple[float, DatabaseStats] | None = None
_failure_cache: tuple[float, Exception] | None = None
_lock = asyncio.Lock()


async def collect_database_stats(pool: Any) -> DatabaseStats:
    """Collect stats within one wall-clock budget and cache all outcomes briefly."""
    global _cache, _failure_cache
    now = monotonic()
    if _cache is not None and now - _cache[0] < CACHE_SECONDS:
        return _cache[1]
    if _failure_cache is not None and now - _failure_cache[0] < FAILURE_CACHE_SECONDS:
        raise _failure_cache[1]
    async with _lock:
        now = monotonic()
        if _cache is not None and now - _cache[0] < CACHE_SECONDS:
            return _cache[1]
        if _failure_cache is not None and now - _failure_cache[0] < FAILURE_CACHE_SECONDS:
            raise _failure_cache[1]
        try:
            async with asyncio.timeout(COLLECTION_TIMEOUT_SECONDS):
                async with pool.acquire() as conn, conn.transaction(readonly=True):
                    await conn.execute(
                        f"SET LOCAL statement_timeout = '{db_stats_queries.STATEMENT_TIMEOUT_MS}ms'"
                    )
                    database_bytes = await db_stats_queries.database_size(conn)
                    tables = await db_stats_queries.table_storage_stats(conn)
                    buckets = await db_stats_queries.monthly_buckets(conn, tables)
            result = DatabaseStats(
                generated_at=datetime.now(UTC),
                database_bytes=database_bytes,
                tables=tables,
                buckets=buckets,
            )
        except Exception as exc:
            _failure_cache = (monotonic(), exc)
            raise
        _cache = (monotonic(), result)
        _failure_cache = None
        return result
