"""Integration coverage for catalog-derived database storage statistics."""

from __future__ import annotations

from typing import Any

import asyncpg
import pytest

from aios.db.pool import create_pool
from aios.db.queries import db_stats
from aios.models.db_stats import TableStorageStats

pytestmark = pytest.mark.integration


async def test_monthly_buckets_decodes_postgres_explain_json(migrated_db_url: str) -> None:
    """The production pool has a jsonb codec, but EXPLAIN returns json text."""
    pool: asyncpg.Pool[Any] = await create_pool(migrated_db_url, min_size=1, max_size=1)
    try:
        async with pool.acquire() as conn:
            buckets = await db_stats.monthly_buckets(
                conn,
                [
                    TableStorageStats(
                        name="accounts",
                        total_bytes=0,
                        heap_bytes=0,
                        index_bytes=0,
                        toast_bytes=0,
                        row_estimate=0,
                        dead_tuple_estimate=0,
                    )
                ],
            )
    finally:
        await pool.close()

    assert len(buckets) == db_stats.BUCKET_MONTHS
    assert {bucket.table for bucket in buckets} == {"accounts"}
    assert all(bucket.row_estimate >= 0 for bucket in buckets)
