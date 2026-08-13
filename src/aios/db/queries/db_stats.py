"""Bounded, catalog-derived database storage introspection."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import asyncpg

from aios.models.db_stats import MonthlyStorageBucket, TableStorageStats

TOP_BUCKET_TABLES = 5
BUCKET_MONTHS = 12
MAX_BUCKET_QUERIES = 24
STATEMENT_TIMEOUT_MS = 5_000


async def database_size(conn: asyncpg.Connection[Any]) -> int:
    return int(await conn.fetchval("SELECT pg_database_size(current_database())"))


async def table_storage_stats(conn: asyncpg.Connection[Any]) -> list[TableStorageStats]:
    rows = await conn.fetch(
        """
        SELECT c.relname AS name,
               pg_total_relation_size(c.oid)::bigint AS total_bytes,
               pg_relation_size(c.oid)::bigint AS heap_bytes,
               pg_indexes_size(c.oid)::bigint AS index_bytes,
               greatest(pg_total_relation_size(c.oid) - pg_relation_size(c.oid)
                        - pg_indexes_size(c.oid), 0)::bigint AS toast_bytes,
               greatest(c.reltuples, 0)::bigint AS row_estimate,
               coalesce(s.n_dead_tup, 0)::bigint AS dead_tuple_estimate
          FROM pg_class c
          JOIN pg_namespace n ON n.oid = c.relnamespace
          LEFT JOIN pg_stat_user_tables s ON s.relid = c.oid
         WHERE c.relkind IN ('r', 'p') AND n.nspname = 'public'
         ORDER BY total_bytes DESC, name
        """
    )
    return [TableStorageStats.model_validate(dict(row)) for row in rows]


async def monthly_buckets(
    conn: asyncpg.Connection[Any], tables: list[TableStorageStats]
) -> list[MonthlyStorageBucket]:
    """Ask the planner for monthly estimates; EXPLAIN never reads table rows."""
    candidates = tables[:TOP_BUCKET_TABLES]
    with_created_at = set(
        await conn.fetchval(
            """
            SELECT coalesce(array_agg(table_name), ARRAY[]::text[])
              FROM information_schema.columns
             WHERE table_schema = 'public' AND column_name = 'created_at'
                   AND table_name = ANY($1::text[])
            """,
            [table.name for table in candidates],
        )
    )
    now = datetime.now(UTC)
    current_month = datetime(now.year, now.month, 1, tzinfo=UTC)
    buckets: list[MonthlyStorageBucket] = []
    for table in candidates:
        if table.name not in with_created_at:
            continue
        # Identifier originates in pg_class, then is quoted by Postgres itself.
        identifier = await conn.fetchval("SELECT format('%I.%I', 'public', $1)", table.name)
        for offset in range(BUCKET_MONTHS - 1, -1, -1):
            if len(buckets) >= MAX_BUCKET_QUERIES:
                return buckets
            year = current_month.year
            month = current_month.month - offset
            while month <= 0:
                year -= 1
                month += 12
            start = datetime(year, month, 1, tzinfo=UTC)
            end = datetime(year + (month == 12), 1 if month == 12 else month + 1, 1, tzinfo=UTC)
            plan = await conn.fetchval(
                f"EXPLAIN (FORMAT JSON) SELECT 1 FROM {identifier} "
                f"WHERE created_at >= '{start.isoformat()}'::timestamptz "
                f"AND created_at < '{end.isoformat()}'::timestamptz"
            )
            if isinstance(plan, str):
                plan = json.loads(plan)
            estimate = int(plan[0]["Plan"]["Plan Rows"])
            approx = (
                round(table.total_bytes * estimate / table.row_estimate)
                if table.row_estimate
                else 0
            )
            buckets.append(
                MonthlyStorageBucket(
                    table=table.name,
                    month=start.strftime("%Y-%m"),
                    row_estimate=estimate,
                    approx_bytes=approx,
                )
            )
    return buckets
