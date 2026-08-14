from __future__ import annotations

from typing import Any

import pytest

from aios.db.queries import db_stats
from aios.models.db_stats import TableStorageStats


class _Connection:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetchval(self, query: str, *args: Any) -> Any:
        self.calls.append((query, args))
        if "information_schema.columns" in query:
            return ["accounts"]
        if query.startswith("SELECT format"):
            return "public.accounts"
        return '[{"Plan": {"Plan Rows": 0}}]'


@pytest.mark.asyncio
async def test_monthly_buckets_casts_identifier_parameter_for_postgres() -> None:
    """Variadic format() cannot infer an asyncpg parameter without a cast."""
    conn = _Connection()
    table = TableStorageStats(
        name="accounts",
        total_bytes=0,
        heap_bytes=0,
        index_bytes=0,
        toast_bytes=0,
        row_estimate=0,
        dead_tuple_estimate=0,
    )

    await db_stats.monthly_buckets(conn, [table])

    format_calls = [call for call in conn.calls if call[0].startswith("SELECT format")]
    assert format_calls == [("SELECT format('%I.%I', 'public', $1::text)", ("accounts",))]
