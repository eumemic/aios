from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest

from aios.services import db_stats


class _Connection:
    @contextlib.asynccontextmanager
    async def transaction(self, *, readonly: bool) -> Any:
        yield

    async def execute(self, query: str) -> None:
        return None


class _Pool:
    @contextlib.asynccontextmanager
    async def acquire(self) -> Any:
        yield _Connection()


async def test_collection_has_one_wall_clock_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    async def never_finishes(conn: Any) -> int:
        await asyncio.sleep(60)
        return 0

    monkeypatch.setattr(db_stats, "_cache", None)
    monkeypatch.setattr(db_stats, "COLLECTION_TIMEOUT_SECONDS", 0.001)
    monkeypatch.setattr(db_stats.db_stats_queries, "database_size", never_finishes)

    with pytest.raises(TimeoutError):
        await db_stats.collect_database_stats(_Pool())
