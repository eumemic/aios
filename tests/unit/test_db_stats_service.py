from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest

from aios.db.queries import db_stats as db_stats_queries
from aios.services import db_stats


class _Connection:
    def transaction(self, *, readonly: bool) -> Any:
        @contextlib.asynccontextmanager
        async def _transaction() -> Any:
            yield

        return _transaction()

    async def execute(self, query: str) -> None:
        return None


class _Pool:
    def __init__(self) -> None:
        self.acquires = 0

    def acquire(self) -> Any:
        @contextlib.asynccontextmanager
        async def _acquire() -> Any:
            self.acquires += 1
            yield _Connection()

        return _acquire()


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    db_stats._cache = None
    db_stats._failure_cache = None


@pytest.mark.asyncio
async def test_collection_has_wall_clock_timeout_and_caches_failure(
    monkeypatch: Any,
) -> None:
    started = asyncio.Event()

    async def slow_database_size(conn: Any) -> int:
        started.set()
        await asyncio.sleep(60)
        return 1

    monkeypatch.setattr(db_stats, "COLLECTION_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(db_stats_queries, "database_size", slow_database_size)
    pool = _Pool()

    with pytest.raises(TimeoutError):
        await db_stats.collect_database_stats(pool)
    assert started.is_set()
    with pytest.raises(TimeoutError):
        await db_stats.collect_database_stats(pool)
    assert pool.acquires == 1
