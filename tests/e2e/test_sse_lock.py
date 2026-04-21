"""E2E tests for the SSE subscriber lock → Postgres pg_locks integration."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest


@pytest.fixture
async def pool(aios_env: dict[str, str]) -> AsyncIterator[Any]:
    from aios.config import get_settings
    from aios.db.pool import create_pool

    settings = get_settings()
    p = await create_pool(settings.db_url, min_size=1, max_size=4)
    yield p
    await p.close()


class TestSubscriberLockRoundTrip:
    async def test_acquire_then_probe_true_release_then_probe_false(
        self, pool: Any, aios_env: dict[str, str]
    ) -> None:
        from aios.config import get_settings
        from aios.db.listen import _normalize_dsn
        from aios.db.sse_lock import acquire_subscriber_lock, has_subscriber

        settings = get_settings()
        dsn = _normalize_dsn(settings.db_url)
        session_id = "sess_01KAAA_LOCK_TEST_XX"

        assert await has_subscriber(pool, session_id) is False

        conn = await asyncpg.connect(dsn)
        try:
            await acquire_subscriber_lock(conn, session_id)
            assert await has_subscriber(pool, session_id) is True
        finally:
            await conn.close()

        # Give pg a moment to release — auto-release on close is
        # synchronous in Postgres, but asyncpg's close is async.
        for _ in range(10):
            if not await has_subscriber(pool, session_id):
                break
            await asyncio.sleep(0.05)
        assert await has_subscriber(pool, session_id) is False

    async def test_multiple_subscribers_coexist(self, pool: Any, aios_env: dict[str, str]) -> None:
        from aios.config import get_settings
        from aios.db.listen import _normalize_dsn
        from aios.db.sse_lock import acquire_subscriber_lock, has_subscriber

        settings = get_settings()
        dsn = _normalize_dsn(settings.db_url)
        session_id = "sess_01KBBB_MULTI_LOCK"

        conn_a = await asyncpg.connect(dsn)
        conn_b = await asyncpg.connect(dsn)
        try:
            await acquire_subscriber_lock(conn_a, session_id)
            await acquire_subscriber_lock(conn_b, session_id)
            assert await has_subscriber(pool, session_id) is True

            await conn_a.close()
            # Subscriber B still holds.
            assert await has_subscriber(pool, session_id) is True
        finally:
            await conn_b.close()

        for _ in range(10):
            if not await has_subscriber(pool, session_id):
                break
            await asyncio.sleep(0.05)
        assert await has_subscriber(pool, session_id) is False

    async def test_different_sessions_are_isolated(
        self, pool: Any, aios_env: dict[str, str]
    ) -> None:
        from aios.config import get_settings
        from aios.db.listen import _normalize_dsn
        from aios.db.sse_lock import acquire_subscriber_lock, has_subscriber

        settings = get_settings()
        dsn = _normalize_dsn(settings.db_url)

        conn = await asyncpg.connect(dsn)
        try:
            await acquire_subscriber_lock(conn, "sess_01KCC_SESSION_A")
            assert await has_subscriber(pool, "sess_01KCC_SESSION_A") is True
            assert await has_subscriber(pool, "sess_01KDD_SESSION_B") is False
        finally:
            await conn.close()
