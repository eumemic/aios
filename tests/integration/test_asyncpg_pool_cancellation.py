"""Regression reproducer for asyncpg pool starvation under cancellation pressure."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable
from typing import Any

import asyncpg
import pytest

from tests.conftest import needs_docker

pytestmark = [pytest.mark.integration, needs_docker]

_POOL_SIZE = 3
_WORKERS = 24
_ROUNDS = 30
_ACQUIRE_TIMEOUT = 2.0


async def _cancel_at_random_point(
    pool: asyncpg.Pool[Any], *, seed: int, started: asyncio.Event
) -> None:
    """Exercise cancellation while queued, querying, or leaving an acquire block."""
    rng = random.Random(seed)
    started.set()
    async with pool.acquire() as conn:
        await conn.fetchval("SELECT pg_sleep($1::double precision), 1", rng.uniform(0, 0.004))
        # An explicit checkpoint makes query-complete/result-delivery cancellation
        # observable instead of relying exclusively on socket timing.
        await asyncio.sleep(0)


async def _cancel_one(pool: asyncpg.Pool[Any], *, seed: int) -> None:
    rng = random.Random(seed)
    started = asyncio.Event()
    task = asyncio.create_task(_cancel_at_random_point(pool, seed=seed, started=started))

    # Some tasks are cancelled while waiting for acquire; others around query
    # completion or context-manager release.  Zero-delay cancellation supplies
    # the acquire-queue pressure seen during the incident wake storm.
    if rng.randrange(3):
        await started.wait()
    await asyncio.sleep(rng.uniform(0, 0.006))
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def _wake_storm(pool: asyncpg.Pool[Any], *, seed: int) -> None:
    for round_number in range(_ROUNDS):
        await _cancel_one(pool, seed=seed * _ROUNDS + round_number)


async def _must_finish(awaitable: Awaitable[Any]) -> Any:
    async with asyncio.timeout(_ACQUIRE_TIMEOUT):
        return await awaitable


@pytest.mark.asyncio
async def test_cancel_storm_does_not_orphan_asyncpg_pool_connections(
    migrated_db_url: str,
) -> None:
    """Cancellation races must neither park tasks nor leak all pool holders."""
    pool = await asyncpg.create_pool(
        migrated_db_url,
        min_size=_POOL_SIZE,
        max_size=_POOL_SIZE,
        command_timeout=_ACQUIRE_TIMEOUT,
    )
    assert pool is not None
    try:
        storm = asyncio.gather(*(_wake_storm(pool, seed=seed) for seed in range(_WORKERS)))
        await _must_finish(storm)

        # A fresh acquire is the externally visible starvation check.  The idle
        # census additionally catches partial leaks that a single acquire would
        # miss while capacity remains.
        async with asyncio.timeout(_ACQUIRE_TIMEOUT):
            async with pool.acquire() as conn:
                assert await conn.fetchval("SELECT 1") == 1
        assert pool.get_size() == _POOL_SIZE
        assert pool.get_idle_size() == _POOL_SIZE
    finally:
        await _must_finish(pool.close())
