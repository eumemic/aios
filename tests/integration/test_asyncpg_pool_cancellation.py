"""Stress cancellation at every asyncpg pool lifecycle boundary.

This is the reproducer harness requested by #2039.  The original production
freeze could not be reproduced with stock asyncio on Python 3.13 and asyncpg
0.31.0.  Keep the negative result as a regression test: unlike a unit fake,
this exercises asyncpg's real protocol, reset path, and waiter queue against
Postgres under a repeatable wake storm.
"""

from __future__ import annotations

import asyncio
import contextvars
import random
from collections.abc import Callable, Coroutine
from typing import Any

import asyncpg
import pytest

pytestmark = pytest.mark.integration

_POOL_SIZE = 2
_BURST_SIZE = 24
_ROUNDS = 12
_SETTLE_TIMEOUT = 5.0
_RESET_STAGE: contextvars.ContextVar[asyncio.Event | None] = contextvars.ContextVar(
    "reset_stage", default=None
)


async def _cancel_when(stage: asyncio.Event, task: asyncio.Task[None]) -> None:
    await stage.wait()
    task.cancel()


async def _run_cancelled(
    operation: Callable[[asyncio.Event], Coroutine[Any, Any, None]],
) -> None:
    stage = asyncio.Event()
    task = asyncio.create_task(operation(stage))
    canceller = asyncio.create_task(_cancel_when(stage, task))
    results = await asyncio.wait_for(
        asyncio.gather(task, canceller, return_exceptions=True),
        timeout=_SETTLE_TIMEOUT,
    )
    assert isinstance(results[0], asyncio.CancelledError)


async def test_pool_cancellation_storm_returns_every_connection(db_url: str) -> None:
    """No waiter is orphaned when cancellation races acquire/query/release.

    The fixed seed controls the order and query timings while each lifecycle
    stage uses an event to guarantee that cancellation actually lands in the
    intended window.  A small pool forces most acquire attempts through
    asyncpg's waiter queue.
    """
    rng = random.Random(2039)
    acquire_saturation_lock = asyncio.Lock()

    async def slow_reset(conn: asyncpg.Connection[Any]) -> None:
        if stage := _RESET_STAGE.get():
            stage.set()
            await asyncio.sleep(0.01)
        await conn.reset()

    pool: asyncpg.Pool[Any] = await asyncpg.create_pool(
        db_url,
        min_size=_POOL_SIZE,
        max_size=_POOL_SIZE,
        reset=slow_reset,
    )
    assert pool is not None

    async def cancel_mid_acquire(stage: asyncio.Event) -> None:
        async with acquire_saturation_lock:
            held = [await pool.acquire() for _ in range(_POOL_SIZE)]
            try:
                stage.set()
                # Await the pool acquisition in this task.  A detached child
                # would survive cancellation, consume the released connection,
                # and leak it because no caller remains to release the result.
                await pool.acquire()
            finally:
                await asyncio.gather(*(pool.release(conn) for conn in held))

    async def cancel_mid_query(stage: asyncio.Event) -> None:
        async with pool.acquire() as conn:
            query = asyncio.create_task(conn.execute("SELECT pg_sleep(0.02)"))
            await asyncio.sleep(0.002)
            stage.set()
            await query

    async def cancel_at_result_delivery(stage: asyncio.Event) -> None:
        async with pool.acquire() as conn:
            query = asyncio.create_task(conn.fetchval("SELECT 1"))
            query.add_done_callback(lambda _future: stage.set())
            await query
            # This is the production-observed boundary: the protocol future is
            # complete, but its consumer has not reached the next await yet.
            await asyncio.sleep(0)

    async def cancel_mid_release(stage: asyncio.Event) -> None:
        conn = await pool.acquire()
        token = _RESET_STAGE.set(stage)
        try:
            await pool.release(conn)
        finally:
            _RESET_STAGE.reset(token)

    operations = [
        cancel_mid_acquire,
        cancel_mid_query,
        cancel_at_result_delivery,
        cancel_mid_release,
    ]

    try:
        for _ in range(_ROUNDS):
            # Do not mix acquire saturation with the other phases.  A task
            # deliberately holding every connection cannot make progress while
            # query/release tasks are themselves queued for a connection.  That
            # tests a harness deadlock rather than asyncpg cancellation.
            rng.shuffle(operations)
            for operation in operations:
                # Cancellation at one boundary may leave asyncpg's shielded
                # cleanup running briefly.  Start the next probe only after the
                # previous probe has settled so probes cannot consume each
                # other's deliberately constrained pool capacity.
                for _ in range(_BURST_SIZE // len(operations)):
                    await _run_cancelled(operation)
                    # Pool.release() shields cleanup from cancellation, so the
                    # cancelled caller can finish before that cleanup task does.
                    # Wait for it rather than overlapping the next probe with
                    # an intentionally constrained pool.
                    async with asyncio.timeout(_SETTLE_TIMEOUT):
                        while pool.get_idle_size() != _POOL_SIZE:  # noqa: ASYNC110
                            await asyncio.sleep(0)

            # Pool census: every physical connection is idle after the burst.
            assert pool.get_size() == _POOL_SIZE
            assert pool.get_idle_size() == _POOL_SIZE

            # Starvation probe: acquisition must remain bounded after every
            # round, rather than only after the entire stress run.
            async with asyncio.timeout(1.0):
                conn = await pool.acquire()
                await pool.release(conn)
    finally:
        await asyncio.wait_for(pool.close(), timeout=_SETTLE_TIMEOUT)
