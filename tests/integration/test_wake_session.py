"""Integration tests for the ``wake_session`` tool handler against a real DB.

These tests run the handler against a testcontainer-Postgres and
inspect the resulting event-log rows. They cover the cross-session
happy path, the same-account check (a session in account A can NOT
wake a session in account B), the rate-limit cap counted from the
target's event log, and the wake-depth cap counted from the source's
event log.

The procrastinate side of ``defer_wake`` is patched out per
``tests/integration/test_worker_result_after_deny.py`` — the SQL
surface under test is the event-log append + permission gates, not
the job-queue enqueue.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.harness import runtime
from aios.tools.wake_session import (
    WAKE_SESSION_MAX_DEPTH,
    WAKE_SESSION_MAX_PER_HOUR,
    WakeSessionDepthExceededError,
    WakeSessionPermissionError,
    WakeSessionRateLimitedError,
    WakeSessionTargetUnavailableError,
    wake_session_handler,
)
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration


@pytest.fixture
async def pool_with_runtime(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[asyncpg.Pool[Any]]:
    """Yield a pool that's also been installed on ``runtime.pool`` so the
    handler's ``runtime.require_pool()`` sees it."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    prev = runtime.pool
    runtime.pool = pool
    try:
        yield pool
    finally:
        runtime.pool = prev
        await pool.close()


async def _seed_account(pool: asyncpg.Pool[Any], account_id: str, display: str) -> None:
    """Idempotently seed a child tenant account under a shared root.

    The ``accounts_one_active_root`` partial unique index only permits one
    non-archived row with ``parent_account_id IS NULL``, so every test
    account must descend from a single root.  We seed ``acc_wake_root``
    on first call and parent each subsequent account under it.
    """
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
            VALUES ('acc_wake_root', NULL, TRUE, 'wake-test-root')
            ON CONFLICT (id) DO NOTHING
            """
        )
        await conn.execute(
            """
            INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
            VALUES ($1, 'acc_wake_root', FALSE, $2)
            ON CONFLICT (id) DO NOTHING
            """,
            account_id,
            display,
        )


async def _count_user_messages(pool: asyncpg.Pool[Any], session_id: str) -> int:
    async with pool.acquire() as conn:
        return int(
            await conn.fetchval(
                """
                SELECT count(*) FROM events
                WHERE session_id = $1
                  AND kind = 'message'
                  AND data->>'role' = 'user'
                """,
                session_id,
            )
            or 0
        )


@pytest.fixture
def patched_defer_wake() -> Any:
    """Patch out the procrastinate enqueue.  The handler's SQL surface
    (event append, permission check, depth/rate-limit reads) is what's
    under test; the job-queue enqueue is exercised by the existing
    ``tests/unit/test_wake.py`` suite."""
    with mock.patch("aios.tools.wake_session.defer_wake", new_callable=AsyncMock) as m:
        yield m


class TestWakeSessionIntegration:
    async def test_happy_path_appends_user_message_to_target(
        self,
        pool_with_runtime: asyncpg.Pool[Any],
        patched_defer_wake: AsyncMock,
    ) -> None:
        pool = pool_with_runtime
        await _seed_account(pool, "acc_wake_a", "wake-test-a")
        _, _, source = await seed_agent_env_session(pool, account_id="acc_wake_a", prefix="src")
        _, _, target = await seed_agent_env_session(pool, account_id="acc_wake_a", prefix="dst")

        result = await wake_session_handler(
            source.id,
            {"target_session_id": target.id, "prompt": "please escalate"},
        )

        assert result == {
            "woken": True,
            "target_session_id": target.id,
            "wake_depth": 1,
        }

        # One new user-message landed on the target.
        assert await _count_user_messages(pool, target.id) == 1

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT data FROM events
                WHERE session_id = $1
                  AND kind = 'message'
                  AND data->>'role' = 'user'
                ORDER BY seq DESC LIMIT 1
                """,
                target.id,
            )
        assert row is not None
        data = queries.parse_jsonb(row["data"])
        assert data["content"] == "please escalate"
        assert data["metadata"]["wake_source_session_id"] == source.id
        assert data["metadata"]["wake_depth"] == 1

        # defer_wake was called against the TARGET with the target's account.
        patched_defer_wake.assert_awaited_once()
        assert patched_defer_wake.await_args is not None
        assert patched_defer_wake.await_args.args[1] == target.id
        assert patched_defer_wake.await_args.kwargs["account_id"] == "acc_wake_a"
        assert patched_defer_wake.await_args.kwargs["cause"] == "agent_wake"

    async def test_cross_account_rejected(
        self,
        pool_with_runtime: asyncpg.Pool[Any],
        patched_defer_wake: AsyncMock,
    ) -> None:
        pool = pool_with_runtime
        await _seed_account(pool, "acc_wake_x", "wake-test-x")
        await _seed_account(pool, "acc_wake_y", "wake-test-y")
        _, _, source = await seed_agent_env_session(pool, account_id="acc_wake_x", prefix="src-x")
        _, _, target = await seed_agent_env_session(pool, account_id="acc_wake_y", prefix="dst-y")

        with pytest.raises(WakeSessionPermissionError):
            await wake_session_handler(
                source.id,
                {"target_session_id": target.id, "prompt": "cross-account"},
            )

        # No message landed on the target — refusal must be side-effect-free.
        assert await _count_user_messages(pool, target.id) == 0
        patched_defer_wake.assert_not_awaited()

    async def test_archived_target_rejected(
        self,
        pool_with_runtime: asyncpg.Pool[Any],
        patched_defer_wake: AsyncMock,
    ) -> None:
        pool = pool_with_runtime
        await _seed_account(pool, "acc_wake_arch", "wake-test-archived")
        _, _, source = await seed_agent_env_session(pool, account_id="acc_wake_arch", prefix="src")
        _, _, target = await seed_agent_env_session(pool, account_id="acc_wake_arch", prefix="dst")
        async with pool.acquire() as conn:
            await queries.archive_session(conn, target.id, account_id="acc_wake_arch")

        with pytest.raises(WakeSessionTargetUnavailableError, match="archived"):
            await wake_session_handler(
                source.id,
                {"target_session_id": target.id, "prompt": "into the void"},
            )
        patched_defer_wake.assert_not_awaited()

    async def test_wake_depth_inherits_then_caps(
        self,
        pool_with_runtime: asyncpg.Pool[Any],
        patched_defer_wake: AsyncMock,
    ) -> None:
        """Stamp a near-cap depth on the source's most recent user message;
        a wake should bump to cap, the next attempt should refuse."""
        pool = pool_with_runtime
        await _seed_account(pool, "acc_wake_depth", "wake-test-depth")
        _, _, source = await seed_agent_env_session(pool, account_id="acc_wake_depth", prefix="src")
        _, _, target = await seed_agent_env_session(pool, account_id="acc_wake_depth", prefix="dst")

        # Stamp depth = MAX_DEPTH - 1 on the source so the next wake lands
        # exactly at the cap and the one after would breach it.
        async with pool.acquire() as conn:
            await queries.append_event(
                conn,
                account_id="acc_wake_depth",
                session_id=source.id,
                kind="message",
                data={
                    "role": "user",
                    "content": "incoming-wake",
                    "metadata": {"wake_depth": WAKE_SESSION_MAX_DEPTH - 1},
                },
            )

        # First call: depth bumps from MAX-1 to MAX. Allowed.
        result = await wake_session_handler(
            source.id,
            {"target_session_id": target.id, "prompt": "first"},
        )
        assert result["wake_depth"] == WAKE_SESSION_MAX_DEPTH

        # Stamp depth = MAX on the source so the next wake would breach.
        async with pool.acquire() as conn:
            await queries.append_event(
                conn,
                account_id="acc_wake_depth",
                session_id=source.id,
                kind="message",
                data={
                    "role": "user",
                    "content": "second-wake-incoming",
                    "metadata": {"wake_depth": WAKE_SESSION_MAX_DEPTH},
                },
            )

        with pytest.raises(WakeSessionDepthExceededError):
            await wake_session_handler(
                source.id,
                {"target_session_id": target.id, "prompt": "should refuse"},
            )

    async def test_rate_limit_caps_per_pair(
        self,
        pool_with_runtime: asyncpg.Pool[Any],
        patched_defer_wake: AsyncMock,
    ) -> None:
        """Bursting wakes from one source to one target trips the hourly cap."""
        pool = pool_with_runtime
        await _seed_account(pool, "acc_wake_rate", "wake-test-rate")
        _, _, source = await seed_agent_env_session(pool, account_id="acc_wake_rate", prefix="src")
        _, _, target = await seed_agent_env_session(pool, account_id="acc_wake_rate", prefix="dst")

        # Burst up to the cap.
        for i in range(WAKE_SESSION_MAX_PER_HOUR):
            await wake_session_handler(
                source.id,
                {"target_session_id": target.id, "prompt": f"wake-{i}"},
            )

        # One more breaches.
        with pytest.raises(WakeSessionRateLimitedError):
            await wake_session_handler(
                source.id,
                {"target_session_id": target.id, "prompt": "over-cap"},
            )

        # The cap counts CAP not CAP+1 messages on the target.
        assert await _count_user_messages(pool, target.id) == WAKE_SESSION_MAX_PER_HOUR

    async def test_rate_limit_is_per_pair_not_per_source(
        self,
        pool_with_runtime: asyncpg.Pool[Any],
        patched_defer_wake: AsyncMock,
    ) -> None:
        """Bursting from one source to TWO different targets must not
        cross-count: each (source, target) pair has its own window."""
        pool = pool_with_runtime
        await _seed_account(pool, "acc_wake_fan", "wake-test-fanout")
        _, _, source = await seed_agent_env_session(pool, account_id="acc_wake_fan", prefix="src")
        _, _, t_a = await seed_agent_env_session(pool, account_id="acc_wake_fan", prefix="dst-a")
        _, _, t_b = await seed_agent_env_session(pool, account_id="acc_wake_fan", prefix="dst-b")

        # Cap-out target A.
        for i in range(WAKE_SESSION_MAX_PER_HOUR):
            await wake_session_handler(
                source.id,
                {"target_session_id": t_a.id, "prompt": f"a-{i}"},
            )
        with pytest.raises(WakeSessionRateLimitedError):
            await wake_session_handler(
                source.id,
                {"target_session_id": t_a.id, "prompt": "a-over"},
            )

        # Target B should still accept the next wake — a different pair.
        result = await wake_session_handler(
            source.id,
            {"target_session_id": t_b.id, "prompt": "b-first"},
        )
        assert result["woken"] is True
