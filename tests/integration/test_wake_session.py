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

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.harness import runtime
from aios.harness.trigger_runner import _run_wake_session
from aios.models.triggers import WakeSessionAction
from aios.services.wake import (
    WAKE_LINEAGE_SPAN_EVENT,
    CrossSessionWakeRoot,
    deliver_cross_session_wake,
)
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


async def _count_lineage_spans(
    pool: asyncpg.Pool[Any], session_id: str, *, source_id: str | None = None
) -> int:
    """Count the trusted ``wake_lineage`` span rows on ``session_id`` — the
    exact relation the per-pair rate cap reads.  When ``source_id`` is given,
    count only spans stamped with that source (the per-pair bucket)."""
    async with pool.acquire() as conn:
        if source_id is None:
            return int(
                await conn.fetchval(
                    """
                    SELECT count(*) FROM events
                    WHERE session_id = $1
                      AND kind = 'span'
                      AND data->>'event' = $2
                    """,
                    session_id,
                    WAKE_LINEAGE_SPAN_EVENT,
                )
                or 0
            )
        return int(
            await conn.fetchval(
                """
                SELECT count(*) FROM events
                WHERE session_id = $1
                  AND kind = 'span'
                  AND data->>'event' = $2
                  AND data->>'wake_source_session_id' = $3
                """,
                session_id,
                WAKE_LINEAGE_SPAN_EVENT,
                source_id,
            )
            or 0
        )


async def _stamp_lineage_span(
    pool: asyncpg.Pool[Any], session_id: str, account_id: str, depth: int
) -> None:
    """Append a system-owned ``wake_lineage`` span — the trusted depth carrier
    the cap reads. Mirrors what ``wake_session_handler`` writes on a real wake.
    """
    from aios.services.wake import WAKE_LINEAGE_SPAN_EVENT

    async with pool.acquire() as conn:
        await queries.append_event(
            conn,
            account_id=account_id,
            session_id=session_id,
            kind="span",
            data={
                "event": WAKE_LINEAGE_SPAN_EVENT,
                "wake_source_session_id": "sess_seed_source",
                "wake_depth": depth,
            },
        )


@pytest.fixture
def patched_defer_wake() -> Any:
    """Patch out the procrastinate enqueue.  The handler's SQL surface
    (event append, permission check, depth/rate-limit reads) is what's
    under test; the job-queue enqueue is exercised by the existing
    ``tests/unit/test_wake.py`` suite.

    ``defer_wake`` now lives in ``aios.services.wake`` (issue #1280 moved the
    cross-session delivery primitive there); ``deliver_cross_session_wake``
    calls it from that module, so the patch must target the definition site."""
    with mock.patch("aios.services.wake.defer_wake", new_callable=AsyncMock) as m:
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
        data = row["data"]
        assert data["content"] == "please escalate"
        assert data["metadata"]["wake_source_session_id"] == source.id
        assert data["metadata"]["wake_depth"] == 1

        # defer_wake was called against the TARGET with the target's account.
        patched_defer_wake.assert_awaited_once()
        assert patched_defer_wake.await_args is not None
        assert patched_defer_wake.await_args.args[1] == target.id
        assert patched_defer_wake.await_args.kwargs["account_id"] == "acc_wake_a"
        assert patched_defer_wake.await_args.kwargs["cause"] == "agent_wake"

    async def test_span_precedes_message_atomically(
        self,
        pool_with_runtime: asyncpg.Pool[Any],
        patched_defer_wake: AsyncMock,
    ) -> None:
        """The trusted ``wake_lineage`` span and its user message land in ONE
        transaction, span FIRST: the span's seq is exactly one below the message's.

        This is the observable proof of the atomicity fix — nothing can interleave
        between them (the session row is locked for the whole transaction, so the
        seqs are adjacent), and the trusted depth carrier is never visible later than
        the message that makes the target sweep-wakeable. A non-atomic message-first
        append would leave a window where the sweep wakes the target into a step that
        reads a stale wake-depth and undercounts the chain."""
        pool = pool_with_runtime
        await _seed_account(pool, "acc_wake_atomic", "wake-test-atomic")
        _, _, source = await seed_agent_env_session(
            pool, account_id="acc_wake_atomic", prefix="src"
        )
        _, _, target = await seed_agent_env_session(
            pool, account_id="acc_wake_atomic", prefix="dst"
        )

        await wake_session_handler(
            source.id,
            {"target_session_id": target.id, "prompt": "escalate"},
        )

        async with pool.acquire() as conn:
            span_seq = await conn.fetchval(
                """
                SELECT seq FROM events
                WHERE session_id = $1 AND kind = 'span'
                  AND data->>'event' = 'wake_lineage'
                ORDER BY seq DESC LIMIT 1
                """,
                target.id,
            )
            msg_seq = await conn.fetchval(
                """
                SELECT seq FROM events
                WHERE session_id = $1 AND kind = 'message' AND data->>'role' = 'user'
                ORDER BY seq DESC LIMIT 1
                """,
                target.id,
            )
        assert span_seq is not None and msg_seq is not None
        assert span_seq == msg_seq - 1, (
            f"span seq {span_seq} must immediately precede message seq {msg_seq} "
            "(span-first, one transaction, nothing interleaved)"
        )

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
        """Stamp a near-cap depth in the trusted wake_lineage span on the
        source; a wake should bump to cap, the next attempt should refuse."""
        pool = pool_with_runtime
        await _seed_account(pool, "acc_wake_depth", "wake-test-depth")
        _, _, source = await seed_agent_env_session(pool, account_id="acc_wake_depth", prefix="src")
        _, _, target = await seed_agent_env_session(pool, account_id="acc_wake_depth", prefix="dst")

        # Stamp depth = MAX_DEPTH - 1 in the TRUSTED span carrier on the source
        # so the next wake lands exactly at the cap and the one after breaches.
        await _stamp_lineage_span(pool, source.id, "acc_wake_depth", WAKE_SESSION_MAX_DEPTH - 1)

        # First call: depth bumps from MAX-1 to MAX. Allowed.
        result = await wake_session_handler(
            source.id,
            {"target_session_id": target.id, "prompt": "first"},
        )
        assert result["wake_depth"] == WAKE_SESSION_MAX_DEPTH

        # Stamp depth = MAX in the trusted span so the next wake would breach.
        await _stamp_lineage_span(pool, source.id, "acc_wake_depth", WAKE_SESSION_MAX_DEPTH)

        with pytest.raises(WakeSessionDepthExceededError):
            await wake_session_handler(
                source.id,
                {"target_session_id": target.id, "prompt": "should refuse"},
            )

    async def test_forged_user_metadata_does_not_evade_depth_cap(
        self,
        pool_with_runtime: asyncpg.Pool[Any],
        patched_defer_wake: AsyncMock,
    ) -> None:
        """Red test for #1083: a caller posts a user message with a FORGED
        ``metadata.wake_depth = 0`` on top of a real near-cap lineage span.
        The cap must compute the TRUE depth from the trusted span and refuse
        — the forged user metadata must be ignored.
        """
        pool = pool_with_runtime
        await _seed_account(pool, "acc_wake_forge", "wake-test-forge")
        _, _, source = await seed_agent_env_session(pool, account_id="acc_wake_forge", prefix="src")
        _, _, target = await seed_agent_env_session(pool, account_id="acc_wake_forge", prefix="dst")

        # The real, system-stamped lineage puts the source at the cap.
        await _stamp_lineage_span(pool, source.id, "acc_wake_forge", WAKE_SESSION_MAX_DEPTH)

        # Attacker injects a LATER user message claiming depth 0 — exactly
        # what the operator-POST / connector paths pass through unstripped.
        async with pool.acquire() as conn:
            await queries.append_event(
                conn,
                account_id="acc_wake_forge",
                session_id=source.id,
                kind="message",
                data={
                    "role": "user",
                    "content": "reset my depth pretty please",
                    "metadata": {"wake_depth": 0, "wake_source_session_id": "sess_forged"},
                },
            )

        # The forged metadata must NOT reset the chain: the cap still sees
        # the trusted depth at MAX and refuses.
        with pytest.raises(WakeSessionDepthExceededError):
            await wake_session_handler(
                source.id,
                {"target_session_id": target.id, "prompt": "evade the cap"},
            )

    async def test_forged_user_metadata_does_not_evade_rate_limit(
        self,
        pool_with_runtime: asyncpg.Pool[Any],
        patched_defer_wake: AsyncMock,
    ) -> None:
        """Red test for #1083 (rate-limit side): the per-pair count must be
        derived from the trusted lineage span, so forged user-message
        metadata can neither inflate nor evade the count.

        Here we cap the pair with REAL wakes, then inject a forged user
        message claiming a DIFFERENT ``wake_source_session_id`` — which, if
        trusted, would lower the count and let the cap be evaded.  The cap
        must still refuse the next wake.
        """
        pool = pool_with_runtime
        await _seed_account(pool, "acc_wake_rforge", "wake-test-rforge")
        _, _, source = await seed_agent_env_session(
            pool, account_id="acc_wake_rforge", prefix="src"
        )
        _, _, target = await seed_agent_env_session(
            pool, account_id="acc_wake_rforge", prefix="dst"
        )

        for i in range(WAKE_SESSION_MAX_PER_HOUR):
            await wake_session_handler(
                source.id,
                {"target_session_id": target.id, "prompt": f"wake-{i}"},
            )

        # Forge a user message on the target whose metadata claims a
        # different source — counted nowhere if user metadata is ignored.
        async with pool.acquire() as conn:
            await queries.append_event(
                conn,
                account_id="acc_wake_rforge",
                session_id=target.id,
                kind="message",
                data={
                    "role": "user",
                    "content": "noise",
                    "metadata": {"wake_source_session_id": "sess_other"},
                },
            )

        with pytest.raises(WakeSessionRateLimitedError):
            await wake_session_handler(
                source.id,
                {"target_session_id": target.id, "prompt": "over-cap"},
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

    # ─── concurrency: the per-(source, target) advisory lock ──────────────
    #
    # The rate cap is enforced by count-then-append under a transaction-scoped
    # ``pg_advisory_xact_lock`` keyed on ``(root.source_id, target)``.  Before
    # the lock, a burst of same-pair ``wake_session`` calls each read the count
    # on a separate pooled connection (READ COMMITTED — each saw only committed
    # rows), all passed the check, and all appended — overshooting the cap by
    # 1.4-2.7x (measured).  The tests below fire concurrent bursts through the
    # same ``wake_session_handler`` the agent tool dispatches to and assert the
    # cap holds EXACTLY, that refusals are side-effect-free, and that distinct
    # pairs do not serialize onto each other.

    async def test_rate_limit_holds_under_concurrent_burst(
        self,
        pool_with_runtime: asyncpg.Pool[Any],
        patched_defer_wake: AsyncMock,
    ) -> None:
        """A concurrent burst of ``wake_session`` calls to one (source,
        target) pair must NOT overshoot ``WAKE_SESSION_MAX_PER_HOUR``.

        Pre-fix, every concurrent caller read a stale count on its own pooled
        connection and all appended (count-read throughput x append-commit
        latency window).  The per-pair advisory lock serializes the count +
        append so each caller sees every prior caller's committed span, and the
        cap holds exactly.
        """
        pool = pool_with_runtime
        await _seed_account(pool, "acc_wake_conc", "wake-test-concurrent")
        _, _, source = await seed_agent_env_session(pool, account_id="acc_wake_conc", prefix="src")
        _, _, target = await seed_agent_env_session(pool, account_id="acc_wake_conc", prefix="dst")

        burst = 30
        results = await asyncio.gather(
            *(
                wake_session_handler(
                    source.id,
                    {"target_session_id": target.id, "prompt": f"wake-{i}"},
                )
                for i in range(burst)
            ),
            return_exceptions=True,
        )

        delivered = await _count_lineage_spans(pool, target.id, source_id=source.id)
        # The cap holds EXACTLY under concurrency — no overshoot.  This is the
        # load-bearing assertion: pre-fix this was 25-27.
        assert delivered == WAKE_SESSION_MAX_PER_HOUR
        # Parity: every delivered span corresponds to exactly one successful
        # tool result; the rest are rate-limited refusals.
        ok = [r for r in results if isinstance(r, dict) and r.get("woken") is True]
        refused = [r for r in results if isinstance(r, WakeSessionRateLimitedError)]
        assert len(ok) == WAKE_SESSION_MAX_PER_HOUR
        assert len(refused) == burst - WAKE_SESSION_MAX_PER_HOUR
        # No other error shape leaked through.
        assert len(ok) + len(refused) == burst
        # Exactly one defer per delivered wake — refusals consume nothing.
        assert patched_defer_wake.await_count == WAKE_SESSION_MAX_PER_HOUR
        # And the user-message rows the windower would materialize match the
        # cap (no context-window displacement by overshot deliveries).
        assert await _count_user_messages(pool, target.id) == WAKE_SESSION_MAX_PER_HOUR

    async def test_rate_limit_holds_from_near_saturated_with_small_burst(
        self,
        pool_with_runtime: asyncpg.Pool[Any],
        patched_defer_wake: AsyncMock,
    ) -> None:
        """The realistic-severity path: the pair is primed to ``cap - 1``
        sequentially (each wake sees a fresh, accurate count — no race), then a
        modest concurrent burst whose remaining budget is 1 fires.  A correct
        serializer delivers exactly one of the burst (total = cap); pre-fix the
        concurrent count-reads all saw ``cap - 1`` and all appended, delivering
        6-8 (overshooting by 5-7)."""
        pool = pool_with_runtime
        await _seed_account(pool, "acc_wake_near", "wake-test-near-saturated")
        _, _, source = await seed_agent_env_session(pool, account_id="acc_wake_near", prefix="src")
        _, _, target = await seed_agent_env_session(pool, account_id="acc_wake_near", prefix="dst")

        # Prime to cap-1 sequentially — accurate counts, no race.
        for i in range(WAKE_SESSION_MAX_PER_HOUR - 1):
            await wake_session_handler(
                source.id,
                {"target_session_id": target.id, "prompt": f"prime-{i}"},
            )
        assert (
            await _count_lineage_spans(pool, target.id, source_id=source.id)
            == WAKE_SESSION_MAX_PER_HOUR - 1
        )

        # Concurrent burst of 10; remaining budget = 1.
        burst = 10
        results = await asyncio.gather(
            *(
                wake_session_handler(
                    source.id,
                    {"target_session_id": target.id, "prompt": f"burst-{i}"},
                )
                for i in range(burst)
            ),
            return_exceptions=True,
        )

        delivered = await _count_lineage_spans(pool, target.id, source_id=source.id)
        # Exactly cap total — only one burst call got the remaining slot.
        assert delivered == WAKE_SESSION_MAX_PER_HOUR
        ok = [r for r in results if isinstance(r, dict) and r.get("woken") is True]
        refused = [r for r in results if isinstance(r, WakeSessionRateLimitedError)]
        assert len(ok) == 1
        assert len(refused) == burst - 1
        assert patched_defer_wake.await_count == WAKE_SESSION_MAX_PER_HOUR

    async def test_concurrent_burst_distinct_targets_share_no_pair_lock(
        self,
        pool_with_runtime: asyncpg.Pool[Any],
        patched_defer_wake: AsyncMock,
    ) -> None:
        """Concurrent wakes from ONE source to TWO targets must NOT serialize
        onto each other — the per-pair lock keys on ``(source, target)``, so
        each pair gets its own full ``WAKE_SESSION_MAX_PER_HOUR`` budget.  This
        proves the lock is per-pair, not per-source (an over-broad key would
        starve the second target)."""
        pool = pool_with_runtime
        await _seed_account(pool, "acc_wake_fanout_conc", "wake-test-fanout-conc")
        _, _, source = await seed_agent_env_session(
            pool, account_id="acc_wake_fanout_conc", prefix="src"
        )
        _, _, t_a = await seed_agent_env_session(
            pool, account_id="acc_wake_fanout_conc", prefix="dst-a"
        )
        _, _, t_b = await seed_agent_env_session(
            pool, account_id="acc_wake_fanout_conc", prefix="dst-b"
        )

        per_target_burst = 15  # > cap, so each target would overshoot in isolation
        results = await asyncio.gather(
            *(
                wake_session_handler(
                    source.id,
                    {"target_session_id": t_a.id, "prompt": f"a-{i}"},
                )
                for i in range(per_target_burst)
            ),
            *(
                wake_session_handler(
                    source.id,
                    {"target_session_id": t_b.id, "prompt": f"b-{i}"},
                )
                for i in range(per_target_burst)
            ),
            return_exceptions=True,
        )

        delivered_a = await _count_lineage_spans(pool, t_a.id, source_id=source.id)
        delivered_b = await _count_lineage_spans(pool, t_b.id, source_id=source.id)
        # Each pair independently caps at WAKE_SESSION_MAX_PER_HOUR — the two
        # pairs ran concurrently, neither starved the other.
        assert delivered_a == WAKE_SESSION_MAX_PER_HOUR
        assert delivered_b == WAKE_SESSION_MAX_PER_HOUR
        ok = [r for r in results if isinstance(r, dict) and r.get("woken") is True]
        refused = [r for r in results if isinstance(r, WakeSessionRateLimitedError)]
        assert len(ok) == 2 * WAKE_SESSION_MAX_PER_HOUR
        assert len(refused) == 2 * per_target_burst - 2 * WAKE_SESSION_MAX_PER_HOUR
        assert patched_defer_wake.await_count == 2 * WAKE_SESSION_MAX_PER_HOUR

    async def test_concurrent_burst_distinct_sources_share_no_pair_lock(
        self,
        pool_with_runtime: asyncpg.Pool[Any],
        patched_defer_wake: AsyncMock,
    ) -> None:
        """Concurrent wakes from TWO sources to ONE target must NOT serialize
        onto each other — the lock keys on ``(source, target)``, so each
        source gets its own per-pair budget into the shared target.  This is
        the case a per-TARGET-row ``SELECT … FOR UPDATE`` would get wrong
        (over-serializing distinct sources onto the popular target row); the
        per-pair advisory lock is scope-matched to the cap and avoids it."""
        pool = pool_with_runtime
        await _seed_account(pool, "acc_wake_many_src", "wake-test-many-src")
        _, _, src_a = await seed_agent_env_session(
            pool, account_id="acc_wake_many_src", prefix="src-a"
        )
        _, _, src_b = await seed_agent_env_session(
            pool, account_id="acc_wake_many_src", prefix="src-b"
        )
        _, _, target = await seed_agent_env_session(
            pool, account_id="acc_wake_many_src", prefix="dst"
        )

        per_source_burst = 15
        results = await asyncio.gather(
            *(
                wake_session_handler(
                    src_a.id,
                    {"target_session_id": target.id, "prompt": f"a-{i}"},
                )
                for i in range(per_source_burst)
            ),
            *(
                wake_session_handler(
                    src_b.id,
                    {"target_session_id": target.id, "prompt": f"b-{i}"},
                )
                for i in range(per_source_burst)
            ),
            return_exceptions=True,
        )

        delivered_from_a = await _count_lineage_spans(pool, target.id, source_id=src_a.id)
        delivered_from_b = await _count_lineage_spans(pool, target.id, source_id=src_b.id)
        # Each source gets its own per-pair budget into the shared target;
        # the target row lock a ``FOR UPDATE`` serializer would need is NOT
        # taken, so the two sources run concurrently.
        assert delivered_from_a == WAKE_SESSION_MAX_PER_HOUR
        assert delivered_from_b == WAKE_SESSION_MAX_PER_HOUR
        ok = [r for r in results if isinstance(r, dict) and r.get("woken") is True]
        refused = [r for r in results if isinstance(r, WakeSessionRateLimitedError)]
        assert len(ok) == 2 * WAKE_SESSION_MAX_PER_HOUR
        assert len(refused) == 2 * per_source_burst - 2 * WAKE_SESSION_MAX_PER_HOUR
        assert patched_defer_wake.await_count == 2 * WAKE_SESSION_MAX_PER_HOUR

    async def test_rate_limit_window_rolls_under_concurrency_after_backdate(
        self,
        pool_with_runtime: asyncpg.Pool[Any],
        patched_defer_wake: AsyncMock,
    ) -> None:
        """The rolling-hour window still rolls after the fix: backdating the
        delivered spans past one hour must free the per-pair budget, and a
        fresh concurrent burst must deliver up to the cap again (not be
        locked out forever by the advisory lock — the lock is transaction-
        scoped and released on commit)."""
        pool = pool_with_runtime
        await _seed_account(pool, "acc_wake_roll", "wake-test-roll")
        _, _, source = await seed_agent_env_session(pool, account_id="acc_wake_roll", prefix="src")
        _, _, target = await seed_agent_env_session(pool, account_id="acc_wake_roll", prefix="dst")

        # Fill the per-pair budget sequentially.
        for i in range(WAKE_SESSION_MAX_PER_HOUR):
            await wake_session_handler(
                source.id,
                {"target_session_id": target.id, "prompt": f"first-{i}"},
            )
        # The next sequential wake is refused.
        with pytest.raises(WakeSessionRateLimitedError):
            await wake_session_handler(
                source.id,
                {"target_session_id": target.id, "prompt": "over-cap"},
            )

        # Age every lineage span on the target past the 1-hour window.
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE events
                   SET created_at = created_at - interval '2 hours'
                 WHERE session_id = $1
                   AND kind = 'span'
                   AND data->>'event' = $2
                """,
                target.id,
                WAKE_LINEAGE_SPAN_EVENT,
            )

        # A concurrent burst after the window rolls must deliver up to the
        # cap again — the lock holds nothing across the boundary.
        burst = 15
        results = await asyncio.gather(
            *(
                wake_session_handler(
                    source.id,
                    {"target_session_id": target.id, "prompt": f"second-{i}"},
                )
                for i in range(burst)
            ),
            return_exceptions=True,
        )
        # The backdated spans no longer count; only the new burst does.
        recent = await _count_lineage_spans(pool, target.id, source_id=source.id)
        # ``_count_lineage_spans`` counts ALL spans (not just recent) — subtract
        # the backdated ones to recover the in-window count.
        async with pool.acquire() as conn:
            in_window = int(
                await conn.fetchval(
                    """
                    SELECT count(*) FROM events
                    WHERE session_id = $1
                      AND kind = 'span'
                      AND data->>'event' = $2
                      AND data->>'wake_source_session_id' = $3
                      AND created_at > now() - interval '1 hour'
                    """,
                    target.id,
                    WAKE_LINEAGE_SPAN_EVENT,
                    source.id,
                )
                or 0
            )
        assert in_window == WAKE_SESSION_MAX_PER_HOUR
        ok = [r for r in results if isinstance(r, dict) and r.get("woken") is True]
        refused = [r for r in results if isinstance(r, WakeSessionRateLimitedError)]
        assert len(ok) == WAKE_SESSION_MAX_PER_HOUR
        assert len(refused) == burst - WAKE_SESSION_MAX_PER_HOUR
        assert recent == 2 * WAKE_SESSION_MAX_PER_HOUR  # old (backdated) + new

    async def test_one_connection_pool_does_not_deadlock(
        self,
        migrated_db_url: str,
        pool_with_runtime: asyncpg.Pool[Any],
        patched_defer_wake: AsyncMock,
    ) -> None:
        """A ``max_size=1`` pool cannot deadlock ``deliver_cross_session_wake``:
        the locked count+append is a short DB-only transaction that releases
        its connection on COMMIT, and the two ``pool.acquire()`` blocks (target
        load, then locked append txn) are sequential, never nested.  Mirrors
        ``test_outbound_tool_quota_db.py::...::test_one_connection_pool_cannot_deadlock``.
        """
        seed_pool = pool_with_runtime
        await _seed_account(seed_pool, "acc_wake_tiny", "wake-test-tiny-pool")
        _, _, source = await seed_agent_env_session(
            seed_pool, account_id="acc_wake_tiny", prefix="src"
        )
        _, _, target = await seed_agent_env_session(
            seed_pool, account_id="acc_wake_tiny", prefix="dst"
        )

        tiny = await create_pool(migrated_db_url, min_size=1, max_size=1)
        try:
            async with asyncio.timeout(10):
                # First wake delivers (depth 1) and releases its connection.
                first = await deliver_cross_session_wake(
                    tiny,
                    target_session_id=target.id,
                    content="first",
                    account_id="acc_wake_tiny",
                    root=CrossSessionWakeRoot(source_id=source.id, source_depth=0),
                    cause="agent_wake",
                )
                assert first == 1
                # The "post-commit lifecycle" step acquires from the SAME
                # exhausted-size pool AFTER the locked txn released — exactly
                # the sequence that would deadlock if the txn held across it.
                async with tiny.acquire() as conn:
                    await conn.fetchval("SELECT 1")
                # Fill the remaining per-pair budget sequentially.
                for i in range(WAKE_SESSION_MAX_PER_HOUR - 1):
                    await deliver_cross_session_wake(
                        tiny,
                        target_session_id=target.id,
                        content=f"fill-{i}",
                        account_id="acc_wake_tiny",
                        root=CrossSessionWakeRoot(source_id=source.id, source_depth=0),
                        cause="agent_wake",
                    )
                # The next sequential call is rate-limited (rolls back, releases
                # the connection + the advisory lock).
                with pytest.raises(WakeSessionRateLimitedError):
                    await deliver_cross_session_wake(
                        tiny,
                        target_session_id=target.id,
                        content="over",
                        account_id="acc_wake_tiny",
                        root=CrossSessionWakeRoot(source_id=source.id, source_depth=0),
                        cause="agent_wake",
                    )
                # A subsequent call still completes after the refusal — the
                # lock was released by the rollback, no stranded holder.
                async with tiny.acquire() as conn:
                    await conn.fetchval("SELECT 1")
        finally:
            await tiny.close()
        delivered = await _count_lineage_spans(seed_pool, target.id, source_id=source.id)
        assert delivered == WAKE_SESSION_MAX_PER_HOUR

    async def test_trigger_action_path_caps_under_concurrent_burst(
        self,
        pool_with_runtime: asyncpg.Pool[Any],
        patched_defer_wake: AsyncMock,
    ) -> None:
        """The ``wake_session`` TRIGGER ACTION routes through the SAME
        ``deliver_cross_session_wake`` with the FIRING TRIGGER as the lineage
        root (``source_id=f"trigger:{trigger_id}"``).  Its per-``(trigger,
        target)`` cap must hold under a concurrent burst of fires for the SAME
        reason the tool path does — the per-pair advisory lock keys on
        ``"trigger:<id>:<target>"`` — with no trigger-runner change.

        Calls ``_run_wake_session`` directly (the runner's dispatch arm) with
        a real ``TriggerRow`` + ``WakeSessionAction``; ``runtime.require_pool``
        sees the fixture's pool, and ``defer_wake`` is patched so no job is
        enqueued.          The runner maps ``WakeSessionRateLimitedError`` to
        ``status="error"``; we assert exactly ``cap`` ``ok`` fires and the
        rest ``error``, and exactly ``cap`` lineage spans stamped with the
        trigger root land on the target.
        """
        pool = pool_with_runtime
        await _seed_account(pool, "acc_wake_trig", "wake-test-trigger")
        _, _, target = await seed_agent_env_session(pool, account_id="acc_wake_trig", prefix="dst")

        trigger = queries.TriggerRow(
            id="trig_01WAKE",
            owner_session_id="sess_trigger_owner_unused",
            account_id="acc_wake_trig",
            name="watchdog",
            source="cron",
            source_spec={},
            action=WakeSessionAction(target_session_id=target.id, content="fire"),
            enabled=True,
            next_fire=None,
            running_since=None,
            last_fire_at=None,
            last_fire_status=None,
            consecutive_failures=0,
            environment_id=None,
            ingest_token_hash=None,
            session_archived_at=None,
            session_parent_run_id=None,
        )
        trigger_source_id = "trigger:trig_01WAKE"
        action = trigger.action
        assert isinstance(action, WakeSessionAction)

        burst = 30
        results = await asyncio.gather(*(_run_wake_session(trigger, action) for _ in range(burst)))

        delivered = await _count_lineage_spans(pool, target.id, source_id=trigger_source_id)
        # The per-(trigger, target) cap holds exactly under concurrency.
        assert delivered == WAKE_SESSION_MAX_PER_HOUR
        ok = [r for r in results if r[0] == "ok"]
        errored = [r for r in results if r[0] == "error"]
        assert len(ok) == WAKE_SESSION_MAX_PER_HOUR
        assert len(errored) == burst - WAKE_SESSION_MAX_PER_HOUR
        # Every error is a rate-limit mapping, not a silent drop.
        assert all("WakeSessionRateLimitedError" in (msg or "") for _, msg, _ in errored)
        # defer_wake called once per delivered wake (the runner does not defer
        # on error — deliver_cross_session_wake raised before reaching it).
        assert patched_defer_wake.await_count == WAKE_SESSION_MAX_PER_HOUR
