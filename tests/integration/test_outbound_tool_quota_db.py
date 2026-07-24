"""Real-Postgres regressions for the durable outbound tool quota (#1903).

Covers the design invariants the redesign review required, against the real
``outbound_tool_reservations`` table and real asyncpg pools:

* atomic count+insert: a simultaneous same-key burst dispatches at most the
  configured capacity, regardless of interleaving;
* a one-connection pool: admission uses a short DB-only transaction and
  releases its connection before any external I/O, so even ``max_size=1``
  cannot deadlock;
* refusals consume nothing: denied retries never extend the rolling lockout;
* rolling expiry: capacity returns exactly when admitted rows age past the
  window, and the next call dispatches;
* conservative accounting: cancellation/crash or local publication failure
  AFTER admission leaves the reservation counting (a possible external side
  effect must not be re-runnable for free), while a refusal path before
  admission consumes nothing;
* canonical-verb keying: MCP-qualified sibling effectors share one quota.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import asyncpg
import pytest

from aios.db.pool import create_pool
from aios.services import outbound_tool_quota
from aios.services.outbound_tool_quota import (
    mark_outbound_dispatch_completed,
    reserve_outbound_tool_quota,
)
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration

_ACCOUNT = "acc_outbound_quota"


@pytest.fixture
async def quota_pool(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str]]:
    """Yield ``(pool, session_id)`` with a seeded live session."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ($1, NULL, TRUE, $2)",
                _ACCOUNT,
                "outbound-quota-test",
            )
        _agent, _env, session = await seed_agent_env_session(
            pool, account_id=_ACCOUNT, prefix="outbound-quota"
        )
        yield pool, session.id
    finally:
        await pool.close()


def _set_quota(monkeypatch: Any, quotas: dict[str, tuple[int, int]]) -> None:
    monkeypatch.setattr(
        outbound_tool_quota,
        "get_settings",
        lambda: SimpleNamespace(outbound_tool_quotas=quotas),
    )


async def _age_reservations(
    pool: asyncpg.Pool[Any], session_id: str, verb: str, seconds: int
) -> None:
    """Backdate every reservation for the key, simulating window passage."""
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE outbound_tool_reservations "
            "SET created_at = created_at - make_interval(secs => $3::bigint) "
            "WHERE session_id = $1 AND verb = $2",
            session_id,
            verb,
            seconds,
        )


class TestAtomicAdmission:
    async def test_simultaneous_burst_cannot_exceed_capacity(
        self, quota_pool: tuple[asyncpg.Pool[Any], str], monkeypatch: Any
    ) -> None:
        """The concurrency regression the review demanded: N tasks admitted
        simultaneously from an empty window dispatch at most ``max_per_window``
        — in-flight admissions are durable rows, visible to every later
        count under the per-key advisory lock."""
        pool, sid = quota_pool
        cap = 3
        burst = 10
        _set_quota(monkeypatch, {"matrix_send": (3600, cap)})

        results = await asyncio.gather(
            *(reserve_outbound_tool_quota(pool, sid, "matrix_send") for _ in range(burst))
        )

        admitted = [r for r in results if r.reservation_id is not None]
        refused = [r for r in results if r.refusal is not None]
        assert len(admitted) == cap
        assert len(refused) == burst - cap
        async with pool.acquire() as conn:
            rows = await conn.fetchval(
                "SELECT count(*) FROM outbound_tool_reservations "
                "WHERE session_id = $1 AND verb = 'matrix_send'",
                sid,
            )
        assert rows == cap

    async def test_one_connection_pool_cannot_deadlock(
        self, migrated_db_url: str, quota_pool: tuple[asyncpg.Pool[Any], str], monkeypatch: Any
    ) -> None:
        """Admission on a ``max_size=1`` pool completes: the reservation is a
        short DB-only transaction that never nests a second acquisition and
        releases its connection before the caller's I/O/publication."""
        _pool, sid = quota_pool
        _set_quota(monkeypatch, {"matrix_send": (3600, 2)})
        tiny = await create_pool(migrated_db_url, min_size=1, max_size=1)
        try:
            async with asyncio.timeout(10):
                first = await reserve_outbound_tool_quota(tiny, sid, "matrix_send")
                assert first.reservation_id is not None
                # The "lifecycle publication" acquires from the SAME
                # exhausted-size pool AFTER admission — this is exactly the
                # sequence that deadlocked the advisory-lock-across-I/O
                # design. It must succeed now.
                async with tiny.acquire() as conn:
                    await conn.fetchval("SELECT 1")
                await mark_outbound_dispatch_completed(tiny, str(first.reservation_id))
                second = await reserve_outbound_tool_quota(tiny, sid, "matrix_send")
                assert second.reservation_id is not None
                third = await reserve_outbound_tool_quota(tiny, sid, "matrix_send")
                assert third.refusal is not None
        finally:
            await tiny.close()

    async def test_contended_one_slot_pool_with_concurrent_waiters(
        self, migrated_db_url: str, quota_pool: tuple[asyncpg.Pool[Any], str], monkeypatch: Any
    ) -> None:
        """Same-key concurrent admissions through a one-slot pool all resolve
        (serialized), rather than waiters starving the holder."""
        _pool, sid = quota_pool
        _set_quota(monkeypatch, {"matrix_send": (3600, 2)})
        tiny = await create_pool(migrated_db_url, min_size=1, max_size=1)
        try:
            async with asyncio.timeout(15):
                results = await asyncio.gather(
                    *(reserve_outbound_tool_quota(tiny, sid, "matrix_send") for _ in range(5))
                )
        finally:
            await tiny.close()
        assert sum(r.reservation_id is not None for r in results) == 2
        assert sum(r.refusal is not None for r in results) == 3


class TestRollingWindowAccounting:
    async def test_denied_retries_do_not_extend_lockout_and_window_rolls(
        self, quota_pool: tuple[asyncpg.Pool[Any], str], monkeypatch: Any
    ) -> None:
        """The rolling-window regression the review demanded: reach the cap
        with successful dispatches, hammer denied retries, advance past the
        original dispatches' window — the next call dispatches regardless of
        the refusals (they inserted nothing)."""
        pool, sid = quota_pool
        window = 60
        _set_quota(monkeypatch, {"matrix_invite": (window, 2)})

        for _ in range(2):
            admission = await reserve_outbound_tool_quota(pool, sid, "matrix_invite")
            assert admission.reservation_id is not None

        for _ in range(5):  # denied retries while capped
            retry = await reserve_outbound_tool_quota(pool, sid, "matrix_invite")
            assert retry.refusal == "quota_exceeded: matrix_invite 2/2 per minute"

        await _age_reservations(pool, sid, "matrix_invite", window + 1)

        after = await reserve_outbound_tool_quota(pool, sid, "matrix_invite")
        assert after.refusal is None
        assert after.reservation_id is not None

    async def test_admitted_capacity_survives_publication_failure(
        self, quota_pool: tuple[asyncpg.Pool[Any], str], monkeypatch: Any
    ) -> None:
        """Post-admission failure (connector outcome unknown / local
        publication failed / crash) must CONSERVATIVELY retain the
        reservation: the external side effect may have happened, so a retry
        cannot exceed the cap by replaying for free."""
        pool, sid = quota_pool
        _set_quota(monkeypatch, {"matrix_send": (3600, 1)})

        admission = await reserve_outbound_tool_quota(pool, sid, "matrix_send")
        assert admission.reservation_id is not None
        # Simulate: connector I/O ran, then result/end-span publication blew
        # up (or the worker died). No completion mark, no cleanup call.
        retry = await reserve_outbound_tool_quota(pool, sid, "matrix_send")
        assert retry.refusal == "quota_exceeded: matrix_send 1/1 per hour"
        # The orphaned row needs no repair step: it ages out with the window.
        await _age_reservations(pool, sid, "matrix_send", 3601)
        recovered = await reserve_outbound_tool_quota(pool, sid, "matrix_send")
        assert recovered.reservation_id is not None

    async def test_completion_mark_is_observability_not_accounting(
        self, quota_pool: tuple[asyncpg.Pool[Any], str], monkeypatch: Any
    ) -> None:
        pool, sid = quota_pool
        _set_quota(monkeypatch, {"matrix_send": (3600, 1)})

        admission = await reserve_outbound_tool_quota(pool, sid, "matrix_send")
        assert admission.reservation_id is not None
        await mark_outbound_dispatch_completed(pool, str(admission.reservation_id))
        async with pool.acquire() as conn:
            state = await conn.fetchval(
                "SELECT state FROM outbound_tool_reservations WHERE id = $1",
                admission.reservation_id,
            )
        assert state == "completed"
        # Completed rows still count: capacity was consumed at admission.
        retry = await reserve_outbound_tool_quota(pool, sid, "matrix_send")
        assert retry.refusal is not None


class TestCanonicalVerbKeying:
    async def test_mcp_siblings_share_one_quota(
        self, quota_pool: tuple[asyncpg.Pool[Any], str], monkeypatch: Any
    ) -> None:
        """``mcp__a__matrix_send`` and ``mcp__b__matrix_send`` draw from the
        same canonical ``matrix_send`` capacity — no per-server bypass."""
        pool, sid = quota_pool
        _set_quota(monkeypatch, {"matrix_send": (3600, 2)})

        a = await reserve_outbound_tool_quota(pool, sid, "mcp__server_a__matrix_send")
        b = await reserve_outbound_tool_quota(pool, sid, "mcp__server_b__matrix_send")
        assert a.reservation_id is not None and b.reservation_id is not None
        third = await reserve_outbound_tool_quota(pool, sid, "matrix_send")
        assert third.refusal == "quota_exceeded: matrix_send 2/2 per hour"

    async def test_sessions_are_isolated(
        self, quota_pool: tuple[asyncpg.Pool[Any], str], monkeypatch: Any
    ) -> None:
        pool, sid = quota_pool
        _set_quota(monkeypatch, {"matrix_send": (3600, 1)})
        _agent, _env, other = await seed_agent_env_session(
            pool, account_id=_ACCOUNT, prefix="outbound-quota-b"
        )

        first = await reserve_outbound_tool_quota(pool, sid, "matrix_send")
        assert first.reservation_id is not None
        capped = await reserve_outbound_tool_quota(pool, sid, "matrix_send")
        assert capped.refusal is not None
        other_session = await reserve_outbound_tool_quota(pool, other.id, "matrix_send")
        assert other_session.reservation_id is not None
