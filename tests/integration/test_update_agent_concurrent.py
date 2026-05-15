"""Integration test: ``queries.update_agent`` serializes concurrent writers.

Pre-fix: the optimistic-concurrency check at ``queries.py:451`` ran
BEFORE the transaction. Two concurrent ``update_agent`` calls with
the same ``expected_version`` both read the same ``current.version``,
both passed the pre-check, both computed
``new_version = current.version + 1``, both started transactions, and
both ran the UPDATE (whose ``WHERE`` clause didn't include
``version = $expected_version``). The first writer's
``INSERT INTO agent_versions (agent_id, version)`` succeeded; the
second's hit ``agent_versions_pkey`` → ``UniqueViolationError`` → HTTP
500 to the loser.

The loser should see a clean ``ConflictError`` (HTTP 409) — same as if
they had been a slow writer arriving after a fast writer's version
had already bumped, just via the sequential pre-check.

Fix: add ``AND version = $expected_version`` to the UPDATE ``WHERE``
so it is the authoritative version check. UPDATE matching zero rows
raises ConflictError; the agent_versions INSERT only runs after a
successful UPDATE, so PK violations are unreachable on the race path.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db.pool import create_pool
from aios.errors import ConflictError
from aios.models.agents import Agent, ToolSpec
from aios.services import agents as agents_service

pytestmark = pytest.mark.integration


@pytest.fixture
async def pool_with_agent(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, Agent]]:
    """Yield ``(pool, account_id, agent)`` for an initialized agent at version 1."""
    pool = await create_pool(migrated_db_url, min_size=2, max_size=8)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_test', NULL, TRUE, 'tenant-test')
                """
            )
        agent = await agents_service.create_agent(
            pool,
            account_id="acc_test",
            name="concurrent-test",
            model="openrouter/test",
            system="initial",
            tools=[ToolSpec(type="bash")],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        yield pool, "acc_test", agent
    finally:
        await pool.close()


async def test_concurrent_updates_loser_gets_clean_conflict_error(
    pool_with_agent: tuple[asyncpg.Pool[Any], str, Agent],
) -> None:
    """Two concurrent ``update_agent`` calls with the same
    ``expected_version`` must produce: one success, one
    :class:`ConflictError`. The loser must NOT see an
    ``asyncpg.UniqueViolationError`` (which leaks as HTTP 500)."""
    pool, account_id, agent = pool_with_agent

    async def _attempt(new_system: str) -> tuple[bool, BaseException | None]:
        try:
            await agents_service.update_agent(
                pool,
                agent.id,
                account_id=account_id,
                expected_version=agent.version,
                system=new_system,
            )
            return True, None
        except BaseException as e:
            return False, e

    # Pool has min_size=2 so both coroutines can hold connections
    # concurrently — proves the race is real and not artificially
    # serialized by pool exhaustion.
    results = await asyncio.gather(_attempt("alpha"), _attempt("beta"))

    successes = [r for r in results if r[0]]
    failures = [r for r in results if not r[0]]

    assert len(successes) == 1, (
        f"expected exactly one success; got {len(successes)} "
        f"(failures: {[type(f[1]).__name__ for f in failures]})"
    )
    assert len(failures) == 1, (
        f"expected exactly one failure; got {len(failures)} (successes: {len(successes)})"
    )

    loser_exception = failures[0][1]
    assert isinstance(loser_exception, ConflictError), (
        f"loser got {type(loser_exception).__name__}: {loser_exception}; "
        f"the pre-transaction expected_version check is racy — both "
        f"writers pass it, then the second hits agent_versions_pkey "
        f"and leaks UniqueViolationError as HTTP 500 instead of 409"
    )
    # The detail must reflect post-race truth: the winner bumped version
    # to agent.version + 1, so the loser's ``current`` field is that.
    # (Without the in-transaction re-read this would still be agent.version,
    # which is stale by the time the conflict is raised.)
    assert loser_exception.detail["expected"] == agent.version
    assert loser_exception.detail["current"] == agent.version + 1, (
        f"detail.current should reflect the winner's bumped version "
        f"(agent.version+1={agent.version + 1}); got "
        f"{loser_exception.detail['current']!r}. Indicates the error "
        f"reports the pre-transaction value instead of the post-conflict "
        f"actual."
    )


async def test_sequential_stale_version_still_raises_conflict(
    pool_with_agent: tuple[asyncpg.Pool[Any], str, Agent],
) -> None:
    """Regression guard: the non-concurrent stale-version path must
    continue to raise ConflictError after the in-transaction check is
    added (the pre-check should still catch the obvious case)."""
    pool, account_id, agent = pool_with_agent

    # First update bumps version to 2.
    await agents_service.update_agent(
        pool,
        agent.id,
        account_id=account_id,
        expected_version=agent.version,
        system="bumped",
    )

    # Second update with the stale expected_version must raise.
    with pytest.raises(ConflictError) as exc_info:
        await agents_service.update_agent(
            pool,
            agent.id,
            account_id=account_id,
            expected_version=agent.version,  # stale (was 1, now 2)
            system="should-fail",
        )

    assert exc_info.value.detail["expected"] == agent.version
    assert exc_info.value.detail["current"] == agent.version + 1
