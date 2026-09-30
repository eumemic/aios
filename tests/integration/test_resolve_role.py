"""Integration coverage for role resolution (#1940) against a real Postgres.

``resolve_role`` maps a role to the LIVE agent of the caller's account: an agent
whose ``name`` equals the role or whose ``metadata.role`` equals it. Archived
agents never match (a re-spawn archives the old id and the role follows the new
one), other accounts never match, and zero / many matches fail loud.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db.pool import create_pool
from aios.errors import ConflictError, NotFoundError
from aios.harness import runtime
from aios.models.agents import Agent
from aios.services import agents as agents_service

pytestmark = pytest.mark.integration

ACC = "acc_role"
OTHER = "acc_role_other"


@pytest.fixture
async def pool(migrated_db_url: str, _reset_db_state: None) -> AsyncIterator[asyncpg.Pool[Any]]:
    p = await create_pool(migrated_db_url, min_size=1, max_size=4)
    prev = runtime.pool
    runtime.pool = p
    try:
        async with p.acquire() as conn:
            for acc, parent in ((ACC, None), (OTHER, ACC)):
                await conn.execute(
                    "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                    "VALUES ($1, $2, TRUE, $1)",
                    acc,
                    parent,
                )
        yield p
    finally:
        runtime.pool = prev
        await p.close()


async def _make(
    pool: asyncpg.Pool[Any], name: str, *, account_id: str = ACC, role: str | None = None
) -> Agent:
    return await agents_service.create_agent(
        pool,
        account_id=account_id,
        name=name,
        model="test/dummy",
        system="x",
        tools=[],
        mcp_servers=None,
        http_servers=None,
        description=None,
        metadata={"role": role} if role is not None else {},
        window_min=1000,
        window_max=100000,
    )


async def test_resolves_by_name(pool: asyncpg.Pool[Any]) -> None:
    agent = await _make(pool, "ops-agent")
    resolved = await agents_service.resolve_role(pool, "ops-agent", account_id=ACC)
    assert resolved.id == agent.id


async def test_resolves_by_metadata_role(pool: asyncpg.Pool[Any]) -> None:
    agent = await _make(pool, "ops-agent-v7", role="ops")
    resolved = await agents_service.resolve_role(pool, "ops", account_id=ACC)
    assert resolved.id == agent.id


async def test_agent_matching_by_both_name_and_metadata_is_one_candidate(
    pool: asyncpg.Pool[Any],
) -> None:
    agent = await _make(pool, "ops", role="ops")
    resolved = await agents_service.resolve_role(pool, "ops", account_id=ACC)
    assert resolved.id == agent.id


async def test_respawn_follows_the_live_agent(pool: asyncpg.Pool[Any]) -> None:
    old = await _make(pool, "ops-agent", role="ops")
    await agents_service.archive_agent(pool, old.id, account_id=ACC)
    new = await _make(pool, "ops-agent", role="ops")
    assert new.id != old.id
    assert (await agents_service.resolve_role(pool, "ops", account_id=ACC)).id == new.id
    assert (await agents_service.resolve_role(pool, "ops-agent", account_id=ACC)).id == new.id


async def test_unbound_role_is_loud(pool: asyncpg.Pool[Any]) -> None:
    await _make(pool, "someone-else")
    with pytest.raises(NotFoundError, match="no live binding for role 'ops'"):
        await agents_service.resolve_role(pool, "ops", account_id=ACC)


async def test_archived_only_binding_is_loud(pool: asyncpg.Pool[Any]) -> None:
    old = await _make(pool, "ops-agent", role="ops")
    await agents_service.archive_agent(pool, old.id, account_id=ACC)
    with pytest.raises(NotFoundError, match="no live binding"):
        await agents_service.resolve_role(pool, "ops", account_id=ACC)


async def test_other_account_never_resolves(pool: asyncpg.Pool[Any]) -> None:
    await _make(pool, "ops-agent", account_id=OTHER, role="ops")
    with pytest.raises(NotFoundError):
        await agents_service.resolve_role(pool, "ops", account_id=ACC)
    with pytest.raises(NotFoundError):
        await agents_service.resolve_role(pool, "ops-agent", account_id=ACC)


async def test_two_live_holders_is_loud_conflict(pool: asyncpg.Pool[Any]) -> None:
    a = await _make(pool, "ops-a", role="ops")
    b = await _make(pool, "ops-b", role="ops")
    with pytest.raises(ConflictError, match="ambiguous") as exc:
        await agents_service.resolve_role(pool, "ops", account_id=ACC)
    assert exc.value.detail is not None
    assert set(exc.value.detail["agent_ids"]) == {a.id, b.id}


async def test_name_and_metadata_on_different_agents_is_conflict(
    pool: asyncpg.Pool[Any],
) -> None:
    await _make(pool, "ops")
    await _make(pool, "ops-v2", role="ops")
    with pytest.raises(ConflictError):
        await agents_service.resolve_role(pool, "ops", account_id=ACC)
