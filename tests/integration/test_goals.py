"""Integration tests for self-issued goals — ``set_goal`` / ``cancel_goal`` (#1414).

DB-backed (testcontainer Postgres). Exercises the service layer end to end against
real session + event rows:

* ``set_goal`` opens a self-edge (``caller == servicer == session``) that shows up
  in ``get_open_request_ids`` and counts as a self-goal.
* the edge write is **idempotent** on the deterministic ``request_id`` — a retried
  call with the same id re-opens the SAME goal exactly once (no phantom double).
* the open-goal **cap** rejects past ``open_goals_per_session_max``.
* ``cancel_goal_response`` retracts a self-goal (``{kind:cancelled}``) and refuses
  a non-self obligation BEFORE any write.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.config import get_settings
from aios.db import queries
from aios.db.pool import create_pool
from aios.services import sessions as service
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration

_ACCOUNT = "acc_goals"
_OTHER_ACCOUNT = "acc_goals_other"


@pytest.fixture
async def pool_env(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str]]:
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            for acct in (_ACCOUNT, _OTHER_ACCOUNT):
                await conn.execute(
                    "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                    "VALUES ($1, NULL, TRUE, 'goals-test')",
                    acct,
                )
        yield pool, _ACCOUNT
    finally:
        await pool.close()


async def test_set_goal_opens_self_edge(pool_env: tuple[asyncpg.Pool[Any], str]) -> None:
    pool, account_id = pool_env
    _a, _e, session = await seed_agent_env_session(pool, account_id=account_id, prefix="goal")
    rid = service.goal_request_id(session.id, "tc_1")

    goal_id = await service.set_goal(
        pool, session.id, account_id=account_id, request_id=rid, goal="keep shipping"
    )
    assert goal_id == rid

    async with pool.acquire() as conn:
        assert rid in await queries.get_open_request_ids(conn, session.id, account_id=account_id)
        caller = await queries.get_request_caller(conn, session.id, request_id=rid)
    # caller == servicer == this session: the reflexive self-edge.
    assert caller == {"kind": "session", "id": session.id}
    assert await service.count_open_self_goals(pool, session.id, account_id=account_id) == 1


async def test_set_goal_idempotent_on_deterministic_id(
    pool_env: tuple[asyncpg.Pool[Any], str],
) -> None:
    """A retried call with the SAME deterministic request_id re-opens one goal."""
    pool, account_id = pool_env
    _a, _e, session = await seed_agent_env_session(pool, account_id=account_id, prefix="idem")
    rid = service.goal_request_id(session.id, "tc_retry")

    await service.set_goal(pool, session.id, account_id=account_id, request_id=rid, goal="g")
    await service.set_goal(pool, session.id, account_id=account_id, request_id=rid, goal="g")

    # Exactly one open self-goal, exactly one request_opened edge.
    assert await service.count_open_self_goals(pool, session.id, account_id=account_id) == 1
    async with pool.acquire() as conn:
        n = await conn.fetchval(
            "SELECT count(*) FROM events WHERE session_id = $1 AND kind = 'lifecycle' "
            "AND data->>'event' = 'request_opened' AND data->>'request_id' = $2",
            session.id,
            rid,
        )
    assert n == 1


async def test_distinct_tool_calls_make_distinct_goals(
    pool_env: tuple[asyncpg.Pool[Any], str],
) -> None:
    pool, account_id = pool_env
    _a, _e, session = await seed_agent_env_session(pool, account_id=account_id, prefix="two")
    for tc in ("tc_a", "tc_b"):
        rid = service.goal_request_id(session.id, tc)
        await service.set_goal(pool, session.id, account_id=account_id, request_id=rid, goal="g")
    assert await service.count_open_self_goals(pool, session.id, account_id=account_id) == 2


async def test_set_goal_cap_rejects(
    pool_env: tuple[asyncpg.Pool[Any], str], monkeypatch: Any
) -> None:
    from aios.errors import RateLimitedError

    pool, account_id = pool_env
    _a, _e, session = await seed_agent_env_session(pool, account_id=account_id, prefix="cap")
    settings = get_settings()
    monkeypatch.setattr(settings, "open_goals_per_session_max", 2)

    for tc in ("tc_1", "tc_2"):
        rid = service.goal_request_id(session.id, tc)
        await service.set_goal(pool, session.id, account_id=account_id, request_id=rid, goal="g")

    with pytest.raises(RateLimitedError):
        rid3 = service.goal_request_id(session.id, "tc_3")
        await service.set_goal(pool, session.id, account_id=account_id, request_id=rid3, goal="g")


async def test_cap_frees_after_answer(
    pool_env: tuple[asyncpg.Pool[Any], str], monkeypatch: Any
) -> None:
    """Answering a goal frees a slot — the cap is concurrency, not a lifetime budget."""
    pool, account_id = pool_env
    _a, _e, session = await seed_agent_env_session(pool, account_id=account_id, prefix="free")
    settings = get_settings()
    monkeypatch.setattr(settings, "open_goals_per_session_max", 1)

    rid = service.goal_request_id(session.id, "tc_1")
    await service.set_goal(pool, session.id, account_id=account_id, request_id=rid, goal="g")
    async with pool.acquire() as conn:
        await queries.write_response_if_absent(
            conn,
            session.id,
            account_id=account_id,
            request_id=rid,
            is_error=False,
            result={"ok": True},
            error=None,
        )
    assert await service.count_open_self_goals(pool, session.id, account_id=account_id) == 0
    # A new goal now admits.
    rid2 = service.goal_request_id(session.id, "tc_2")
    await service.set_goal(pool, session.id, account_id=account_id, request_id=rid2, goal="g2")


async def test_cancel_goal_self_then_non_self(pool_env: tuple[asyncpg.Pool[Any], str]) -> None:
    from aios.tools.goals import cancel_goal_response

    pool, account_id = pool_env
    _a, _e, session = await seed_agent_env_session(pool, account_id=account_id, prefix="cancel")

    # A real self-goal cancels.
    rid = service.goal_request_id(session.id, "tc_self")
    await service.set_goal(pool, session.id, account_id=account_id, request_id=rid, goal="g")
    status = await cancel_goal_response(
        pool, session.id, account_id=account_id, goal_id=rid, by="self"
    )
    assert status == "cancelled"
    assert await service.count_open_self_goals(pool, session.id, account_id=account_id) == 0

    # A peer obligation (caller is a DIFFERENT session) is refused BEFORE any write.
    async with pool.acquire() as conn, conn.transaction():
        from aios.models.attenuation import surface_of
        from aios.services import agents as agents_service

        agent = await agents_service.load_for_session(pool, session, account_id=account_id)
        fs = surface_of(agent)
        await queries.append_request_opened(
            conn,
            session_id=session.id,
            account_id=account_id,
            request_id="req_peer",
            caller={"kind": "session", "id": "ses_other"},
            depth=0,
            environment_id=session.environment_id,
            frozen_surface={
                "tools": [t.model_dump() for t in fs.tools],
                "mcp_servers": [s.model_dump() for s in fs.mcp_servers],
                "http_servers": [s.model_dump() for s in fs.http_servers],
            },
            vault_ids=[],
            awaited=True,
        )
    status = await cancel_goal_response(
        pool, session.id, account_id=account_id, goal_id="req_peer", by="self"
    )
    assert status == "not_self_goal"
    # The peer obligation is still open (never stamped cancelled).
    async with pool.acquire() as conn:
        assert "req_peer" in await queries.get_open_request_ids(
            conn, session.id, account_id=account_id
        )
