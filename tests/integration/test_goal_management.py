"""Integration tests for the explicit goal-management builtins (#1508).

DB-backed (testcontainer Postgres). These drive the REAL service/query path —
``create_goal`` opening a self-referential awaited obligation via
``sessions_service.invoke`` (#1414 self-goal), and ``complete_goal`` / ``fail_goal``
writing the ``request_response`` half via ``respond_to_request`` — and assert the
acceptance criteria against the same open-obligation queries the quiescence guard
and the obligations tail block read:

* ``create_goal`` opens an obligation that lands in the session's OPEN set
  (``get_open_request_ids`` / ``get_open_obligations``) as a ``self`` caller — so
  the quiescence guard holds the session (it cannot go idle) until it's closed;
* ``list_goals`` enumerates exactly the open self-goals;
* ``complete_goal`` / ``fail_goal`` emit the ``request_response`` half, draining the
  open set so the session may quiesce;
* the per-session open-goal admission cap is enforced with a clear error.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest import mock

import asyncpg
import pytest

from aios.config import get_settings
from aios.db import queries
from aios.db.pool import create_pool
from aios.harness import runtime
from aios.tools.invoke import invoke_builtin
from aios.tools.registry import ToolResult
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.asyncio

_ACCOUNT = "acc_goal_mgmt"


@pytest.fixture
async def pool_session(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str]]:
    """Yield ``(pool, account_id, session_id)`` with ``runtime.pool`` bound so the
    builtin handlers' ``runtime.require_pool()`` resolves to the test pool."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    prev_pool = runtime.pool
    prev_box = runtime.crypto_box
    runtime.pool = pool
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ($1, NULL, TRUE, 'goal-mgmt-test')",
                _ACCOUNT,
            )
        _agent, _env, session = await seed_agent_env_session(
            pool, account_id=_ACCOUNT, prefix="goal_seed"
        )
        yield pool, _ACCOUNT, session.id
    finally:
        runtime.pool = prev_pool
        runtime.crypto_box = prev_box
        await pool.close()


async def _open_ids(pool: asyncpg.Pool[Any], session_id: str) -> list[str]:
    async with pool.acquire() as conn:
        return await queries.get_open_request_ids(conn, session_id, account_id=_ACCOUNT)


async def _open_obligations(pool: asyncpg.Pool[Any], session_id: str) -> list[Any]:
    async with pool.acquire() as conn:
        return await queries.get_open_obligations(conn, session_id, account_id=_ACCOUNT)


async def test_create_goal_opens_holding_self_obligation(
    pool_session: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """create_goal opens a self-goal that lands in the OPEN set as a ``self`` caller —
    the obligation the quiescence guard reads to hold the session."""
    pool, _account, session_id = pool_session
    assert await _open_ids(pool, session_id) == []  # nothing owed yet

    out = await invoke_builtin(
        session_id, "create_goal", {"goal": "ship the feature", "acceptance_criteria": "tests pass"}
    )
    assert isinstance(out, dict)
    goal_id = out["goal_id"]

    # The session now owes exactly this obligation (the quiescence-guard open set).
    assert await _open_ids(pool, session_id) == [goal_id]
    obligations = await _open_obligations(pool, session_id)
    assert len(obligations) == 1
    ob = obligations[0]
    assert ob.request_id == goal_id
    # A self-goal: a ``session`` caller that is the session ITSELF (#1414).
    assert ob.caller_kind == "session"
    assert ob.caller_id == session_id


async def test_list_goals_enumerates_open_self_goals(
    pool_session: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    _pool, _account, session_id = pool_session
    g1 = (await invoke_builtin(session_id, "create_goal", {"goal": "goal one"}))["goal_id"]
    g2 = (await invoke_builtin(session_id, "create_goal", {"goal": "goal two"}))["goal_id"]

    out = await invoke_builtin(session_id, "list_goals", {})
    ids = [g["goal_id"] for g in out["goals"]]
    assert ids == [g1, g2]  # oldest-first
    assert out["goals"][0]["goal"]  # carries the summary text


async def test_complete_goal_drains_open_set(
    pool_session: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """complete_goal emits the request_response half so the session may quiesce."""
    pool, _account, session_id = pool_session
    goal_id = (await invoke_builtin(session_id, "create_goal", {"goal": "do it"}))["goal_id"]
    assert await _open_ids(pool, session_id) == [goal_id]

    out = await invoke_builtin(
        session_id, "complete_goal", {"goal_id": goal_id, "evidence": "verified"}
    )
    assert out == {"goal_id": goal_id, "status": "completed"}
    # Obligation closed → open set empty → quiescence guard no longer holds.
    assert await _open_ids(pool, session_id) == []


async def test_fail_goal_drains_open_set(
    pool_session: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    pool, _account, session_id = pool_session
    goal_id = (await invoke_builtin(session_id, "create_goal", {"goal": "do it"}))["goal_id"]
    assert await _open_ids(pool, session_id) == [goal_id]

    out = await invoke_builtin(
        session_id, "fail_goal", {"goal_id": goal_id, "reason": "infeasible"}
    )
    assert out == {"goal_id": goal_id, "status": "failed"}
    assert await _open_ids(pool, session_id) == []


async def test_open_goal_cap_enforced(
    pool_session: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    pool, _account, session_id = pool_session
    cap = get_settings().session_open_goals_max
    with mock.patch("aios.tools.goal_management.get_settings") as gs:
        gs.return_value = mock.Mock(session_open_goals_max=2)
        a = await invoke_builtin(session_id, "create_goal", {"goal": "g1"})
        b = await invoke_builtin(session_id, "create_goal", {"goal": "g2"})
        assert "goal_id" in a and "goal_id" in b
        over = await invoke_builtin(session_id, "create_goal", {"goal": "g3"})
        assert isinstance(over, ToolResult)
        assert over.is_error
        assert "cap" in over.content.lower()
        # No third obligation opened.
        assert len(await _open_ids(pool, session_id)) == 2

    # Freeing a slot re-admits (concurrency cap, not a lifetime budget).
    with mock.patch("aios.tools.goal_management.get_settings") as gs:
        gs.return_value = mock.Mock(session_open_goals_max=2)
        await invoke_builtin(session_id, "complete_goal", {"goal_id": a["goal_id"]})
        c = await invoke_builtin(session_id, "create_goal", {"goal": "g4"})
        assert "goal_id" in c
    # Sanity: the global default cap is comfortably above 0.
    assert cap >= 1
