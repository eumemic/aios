"""Integration tests for the explicit goal-management builtins (#1508, #1512).

DB-backed (testcontainer Postgres). These drive the REAL service/query path —
``create_goal`` opening a self-referential awaited obligation via
``sessions_service.invoke`` (#1414 self-goal) carrying the REQUIRED ``output_schema``
on its ``request_opened`` frame, and ``complete_goal`` / ``fail_goal`` writing the
``request_response`` half via ``respond_to_request`` — and assert the acceptance
criteria against the same open-obligation queries the quiescence guard and the
obligations tail block read:

* ``create_goal`` opens an obligation that lands in the session's OPEN set
  (``get_open_request_ids`` / ``get_open_obligations``) as a ``self`` caller — so
  the quiescence guard holds the session (it cannot go idle) until it's closed —
  and persists its ``output_schema`` on the trusted ``request_opened`` edge
  (``get_request_output_schema``), the same way ``call_*`` carry it (#1512);
* ``list_goals`` enumerates exactly the open self-goals;
* ``complete_goal`` validates its ``result`` against that persisted schema — a
  conforming result drains the open set; a non-conforming one is rejected with
  ``output_schema_violation`` and the obligation stays open (#1512);
* ``fail_goal`` emits the ``request_response`` half, draining the open set;
* the per-session open-goal admission cap is enforced with a clear error.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock

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

# The completion contract a goal pins up front (#1512) — a representative output_schema.
_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"shipped": {"type": "boolean"}},
    "required": ["shipped"],
}


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
        # The goal handlers drive the real service path (create_goal →
        # sessions_service.invoke → defer_wake; complete_goal/fail_goal →
        # respond_to_request → defer_wake/defer_run_wake). With no live worker
        # the procrastinate app pool is never opened, so the deferrals are
        # patched out — matching the model-task-tools integration fixture.
        with (
            mock.patch("aios.services.wake.defer_wake", new=AsyncMock()),
            mock.patch("aios.services.workflows.defer_run_wake", new=AsyncMock()),
        ):
            yield pool, _ACCOUNT, session.id
    finally:
        runtime.pool = prev_pool
        runtime.crypto_box = prev_box
        await pool.close()


def _gid(out: ToolResult | dict[str, Any]) -> str:
    """Narrow a successful handler result and return its ``goal_id``."""
    assert isinstance(out, dict)
    goal_id = out["goal_id"]
    assert isinstance(goal_id, str)
    return goal_id


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
        session_id, "create_goal", {"goal": "ship the feature", "output_schema": _SCHEMA}
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
    # The completion contract is persisted on the trusted request_opened edge, the
    # same way call_* carry output_schema (#1512) — complete_goal reads it from here.
    async with pool.acquire() as conn:
        persisted = await queries.get_request_output_schema(conn, session_id, request_id=goal_id)
    assert persisted == _SCHEMA


async def test_list_goals_enumerates_open_self_goals(
    pool_session: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    _pool, _account, session_id = pool_session
    g1 = _gid(
        await invoke_builtin(
            session_id, "create_goal", {"goal": "goal one", "output_schema": _SCHEMA}
        )
    )
    g2 = _gid(
        await invoke_builtin(
            session_id, "create_goal", {"goal": "goal two", "output_schema": _SCHEMA}
        )
    )

    out = await invoke_builtin(session_id, "list_goals", {})
    assert isinstance(out, dict)
    ids = [g["goal_id"] for g in out["goals"]]
    assert ids == [g1, g2]  # oldest-first
    assert out["goals"][0]["goal"]  # carries the summary text


async def test_complete_goal_drains_open_set(
    pool_session: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """complete_goal validates result against the goal's output_schema, then emits
    the request_response half so the session may quiesce (#1512)."""
    pool, _account, session_id = pool_session
    goal_id = _gid(
        await invoke_builtin(session_id, "create_goal", {"goal": "do it", "output_schema": _SCHEMA})
    )
    assert await _open_ids(pool, session_id) == [goal_id]

    # A non-conforming result is rejected (output_schema_violation) — the goal stays open.
    bad = await invoke_builtin(
        session_id, "complete_goal", {"goal_id": goal_id, "result": {"shipped": "nope"}}
    )
    assert isinstance(bad, ToolResult)
    assert bad.is_error
    assert isinstance(bad.content, str)
    assert "output_schema_violation" in bad.content
    assert await _open_ids(pool, session_id) == [goal_id]  # still owed

    # A conforming result closes it.
    out = await invoke_builtin(
        session_id, "complete_goal", {"goal_id": goal_id, "result": {"shipped": True}}
    )
    assert out == {"goal_id": goal_id, "status": "completed"}
    # Obligation closed → open set empty → quiescence guard no longer holds.
    assert await _open_ids(pool, session_id) == []


async def test_fail_goal_drains_open_set(
    pool_session: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    pool, _account, session_id = pool_session
    goal_id = _gid(
        await invoke_builtin(session_id, "create_goal", {"goal": "do it", "output_schema": _SCHEMA})
    )
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
        a = await invoke_builtin(
            session_id, "create_goal", {"goal": "g1", "output_schema": _SCHEMA}
        )
        b = await invoke_builtin(
            session_id, "create_goal", {"goal": "g2", "output_schema": _SCHEMA}
        )
        assert isinstance(a, dict) and isinstance(b, dict)
        assert "goal_id" in a and "goal_id" in b
        over = await invoke_builtin(
            session_id, "create_goal", {"goal": "g3", "output_schema": _SCHEMA}
        )
        assert isinstance(over, ToolResult)
        assert over.is_error
        assert isinstance(over.content, str)
        assert "cap" in over.content.lower()
        # No third obligation opened.
        assert len(await _open_ids(pool, session_id)) == 2

    # Freeing a slot re-admits (concurrency cap, not a lifetime budget).
    with mock.patch("aios.tools.goal_management.get_settings") as gs:
        gs.return_value = mock.Mock(session_open_goals_max=2)
        await invoke_builtin(
            session_id, "complete_goal", {"goal_id": a["goal_id"], "result": {"shipped": True}}
        )
        c = await invoke_builtin(
            session_id, "create_goal", {"goal": "g4", "output_schema": _SCHEMA}
        )
        assert isinstance(c, dict)
        assert "goal_id" in c
    # Sanity: the global default cap is comfortably above 0.
    assert cap >= 1
