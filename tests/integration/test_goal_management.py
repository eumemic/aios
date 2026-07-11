"""Integration tests for the explicit goal-management builtins (#1508, #1512, #1518).

DB-backed (testcontainer Postgres). These drive the REAL service/query path —
``create_goal`` opening a self-referential awaited obligation via
``sessions_service.invoke`` (#1414 self-goal) carrying the REQUIRED ``output_schema``
on its ``request_opened`` frame, and the general ``return`` / ``error`` answer verbs
writing the ``request_response`` via ``respond_to_request`` — and assert the
acceptance criteria against the same open-obligation queries the quiescence guard
and the obligations tail block read:

* ``create_goal`` opens an obligation that lands in the session's OPEN set
  (``het_open_request_ids`` / ``het_open_obligations``) as a ``self`` caller — so
  the quiescence guard holds the session (it cannot go idle) until it's closed —
  and persists its ``output_schema`` on the trusted ``request_opened` edge
  (``get_request_output_schema``), the same way ``call_*`` carry it (#1512);
* ``list_obligations`` enumerates open self-goals through the general obligations view;
* a self-goal is closed through the general source-agnostic verbs (#1518: the
  self-only ``complete_goal``/``fail_goal`` are retired). ``return`` validates its
  ``value`` against that persisted schema servicer-side — a conforming value drains
  the open set; a non-conforming one is rejected with ``output_schema_violation`` and
  the obligation stays open (#1512), so no validation from #1513 is lost;
* ``error`` abandons the goal, draining the open set;
* ``return``/``error`` still refuse a ``request_id`` that is not an open obligation
  of the session (unchanged behavior);
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
        # sessions_service.invoke → defer_wake; closing a self-goal with the general
        # return/error verbs → respond_to_request → defer_wake/defer_run_wake). With
        # no live worker
        # the procrastinate app pool is never opened, so the deferrals are
        # patched out — matching the model-task-tools integration fixture.
        with (
            mock.patch("aios.services.sessions.defer_wake", new=AsyncMock()),
            mock.patch("aios.jobs.app.defer_wake", new=AsyncMock()),
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
    # same way call_* carry output_schema (#1512) — `return` reads it from here.
    async with pool.acquire() as conn:
        persisted = await queries.get_request_output_schema(conn, session_id, request_id=goal_id)
    assert persisted == _SCHEMA


async def test_list_obligations_enumerates_open_self_goals(
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

    out = await invoke_builtin(session_id, "list_obligations", {})
    assert isinstance(out, dict)
    # The general obligations view can also contain the fixture's incoming API
    # request, so select the self-goal rows rather than assuming every row is a goal.
    rows = [row for row in out["obligations"] if row["origin"] == "self"]
    assert [row["request_id"] for row in rows] == [g1, g2]  # oldest-first
    assert rows[0]["summary"] == "goal one"


async def test_return_closes_self_goal_with_schema_gate(
    pool_session: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """#1518: a self-goal is closed with the general ``return`` verb (no
    ``complete_goal``). ``return`` validates its ``value`` against the goal's
    persisted ``output_schema`` SERVICER-SIDE — a non-conforming value is rejected
    with ``output_schema_violation`` and the goal stays open (no validation from
    #1513 is lost), a conforming value drains the open set so the session may quiesce.
    """
    pool, _account, session_id = pool_session
    goal_id = _gid(
        await invoke_builtin(session_id, "create_goal", {"goal": "do it", "output_schema": _SCHEMA})
    )
    assert await _open_ids(pool, session_id) == [goal_id]

    # A non-conforming value is rejected (output_schema_violation) — the goal stays open.
    bad = await invoke_builtin(
        session_id, "return", {"request_id": goal_id, "value": {"shipped": "nope"}}
    )
    assert isinstance(bad, ToolResult)
    assert bad.is_error
    assert isinstance(bad.content, str)
    assert "output_schema_violation" in bad.content
    assert await _open_ids(pool, session_id) == [goal_id]  # still owed

    # A conforming value closes it.
    out = await invoke_builtin(
        session_id, "return", {"request_id": goal_id, "value": {"shipped": True}}
    )
    assert isinstance(out, dict)
    assert out["status"] == "returned"
    # Obligation closed → open set empty → quiescence guard no longer holds.
    assert await _open_ids(pool, session_id) == []


async def test_error_abandons_self_goal(
    pool_session: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """#1518: a self-goal is abandoned with the general ``error`` verb (no
    ``fail_goal``), draining the open set."""
    pool, _account, session_id = pool_session
    goal_id = _gid(
        await invoke_builtin(session_id, "create_goal", {"goal": "do it", "output_schema": _SCHEMA})
    )
    assert await _open_ids(pool, session_id) == [goal_id]

    out = await invoke_builtin(
        session_id, "error", {"request_id": goal_id, "message": "infeasible"}
    )
    assert isinstance(out, dict)
    assert out["status"] == "errored"
    assert await _open_ids(pool, session_id) == []


async def test_return_refuses_unknown_request_id(
    pool_session: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """#1518 acceptance: ``return``/``error`` still refuse a ``request_id`` that is
    not an open obligation of the session — unchanged behavior."""
    pool, _account, session_id = pool_session
    goal_id = _gid(
        await invoke_builtin(session_id, "create_goal", {"goal": "do it", "output_schema": _SCHEMA})
    )

    out = await invoke_builtin(
        session_id, "return", {"request_id": "req_does_not_exist", "value": {"shipped": True}}
    )
    assert isinstance(out, ToolResult)
    assert out.is_error
    # The real open goal is untouched by the bogus answer.
    assert await _open_ids(pool, session_id) == [goal_id]

    err = await invoke_builtin(
        session_id, "error", {"request_id": "req_does_not_exist", "message": "nope"}
    )
    assert isinstance(err, ToolResult)
    assert err.is_error
    assert await _open_ids(pool, session_id) == [goal_id]


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

    # Freeing a slot re-admits (concurrency cap, not a lifetime budget). The goal is
    # closed through the general `return` verb (#1518: complete_goal is retired).
    with mock.patch("aios.tools.goal_management.get_settings") as gs:
        gs.return_value = mock.Mock(session_open_goals_max=2)
        await invoke_builtin(
            session_id, "return", {"request_id": a["goal_id"], "value": {"shipped": True}}
        )
        c = await invoke_builtin(
            session_id, "create_goal", {"goal": "g4", "output_schema": _SCHEMA}
        )
        assert isinstance(c, dict)
        assert "goal_id" in c
    # Sanity: the global default cap is comfortably above 0.
    assert cap >= 1
