"""DB-backed trace walk + reverse-index tests (#1149).

Exercises the children-of reverse lookup and the full ``get_trace`` walk against
real ``events`` / ``wf_runs`` / ``wf_run_events`` rows (testcontainer Postgres),
and pins that the new ``0103`` reverse indexes are actually used by the
children-of predicate (the EXPLAIN-asserting test the lock-direction-defer-DDL
plan called for).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

import asyncpg
import pytest

from aios.db.pool import create_pool
from aios.db.queries import trace as trace_q
from aios.db.queries import workflows as wf_queries
from aios.services import trace as trace_service
from aios.services import workflows as wf_service
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration

_ACCOUNT = "acc_trace_walk"


@pytest.fixture
async def pool_env(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str]]:
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ($1, NULL, TRUE, 'trace-walk-test')",
                _ACCOUNT,
            )
        _agent, env, _session = await seed_agent_env_session(
            pool, account_id=_ACCOUNT, prefix="trace-walk"
        )
        yield pool, _ACCOUNT, env.id
    finally:
        await pool.close()


async def _seed_run(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    environment_id: str,
    caller: dict[str, Any] | None = None,
    request_id: str | None = None,
) -> str:
    wf = await wf_service.create_workflow(
        pool,
        account_id=account_id,
        name=f"trace-walk-wf-{uuid4().hex}",
        script="def main(ctx):\n    return None\n",
        description=None,
        tools=[],
    )
    async with pool.acquire() as conn:
        run = await wf_queries.insert_wf_run(
            conn,
            account_id=account_id,
            workflow_id=wf.id,
            environment_id=environment_id,
            parent_run_id=None,
            launcher_session_id=None,
            request_id=request_id,
            caller=caller,
            script=wf.script,
            script_sha="x" * 64,
            host_semantics_epoch=1,
            input=None,
            tools=[],
            mcp_servers=[],
            http_servers=[],
            budget_usd=None,
            default_child_model="openrouter/test",
            depth=10,
        )
    return run.id


async def test_children_of_unions_edge_and_fk(
    pool_env: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    pool, account_id, env_id = pool_env
    root = await _seed_run(pool, account_id=account_id, environment_id=env_id)
    # A sub-run whose caller edge names the root run.
    sub = await _seed_run(
        pool,
        account_id=account_id,
        environment_id=env_id,
        caller={"kind": "run", "id": root},
        request_id="req-sub",
    )
    async with pool.acquire() as conn:
        kids = await trace_q.children_of(
            conn, caller_kind="run", caller_id=root, account_id=account_id
        )
    ids = {(k.kind, k.id) for k in kids}
    assert ("run", sub) in ids


async def test_children_of_is_account_scoped(
    pool_env: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    pool, account_id, env_id = pool_env
    root = await _seed_run(pool, account_id=account_id, environment_id=env_id)
    await _seed_run(
        pool,
        account_id=account_id,
        environment_id=env_id,
        caller={"kind": "run", "id": root},
    )
    # A different tenant must see nothing for this caller id.
    async with pool.acquire() as conn:
        kids = await trace_q.children_of(
            conn, caller_kind="run", caller_id=root, account_id="acc_other"
        )
    assert kids == []


async def test_get_trace_dfs_root_first(
    pool_env: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    pool, account_id, env_id = pool_env
    root = await _seed_run(pool, account_id=account_id, environment_id=env_id)
    sub = await _seed_run(
        pool,
        account_id=account_id,
        environment_id=env_id,
        caller={"kind": "run", "id": root},
        request_id="req-sub",
    )
    resp = await trace_service.get_trace(pool, root_kind="run", root_id=root, account_id=account_id)
    node_ids = [(e.kind, e.id) for e in resp.entries if e.kind in ("run", "session")]
    assert node_ids[0] == ("run", root)  # DFS pre-order: root first
    assert ("run", sub) in node_ids


async def test_get_trace_ceiling_truncates(
    pool_env: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    pool, account_id, env_id = pool_env
    root = await _seed_run(pool, account_id=account_id, environment_id=env_id)
    for _ in range(3):
        await _seed_run(
            pool,
            account_id=account_id,
            environment_id=env_id,
            caller={"kind": "run", "id": root},
        )
    resp = await trace_service.get_trace(
        pool, root_kind="run", root_id=root, account_id=account_id, max_nodes=2
    )
    assert resp.truncated is not None
    assert resp.truncated.at_nodes == 2


async def test_reverse_index_used_by_children_of_predicate(
    pool_env: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """EXPLAIN pins the 0103 reverse indexes to the exact children-of predicate."""
    pool, account_id, env_id = pool_env
    root = await _seed_run(pool, account_id=account_id, environment_id=env_id)
    async with pool.acquire() as conn:
        # The runtime planner only prefers the index on a non-trivial table; force
        # index consideration so the plan shape is deterministic in a small test
        # table (seqscan is otherwise cheaper).
        await conn.execute("SET enable_seqscan = off")
        wf_plan = await conn.fetch(
            "EXPLAIN SELECT r.id FROM wf_runs r "
            "WHERE r.account_id = $1 AND r.caller->>'kind' = $2 AND r.caller->>'id' = $3",
            account_id,
            "run",
            root,
        )
        ev_plan = await conn.fetch(
            "EXPLAIN SELECT e.session_id FROM events e "
            "WHERE e.account_id = $1 AND e.kind = 'lifecycle' "
            "AND e.data->>'event' = 'request_opened' "
            "AND e.data->'caller'->>'kind' = $2 AND e.data->'caller'->>'id' = $3",
            account_id,
            "run",
            root,
        )
    wf_text = "\n".join(r["QUERY PLAN"] for r in wf_plan)
    ev_text = "\n".join(r["QUERY PLAN"] for r in ev_plan)
    assert "wf_runs_caller_idx" in wf_text
    assert "events_request_opened_caller_idx" in ev_text
