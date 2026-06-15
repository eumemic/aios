"""Integration tests for the ``request_opened`` request edge (#1123).

The trusted *ask* half of the request edge: a typed ``request_opened`` lifecycle
event appended by the launch-path creation functions in the same transaction as
the servicer they open. ``get_open_request_ids`` derives the open set as
``asked(request_opened) MINUS answered(request_response)``.

DB-backed (testcontainer Postgres): exercises ``append_request_opened``,
``get_open_request_ids``, and the ``create_child_session`` edge emission /
replay-idempotency end to end against real session + event rows. The
service-writer-only invariant is covered structurally in
``tests/unit/test_request_opened_edge.py``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.models.attenuation import Surface
from aios.services import sessions as service
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration

_ACCOUNT = "acc_request_opened"


@pytest.fixture
async def pool_env(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str, str]]:
    """Yield ``(pool, account_id, agent_id, environment_id)`` for a fresh tenant."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ($1, NULL, TRUE, 'request-opened-test')",
                _ACCOUNT,
            )
        agent, env, _session = await seed_agent_env_session(
            pool, account_id=_ACCOUNT, prefix="request-opened"
        )
        yield pool, _ACCOUNT, agent.id, env.id
    finally:
        await pool.close()


async def _seed_parent_run(pool: asyncpg.Pool[Any], *, account_id: str, environment_id: str) -> str:
    """Insert a minimal workflow + run to satisfy the child's ``parent_run_id`` FK."""
    from aios.db.queries import workflows as wf_queries
    from aios.services import workflows as wf_service

    wf = await wf_service.create_workflow(
        pool,
        account_id=account_id,
        name="request-opened-wf",
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
            depth=10,  # #1124: root-budget seed for a directly-inserted run
            script=wf.script,
            script_sha="x" * 64,
            host_semantics_epoch=1,
            input=None,
            tools=[],
            mcp_servers=[],
            http_servers=[],
            budget_usd=None,
            default_child_model="openrouter/test",
        )
    return run.id


async def _child_session_id(pool: asyncpg.Pool[Any], account_id: str) -> str:
    return f"ses_child_{account_id[-6:]}_x"


# ─── append_request_opened + get_open_request_ids ────────────────────────────


async def test_open_request_ids_empty_for_ordinary_session(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str],
) -> None:
    pool, account_id, _agent_id, _env_id = pool_env
    _agent, _env, session = await seed_agent_env_session(
        pool, account_id=account_id, prefix="ordinary"
    )
    async with pool.acquire() as conn:
        assert await queries.get_open_request_ids(conn, session.id, account_id=account_id) == []


async def test_request_opened_appears_then_response_drops_it(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str],
) -> None:
    pool, account_id, _agent_id, env_id = pool_env
    _agent, _env, session = await seed_agent_env_session(
        pool, account_id=account_id, prefix="asked"
    )
    async with pool.acquire() as conn, conn.transaction():
        await queries.append_request_opened(
            conn,
            session_id=session.id,
            account_id=account_id,
            request_id="req-1",
            caller={"kind": "run", "id": "run_abc"},
            depth=2,
            environment_id=env_id,
            frozen_surface={"tools": [], "mcp_servers": [], "http_servers": []},
            vault_ids=["vault_x"],
        )
    async with pool.acquire() as conn:
        assert await queries.get_open_request_ids(conn, session.id, account_id=account_id) == [
            "req-1"
        ]
    # Answering via request_response drops it from the open set.
    async with pool.acquire() as conn:
        wrote = await queries.write_response_if_absent(
            conn,
            session.id,
            account_id=account_id,
            request_id="req-1",
            is_error=False,
            result={"ok": True},
            error=None,
        )
        assert wrote is True
    async with pool.acquire() as conn:
        assert await queries.get_open_request_ids(conn, session.id, account_id=account_id) == []


async def test_request_opened_frame_shape(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str],
) -> None:
    pool, account_id, _agent_id, env_id = pool_env
    _agent, _env, session = await seed_agent_env_session(
        pool, account_id=account_id, prefix="shape"
    )
    async with pool.acquire() as conn, conn.transaction():
        await queries.append_request_opened(
            conn,
            session_id=session.id,
            account_id=account_id,
            request_id="req-shape",
            caller={"kind": "session", "id": "ses_launcher"},
            depth=3,
            environment_id=env_id,
            frozen_surface={"tools": [], "mcp_servers": [], "http_servers": []},
            vault_ids=["v1", "v2"],
        )
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT kind, data FROM events WHERE session_id = $1 "
            "AND kind = 'lifecycle' AND data->>'event' = 'request_opened'",
            session.id,
        )
    assert row is not None
    assert row["kind"] == "lifecycle"
    data = queries.parse_jsonb(row["data"])
    assert data["event"] == "request_opened"
    assert data["request_id"] == "req-shape"
    assert data["caller"] == {"kind": "session", "id": "ses_launcher"}
    assert data["depth"] == 3
    assert data["environment_id"] == env_id
    assert data["frozen_surface"] == {"tools": [], "mcp_servers": [], "http_servers": []}
    assert data["vault_ids"] == ["v1", "v2"]


# ─── create_child_session: one edge per request, replay exactly-once ─────────


async def test_create_child_session_opens_exactly_one_edge_and_replay_idempotent(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str],
) -> None:
    pool, account_id, agent_id, env_id = pool_env
    parent_run_id = await _seed_parent_run(pool, account_id=account_id, environment_id=env_id)
    child_id = await _child_session_id(pool, account_id)
    surface = Surface(tools=[], mcp_servers=[], http_servers=[])

    created = await service.create_child_session(
        pool,
        session_id=child_id,
        account_id=account_id,
        agent_id=agent_id,
        environment_id=env_id,
        agent_version=1,
        model="openrouter/test",
        parent_run_id=parent_run_id,
        surface=surface,
        vault_ids=[],
        request_id="req-child",
        input="hello",
        depth=1,
    )
    assert created is True

    async with pool.acquire() as conn:
        n = await conn.fetchval(
            "SELECT count(*) FROM events WHERE session_id = $1 "
            "AND kind = 'lifecycle' AND data->>'event' = 'request_opened'",
            child_id,
        )
        assert n == 1
        # The trusted edge — not the forgeable blob — drives the open set.
        assert await queries.get_open_request_ids(conn, child_id, account_id=account_id) == [
            "req-child"
        ]
        # Dual-write invariant: the legacy metadata.request blob is still present.
        blob = await conn.fetchval(
            "SELECT data->'metadata'->'request'->>'request_id' FROM events "
            "WHERE session_id = $1 AND kind = 'message' AND role = 'user'",
            child_id,
        )
        assert blob == "req-child"

    # Replay: a second spawn hits ON CONFLICT → returns False → no second edge.
    replayed = await service.create_child_session(
        pool,
        session_id=child_id,
        account_id=account_id,
        agent_id=agent_id,
        environment_id=env_id,
        agent_version=1,
        model="openrouter/test",
        parent_run_id=parent_run_id,
        surface=surface,
        vault_ids=[],
        request_id="req-child",
        input="hello",
        depth=1,
    )
    assert replayed is False
    async with pool.acquire() as conn:
        n = await conn.fetchval(
            "SELECT count(*) FROM events WHERE session_id = $1 "
            "AND kind = 'lifecycle' AND data->>'event' = 'request_opened'",
            child_id,
        )
        assert n == 1  # exactly once across the replay


async def test_create_child_session_rollback_leaves_no_edge(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str],
) -> None:
    """The edge is written in the same transaction as the session row: if the
    transaction rolls back, neither the child row nor the edge survive."""
    pool, account_id, agent_id, env_id = pool_env
    parent_run_id = await _seed_parent_run(pool, account_id=account_id, environment_id=env_id)
    child_id = "ses_rollback_child"

    class _Boom(Exception):
        pass

    # Drive insert_child_session + append_request_opened in a transaction we abort.
    with pytest.raises(_Boom):
        async with pool.acquire() as conn, conn.transaction():
            child = await queries.insert_child_session(
                conn,
                session_id=child_id,
                account_id=account_id,
                agent_id=agent_id,
                environment_id=env_id,
                agent_version=1,
                model="openrouter/test",
                parent_run_id=parent_run_id,
                tools=[],
                mcp_servers=[],
                http_servers=[],
            )
            assert child is not None
            await queries.append_request_opened(
                conn,
                session_id=child_id,
                account_id=account_id,
                request_id="req-rb",
                caller={"kind": "run", "id": parent_run_id},
                depth=0,
                environment_id=env_id,
                frozen_surface={"tools": [], "mcp_servers": [], "http_servers": []},
                vault_ids=[],
            )
            raise _Boom

    async with pool.acquire() as conn:
        sess = await conn.fetchval("SELECT count(*) FROM sessions WHERE id = $1", child_id)
        edges = await conn.fetchval(
            "SELECT count(*) FROM events WHERE session_id = $1 "
            "AND kind = 'lifecycle' AND data->>'event' = 'request_opened'",
            child_id,
        )
    assert sess == 0
    assert edges == 0


# ─── #1124: cycles bounded by construction (no wait-for-graph) ────────────────


class _DepthExceeded(Exception):
    """Stand-in for ``WorkflowRunDepthExceededError`` at a session→session hop.

    #1124 makes the trusted ``depth`` scalar on the edge the cycle bound: each hop
    refuses BEFORE writing the child edge when the parent edge has no budget left.
    The session-invoke builtin (#1127) will raise the real depth-exceeded error; this
    test models the same refuse-before-write rule against the edge primitive so the
    bound is proven independently of that future call site.
    """


async def _invoke_session_hop(
    pool: asyncpg.Pool[Any],
    *,
    target_session_id: str,
    caller_session_id: str,
    request_id: str,
    parent_depth: int,
    account_id: str,
    environment_id: str,
) -> int:
    """Model ONE trusted session→session hop: refuse-before-write at the floor, else
    decrement and stamp ``parent_depth - 1`` onto the target's ``request_opened`` edge.

    This is exactly the rule #1124 applies at every trusted-invocation hop — no
    wait-for-graph, no cycle detection: the decrement IS the bound. Returns the
    child edge's depth.
    """
    child_depth = parent_depth - 1
    if child_depth < 0:
        # Refuse BEFORE writing the edge — no over-budget edge ever lands.
        raise _DepthExceeded
    async with pool.acquire() as conn, conn.transaction():
        await queries.append_request_opened(
            conn,
            session_id=target_session_id,
            account_id=account_id,
            request_id=request_id,
            caller={"kind": "session", "id": caller_session_id},
            depth=child_depth,
            environment_id=environment_id,
            frozen_surface={"tools": [], "mcp_servers": [], "http_servers": []},
            vault_ids=[],
        )
    return child_depth


async def test_session_to_session_cycle_terminates_at_budget_by_construction(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str],
) -> None:
    """A session→session A↔B cycle bottoms out at the budget BY CONSTRUCTION.

    Two sessions A and B invoke each other in a loop — the classic wait-for-graph
    cycle a run-only up-walk could never bound. #1124's DOWN-counter alone terminates
    it: each hop stamps ``parent_depth - 1`` onto the callee's edge and refuses before
    write at the floor. The cycle runs for exactly ``WORKFLOW_RUN_MAX_DEPTH`` edges
    (depths budget-1 .. 0) and then the next hop refuses — with NO cycle-detection
    code anywhere. The depth budget IS the cycle bound.
    """
    from aios.workflows.service import WORKFLOW_RUN_MAX_DEPTH

    pool, account_id, _agent_id, env_id = pool_env
    _aa, _ae, sess_a = await seed_agent_env_session(pool, account_id=account_id, prefix="cycle-a")
    _ba, _be, sess_b = await seed_agent_env_session(pool, account_id=account_id, prefix="cycle-b")
    ids = [sess_a.id, sess_b.id]

    # A is the edgeless root (a foreground session) seeded at the full budget; it
    # invokes B, which invokes A, which invokes B... alternating endpoints forever.
    depth = WORKFLOW_RUN_MAX_DEPTH
    hops = 0
    with pytest.raises(_DepthExceeded):
        while True:
            caller = ids[hops % 2]
            target = ids[(hops + 1) % 2]
            depth = await _invoke_session_hop(
                pool,
                target_session_id=target,
                caller_session_id=caller,
                request_id=f"cycle-req-{hops}",
                parent_depth=depth,
                account_id=account_id,
                environment_id=env_id,
            )
            hops += 1
            assert hops <= WORKFLOW_RUN_MAX_DEPTH + 1  # bounded — never an infinite loop

    # Exactly WORKFLOW_RUN_MAX_DEPTH edges were written (depths budget-1 down to 0);
    # the (budget+1)-th hop refused before writing. The A↔B cycle is bounded.
    assert hops == WORKFLOW_RUN_MAX_DEPTH
    async with pool.acquire() as conn:
        total_edges = await conn.fetchval(
            "SELECT count(*) FROM events WHERE session_id = ANY($1::text[]) "
            "AND kind = 'lifecycle' AND data->>'event' = 'request_opened'",
            ids,
        )
        min_depth = await conn.fetchval(
            "SELECT min((data->>'depth')::int) FROM events WHERE session_id = ANY($1::text[]) "
            "AND kind = 'lifecycle' AND data->>'event' = 'request_opened'",
            ids,
        )
    assert total_edges == WORKFLOW_RUN_MAX_DEPTH
    assert min_depth == 0  # the floor — the deepest edge carries depth 0, then refusal
