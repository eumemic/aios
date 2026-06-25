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
        script="async def main(input):\n    return None\n",
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


# ─── #1197: the `awaited` bit — Ask ⇒ true, Tell ⇒ false ─────────────────────


async def test_ask_create_child_session_writes_awaited_true(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str],
) -> None:
    """The behavior-preserved ``Ask(NewSession)`` path writes ``awaited=true`` and
    the child owes its response (it appears in the open set)."""
    pool, account_id, agent_id, env_id = pool_env
    parent_run_id = await _seed_parent_run(pool, account_id=account_id, environment_id=env_id)
    child_id = "ses_ask_child"
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
        request_id="req-ask",
        input="hello",
        depth=1,
    )
    assert created is True
    async with pool.acquire() as conn:
        awaited = await conn.fetchval(
            "SELECT data->>'awaited' FROM events WHERE session_id = $1 "
            "AND kind = 'lifecycle' AND data->>'event' = 'request_opened'",
            child_id,
        )
        assert awaited == "true"
        # An awaited edge IS in the open set — the child owes a response.
        assert await queries.get_open_request_ids(conn, child_id, account_id=account_id) == [
            "req-ask"
        ]


async def test_tell_new_session_writes_unawaited_edge_with_no_obligation(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str],
) -> None:
    """``Tell(NewSession)`` (``awaited=False``) writes a REAL ``request_opened``
    row — lineage/depth/surface carrier — but it is EXCLUDED from the open set, so
    the fire-and-forget child owes NO response (the awaited-triad filter)."""
    pool, account_id, agent_id, env_id = pool_env
    parent_run_id = await _seed_parent_run(pool, account_id=account_id, environment_id=env_id)
    child_id = "ses_tell_child"
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
        input="fire-and-forget",
        depth=1,
        awaited=False,  # the Tell arm — no request_id needed
    )
    assert created is True
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT data FROM events WHERE session_id = $1 "
            "AND kind = 'lifecycle' AND data->>'event' = 'request_opened'",
            child_id,
        )
        assert row is not None  # a real edge row exists (lineage/depth/surface carrier)
        data = queries.parse_jsonb(row["data"])
        assert data["awaited"] is False
        assert data["depth"] == 1  # depth still carried
        # ...but it is NOT in the open set — no response obligation.
        assert await queries.get_open_request_ids(conn, child_id, account_id=account_id) == []


async def test_tell_new_session_rejects_output_schema(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str],
) -> None:
    """A ``Tell`` cannot carry an ``output_schema`` (it owes no response)."""
    from aios.errors import ValidationError

    pool, account_id, agent_id, env_id = pool_env
    parent_run_id = await _seed_parent_run(pool, account_id=account_id, environment_id=env_id)
    surface = Surface(tools=[], mcp_servers=[], http_servers=[])
    with pytest.raises(ValidationError):
        await service.create_child_session(
            pool,
            session_id="ses_tell_bad",
            account_id=account_id,
            agent_id=agent_id,
            environment_id=env_id,
            agent_version=1,
            model="openrouter/test",
            parent_run_id=parent_run_id,
            surface=surface,
            vault_ids=[],
            input="x",
            depth=1,
            awaited=False,
            output_schema={"type": "object"},
        )


async def test_legacy_edge_without_awaited_field_reads_as_awaited(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str],
) -> None:
    """A pre-#1197 ``request_opened`` row with no ``awaited`` field reads as
    awaited (additive/legacy) — it still owes a response."""
    pool, account_id, _agent_id, env_id = pool_env
    _agent, _env, session = await seed_agent_env_session(
        pool, account_id=account_id, prefix="legacy-awaited"
    )
    # Hand-write a legacy frame WITHOUT the awaited key. ``events.seq`` is
    # ``bigint NOT NULL`` (gapless per session, allocated off the session's
    # ``last_event_seq`` counter — see ``queries.events.append_event``), so the
    # raw INSERT must allocate a seq the same way rather than omit the column.
    async with pool.acquire() as conn, conn.transaction():
        seq = await conn.fetchval(
            "UPDATE sessions SET last_event_seq = last_event_seq + 1 "
            "WHERE id = $1 AND account_id = $2 RETURNING last_event_seq",
            session.id,
            account_id,
        )
        await conn.execute(
            "INSERT INTO events (id, account_id, session_id, seq, kind, data) "
            "VALUES ('evt_legacy_awaited', $1, $2, $3, 'lifecycle', $4::jsonb)",
            account_id,
            session.id,
            seq,
            '{"event":"request_opened","request_id":"req-legacy",'
            '"caller":{"kind":"run","id":"run_x"},"depth":0,'
            '"environment_id":"' + env_id + '","frozen_surface":'
            '{"tools":[],"mcp_servers":[],"http_servers":[]},"vault_ids":[]}',
        )
    async with pool.acquire() as conn:
        assert await queries.get_open_request_ids(conn, session.id, account_id=account_id) == [
            "req-legacy"
        ]


async def test_tell_existing_session_appends_message_no_edge_channel_less(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``Tell(ExistingSession)`` appends a channel-less user message + wakes, opens
    NO request edge, and never renders to a connector (no ``orig_channel``)."""
    pool, account_id, _agent_id, _env_id = pool_env
    _agent, _env, session = await seed_agent_env_session(
        pool, account_id=account_id, prefix="tell-existing"
    )

    deferred: list[tuple[str, str]] = []

    async def _fake_defer_wake(_pool: Any, sid: str, *, cause: str, account_id: str) -> None:
        deferred.append((sid, cause))

    import aios.services.wake as wake_module

    monkeypatch.setattr(wake_module, "defer_wake", _fake_defer_wake)
    ok = await service.stimulate(
        pool,
        service.TellExistingSession(session_id=session.id, content="go look", cause="message"),
        account_id=account_id,
    )
    assert ok is True
    assert deferred == [(session.id, "message")]

    async with pool.acquire() as conn:
        # A user message landed...
        msg = await conn.fetchrow(
            "SELECT data, orig_channel FROM events WHERE session_id = $1 "
            "AND kind = 'message' AND role = 'user' ORDER BY seq DESC LIMIT 1",
            session.id,
        )
        assert msg is not None
        assert queries.parse_jsonb(msg["data"])["content"] == "go look"
        # ...channel-less (never renders to a connector).
        assert msg["orig_channel"] is None
        # ...and NO request edge was opened.
        edges = await conn.fetchval(
            "SELECT count(*) FROM events WHERE session_id = $1 "
            "AND kind = 'lifecycle' AND data->>'event' = 'request_opened'",
            session.id,
        )
        assert edges == 0


# ─── #1124: the DOWN-counting trusted depth on the edge bounds cycles ─────────


async def test_session_to_session_cycle_bounded_by_construction(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str],
) -> None:
    """A session→session A↔B ping-pong terminates at the shared budget BY
    CONSTRUCTION (#1124) — no wait-for-graph, no cycle detection. Each trusted
    edge carries ``parent_depth - 1``; once the budget is spent the next hop
    would carry a negative depth, which the decrement rule refuses. This is the
    coverage the run-only ``run_ancestor_depth`` CTE could never provide: the
    cycle bound is the depth budget itself, applied uniformly to the edge.
    """
    pool, account_id, _agent_id, env_id = pool_env
    from aios.workflows.service import INVOKE_MAX_DEPTH

    _a_agent, _a_env, sess_a = await seed_agent_env_session(
        pool, account_id=account_id, prefix="cycle-a"
    )
    _b_agent, _b_env, sess_b = await seed_agent_env_session(
        pool, account_id=account_id, prefix="cycle-b"
    )

    # Replay the A↔B invoke-edge cycle the decrement rule produces: an edgeless
    # root seeds at the full budget, and each trusted hop stamps parent_depth - 1.
    # The loop stops emitting the moment a hop has no budget left to spend.
    targets = [sess_b.id, sess_a.id]  # A invokes B, B invokes A, A invokes B, ...
    callers = [sess_a.id, sess_b.id]
    depth = INVOKE_MAX_DEPTH
    hops = 0
    async with pool.acquire() as conn, conn.transaction():
        while depth >= 1:  # refuse-before-write: a 0-budget caller opens no edge
            child_depth = depth - 1
            await queries.append_request_opened(
                conn,
                session_id=targets[hops % 2],
                account_id=account_id,
                request_id=f"req-cycle-{hops}",
                caller={"kind": "session", "id": callers[hops % 2]},
                depth=child_depth,
                environment_id=env_id,
                frozen_surface={"tools": [], "mcp_servers": [], "http_servers": []},
                vault_ids=[],
            )
            hops += 1
            depth = child_depth

    # The cycle bottomed out after exactly INVOKE_MAX_DEPTH hops — purely from the
    # decrement, regardless of the A↔B structure.
    assert hops == INVOKE_MAX_DEPTH
    async with pool.acquire() as conn:
        total = await conn.fetchval(
            "SELECT count(*) FROM events WHERE account_id = $1 "
            "AND kind = 'lifecycle' AND data->>'event' = 'request_opened' "
            "AND (data->>'request_id') LIKE 'req-cycle-%'",
            account_id,
        )
    assert total == INVOKE_MAX_DEPTH


# ─── #1413: get_open_obligations — the tail-injected obligations data source ──


async def test_get_open_obligations_empty_for_no_edge_session(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str],
) -> None:
    pool, account_id, _agent_id, _env_id = pool_env
    _agent, _env, session = await seed_agent_env_session(
        pool, account_id=account_id, prefix="obl_empty"
    )
    async with pool.acquire() as conn:
        assert await queries.get_open_obligations(conn, session.id, account_id=account_id) == []


async def test_get_open_obligations_projects_caller_kind_opened_at_summary(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str],
) -> None:
    pool, account_id, _agent_id, env_id = pool_env
    _agent, _env, session = await seed_agent_env_session(
        pool, account_id=account_id, prefix="obl_proj"
    )
    async with pool.acquire() as conn, conn.transaction():
        await queries.append_request_opened(
            conn,
            session_id=session.id,
            account_id=account_id,
            request_id="req-proj",
            caller={"kind": "run", "id": "run_xyz"},
            depth=1,
            environment_id=env_id,
            frozen_surface={"tools": [], "mcp_servers": [], "http_servers": []},
            vault_ids=[],
            summary="research the topic",
        )
    async with pool.acquire() as conn:
        obs = await queries.get_open_obligations(conn, session.id, account_id=account_id)
    assert len(obs) == 1
    o = obs[0]
    assert o.request_id == "req-proj"
    assert o.caller_kind == "run"
    assert o.caller_id == "run_xyz"
    assert o.summary == "research the topic"
    assert o.opened_at is not None
    # No output_schema persisted on this edge → None (additive, no migration).
    assert o.output_schema is None


async def test_get_open_obligations_projects_output_schema(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str],
) -> None:
    """#1522: the widened owed-read-model SELECTs the persisted ``output_schema``
    off the ``request_opened`` frame (the same datum ``get_request_output_schema``
    reads) — additive, ``None`` when absent. The cheap guard path
    (``get_open_request_ids``) is unaffected; only this content model carries it."""
    pool, account_id, _agent_id, env_id = pool_env
    _agent, _env, session = await seed_agent_env_session(
        pool, account_id=account_id, prefix="obl_schema"
    )
    schema = {
        "type": "object",
        "properties": {"shipped": {"type": "boolean"}},
        "required": ["shipped"],
    }
    async with pool.acquire() as conn, conn.transaction():
        await queries.append_request_opened(
            conn,
            session_id=session.id,
            account_id=account_id,
            request_id="req-schema",
            caller={"kind": "session", "id": session.id},  # a self-goal edge
            depth=0,
            environment_id=env_id,
            frozen_surface={"tools": [], "mcp_servers": [], "http_servers": []},
            vault_ids=[],
            summary="ship the thing",
            output_schema=schema,
        )
    async with pool.acquire() as conn:
        obs = await queries.get_open_obligations(conn, session.id, account_id=account_id)
        # The widened model carries the contract...
        assert len(obs) == 1
        assert obs[0].output_schema == schema
        # ...and reads the SAME datum get_request_output_schema reads off the frame.
        via_schema_reader = await queries.get_request_output_schema(
            conn, session.id, request_id="req-schema"
        )
    assert via_schema_reader == schema


async def test_get_open_obligations_oldest_first_and_excludes_answered(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str],
) -> None:
    pool, account_id, _agent_id, env_id = pool_env
    _agent, _env, session = await seed_agent_env_session(
        pool, account_id=account_id, prefix="obl_order"
    )
    for rid in ("req-1", "req-2", "req-3"):
        async with pool.acquire() as conn, conn.transaction():
            await queries.append_request_opened(
                conn,
                session_id=session.id,
                account_id=account_id,
                request_id=rid,
                caller={"kind": "session", "id": "ses_caller"},
                depth=0,
                environment_id=env_id,
                frozen_surface={"tools": [], "mcp_servers": [], "http_servers": []},
                vault_ids=[],
                summary=f"summary {rid}",
            )
    # Answer the middle one — it drops out of the open set.
    async with pool.acquire() as conn:
        await queries.write_response_if_absent(
            conn,
            session.id,
            account_id=account_id,
            request_id="req-2",
            is_error=False,
            result={"ok": True},
            error=None,
        )
    async with pool.acquire() as conn:
        obs = await queries.get_open_obligations(conn, session.id, account_id=account_id)
    assert [o.request_id for o in obs] == ["req-1", "req-3"]  # oldest-first, answered excluded


async def test_get_open_obligations_excludes_unawaited_tell_edge(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str],
) -> None:
    """A ``Tell`` (``awaited=False``) edge owes no response — excluded from the
    obligations set, in lockstep with ``get_open_request_ids``."""
    pool, account_id, _agent_id, env_id = pool_env
    _agent, _env, session = await seed_agent_env_session(
        pool, account_id=account_id, prefix="obl_tell"
    )
    async with pool.acquire() as conn, conn.transaction():
        await queries.append_request_opened(
            conn,
            session_id=session.id,
            account_id=account_id,
            request_id="req-tell",
            caller={"kind": "run", "id": "run_t"},
            depth=0,
            environment_id=env_id,
            frozen_surface={"tools": [], "mcp_servers": [], "http_servers": []},
            vault_ids=[],
            awaited=False,
            summary="fire and forget",
        )
    async with pool.acquire() as conn:
        assert await queries.get_open_obligations(conn, session.id, account_id=account_id) == []


async def test_get_open_obligations_summary_absent_reads_as_none(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str],
) -> None:
    """A pre-#1413 frame written WITHOUT a summary reads back with ``summary=None``
    (additive — id-only render line, no crash, no migration)."""
    pool, account_id, _agent_id, env_id = pool_env
    _agent, _env, session = await seed_agent_env_session(
        pool, account_id=account_id, prefix="obl_nosum"
    )
    async with pool.acquire() as conn, conn.transaction():
        await queries.append_request_opened(
            conn,
            session_id=session.id,
            account_id=account_id,
            request_id="req-nosum",
            caller={"kind": "api", "id": "api_caller"},
            depth=0,
            environment_id=env_id,
            frozen_surface={"tools": [], "mcp_servers": [], "http_servers": []},
            vault_ids=[],
            # summary omitted → not written to the frame
        )
        # Confirm the key is genuinely absent on the persisted frame.
        row = await conn.fetchrow(
            "SELECT data FROM events WHERE session_id = $1 "
            "AND kind = 'lifecycle' AND data->>'event' = 'request_opened'",
            session.id,
        )
    data = queries.parse_jsonb(row["data"])
    assert "summary" not in data
    async with pool.acquire() as conn:
        obs = await queries.get_open_obligations(conn, session.id, account_id=account_id)
    assert len(obs) == 1
    assert obs[0].summary is None
    assert obs[0].caller_kind == "api"
