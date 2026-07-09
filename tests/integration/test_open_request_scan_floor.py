"""Integration tests for the #1747 monotone open-request scan-floor.

``get_open_obligations`` / ``get_open_request_ids`` anti-join EVERY
``request_opened`` edge a session ever accrued, every step — unbounded
lifetime growth on the hottest path for a re-invoked servicer session
(#1127/#1128). The fix is a monotone per-session floor
(``sessions.open_request_scan_floor``, migration 0134) plus a supporting
partial index (``events_request_opened_seq_idx``, migration 0135):
:func:`queries.advance_open_request_scan_floor` ratchets the floor forward
past *answered* edges only, and the two shared readers
(:func:`queries.get_open_request_ids` / :func:`queries.get_open_obligations`)
bound their scan to ``req.seq >= floor`` via
``open_request_anti_join(floor_bounded=True)``.

Modeled on ``test_request_opened_edge.py`` (session/edge fixtures) and
``test_migrations_0128_inbound_budget_index.py`` (pg_indexes + ``EXPLAIN
(FORMAT JSON)`` plan-shape).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.models.sessions import Ok
from tests.conftest import _docker_available, needs_docker
from tests.integration.conftest import seed_agent_env_session
from tests.integration.test_migrations import _alembic_url, _run_alembic

pytestmark = pytest.mark.integration

_ACCOUNT = "acc_scan_floor"


# ─── shared session/pool fixture (mirrors test_request_opened_edge.py) ───────


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
                "VALUES ($1, NULL, TRUE, 'scan-floor-test')",
                _ACCOUNT,
            )
        agent, env, _session = await seed_agent_env_session(
            pool, account_id=_ACCOUNT, prefix="scan-floor"
        )
        yield pool, _ACCOUNT, agent.id, env.id
    finally:
        await pool.close()


async def _open(
    pool: asyncpg.Pool[Any],
    *,
    session_id: str,
    account_id: str,
    environment_id: str,
    request_id: str,
    awaited: bool = True,
) -> None:
    async with pool.acquire() as conn, conn.transaction():
        await queries.append_request_opened(
            conn,
            session_id=session_id,
            account_id=account_id,
            request_id=request_id,
            caller={"kind": "run", "id": "run_abc"},
            depth=1,
            environment_id=environment_id,
            frozen_surface={"tools": [], "mcp_servers": [], "http_servers": []},
            vault_ids=[],
            awaited=awaited,
        )


async def _answer(pool: asyncpg.Pool[Any], *, session_id: str, account_id: str, request_id: str) -> None:
    async with pool.acquire() as conn:
        wrote = await queries.write_response_if_absent(
            conn,
            session_id,
            account_id=account_id,
            request_id=request_id,
            outcome=Ok(result={"ok": True}),
        )
        assert wrote is True


async def _advance(pool: asyncpg.Pool[Any], *, session_id: str, account_id: str) -> None:
    async with pool.acquire() as conn:
        await queries.advance_open_request_scan_floor(conn, session_id, account_id=account_id)


async def _floor(pool: asyncpg.Pool[Any], *, session_id: str) -> int:
    async with pool.acquire() as conn:
        v = await conn.fetchval(
            "SELECT open_request_scan_floor FROM sessions WHERE id = $1", session_id
        )
    assert v is not None
    return int(v)


async def _seq_of(pool: asyncpg.Pool[Any], *, session_id: str, request_id: str) -> int:
    async with pool.acquire() as conn:
        v = await conn.fetchval(
            "SELECT seq FROM events WHERE session_id = $1 AND kind = 'lifecycle' "
            "AND data->>'event' = 'request_opened' AND data->>'request_id' = $2",
            session_id,
            request_id,
        )
    assert v is not None
    return int(v)


# ─── 1. floor edge stays visible to both readers ──────────────────────────────


async def test_floor_edge_stays_visible_to_both_readers(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str],
) -> None:
    pool, account_id, _agent_id, env_id = pool_env
    _agent, _env, session = await seed_agent_env_session(pool, account_id=account_id, prefix="t1")

    await _open(pool, session_id=session.id, account_id=account_id, environment_id=env_id, request_id="A")
    await _advance(pool, session_id=session.id, account_id=account_id)

    seq_a = await _seq_of(pool, session_id=session.id, request_id="A")
    assert await _floor(pool, session_id=session.id) == seq_a

    async with pool.acquire() as conn:
        ids = await queries.get_open_request_ids(conn, session.id, account_id=account_id)
        obligations = await queries.get_open_obligations(conn, session.id, account_id=account_id)
    assert ids == ["A"]
    assert [o.request_id for o in obligations] == ["A"]


# ─── 2. floor strictly advances past an answered edge ────────────────────────


async def test_floor_advances_strictly_past_answered_edge(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str],
) -> None:
    pool, account_id, _agent_id, env_id = pool_env
    _agent, _env, session = await seed_agent_env_session(pool, account_id=account_id, prefix="t2")

    await _open(pool, session_id=session.id, account_id=account_id, environment_id=env_id, request_id="A")
    await _advance(pool, session_id=session.id, account_id=account_id)
    seq_a = await _seq_of(pool, session_id=session.id, request_id="A")

    await _answer(pool, session_id=session.id, account_id=account_id, request_id="A")
    await _advance(pool, session_id=session.id, account_id=account_id)

    floor_after_answer = await _floor(pool, session_id=session.id)
    assert floor_after_answer > seq_a, "floor must strictly pass an answered edge"

    await _open(pool, session_id=session.id, account_id=account_id, environment_id=env_id, request_id="B")
    async with pool.acquire() as conn:
        ids = await queries.get_open_request_ids(conn, session.id, account_id=account_id)
    assert ids == ["B"]


# ─── 3. monotone, no-op write when there is no request edge ──────────────────


async def test_monotone_and_zero_write_with_no_request_edge(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str],
) -> None:
    pool, account_id, _agent_id, _env_id = pool_env
    _agent, _env, session = await seed_agent_env_session(pool, account_id=account_id, prefix="t3")

    async with pool.acquire() as conn:
        result = await conn.execute(
            "WITH open_edges AS ("
            "    SELECT req.seq FROM events req WHERE "
            + queries.open_request_anti_join(
                sid="$1", acct="$2", awaited_only=True, floor_bounded=True
            )
            + "), f AS ("
            "    SELECT COALESCE("
            "        (SELECT min(seq) FROM open_edges),"
            "        (SELECT max(seq) + 1 FROM events e"
            "           WHERE e.session_id = $1 AND e.account_id = $2"
            "             AND e.kind = 'lifecycle' AND e.data->>'event' = 'request_opened')"
            "    ) AS v"
            ")"
            "UPDATE sessions"
            "   SET open_request_scan_floor = (SELECT v FROM f)"
            " WHERE id = $1 AND account_id = $2"
            "   AND (SELECT v FROM f) IS NOT NULL"
            "   AND open_request_scan_floor < (SELECT v FROM f)",
            session.id,
            account_id,
        )
    # asyncpg execute() returns a command-tag string, e.g. "UPDATE 0"
    assert result == "UPDATE 0", result
    assert await _floor(pool, session_id=session.id) == 0

    # Repeated advances stay a no-op / never decrease (0, still).
    await _advance(pool, session_id=session.id, account_id=account_id)
    assert await _floor(pool, session_id=session.id) == 0


# ─── 4. interleaved answered/answered/open — only the open one returned ──────


async def test_interleaved_sequence_only_open_edge_survives(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str],
) -> None:
    pool, account_id, _agent_id, env_id = pool_env
    _agent, _env, session = await seed_agent_env_session(pool, account_id=account_id, prefix="t4")

    await _open(pool, session_id=session.id, account_id=account_id, environment_id=env_id, request_id="A")
    await _advance(pool, session_id=session.id, account_id=account_id)
    await _answer(pool, session_id=session.id, account_id=account_id, request_id="A")
    await _advance(pool, session_id=session.id, account_id=account_id)

    await _open(pool, session_id=session.id, account_id=account_id, environment_id=env_id, request_id="B")
    await _advance(pool, session_id=session.id, account_id=account_id)
    await _answer(pool, session_id=session.id, account_id=account_id, request_id="B")
    await _advance(pool, session_id=session.id, account_id=account_id)

    await _open(pool, session_id=session.id, account_id=account_id, environment_id=env_id, request_id="C")
    await _advance(pool, session_id=session.id, account_id=account_id)

    seq_c = await _seq_of(pool, session_id=session.id, request_id="C")
    async with pool.acquire() as conn:
        ids = await queries.get_open_request_ids(conn, session.id, account_id=account_id)
    assert ids == ["C"]
    assert await _floor(pool, session_id=session.id) <= seq_c


# ─── 5. demote reader (awaited_only=False) is NOT floor-bounded ──────────────


async def test_demote_reader_stays_unbounded_and_sees_old_unawaited_tell(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str],
) -> None:
    pool, account_id, _agent_id, env_id = pool_env
    _agent, _env, session = await seed_agent_env_session(pool, account_id=account_id, prefix="t5")

    # An old unawaited Tell edge, answered-irrelevant (it owes no response).
    await _open(
        pool,
        session_id=session.id,
        account_id=account_id,
        environment_id=env_id,
        request_id="tell-1",
        awaited=False,
    )
    # Advance the floor via a separate, later AWAITED+answered edge so the
    # floor moves well past the unawaited Tell's seq.
    await _open(pool, session_id=session.id, account_id=account_id, environment_id=env_id, request_id="A")
    await _advance(pool, session_id=session.id, account_id=account_id)
    await _answer(pool, session_id=session.id, account_id=account_id, request_id="A")
    await _advance(pool, session_id=session.id, account_id=account_id)

    tell_seq = await _seq_of(pool, session_id=session.id, request_id="tell-1")
    floor = await _floor(pool, session_id=session.id)
    assert floor > tell_seq, "test setup requires the floor to have passed the Tell edge"

    # get_wake_priority_context uses awaited_only=False and must NOT be
    # floor-bounded — it should still see the old Tell edge as the latest
    # still-open (unawaited) edge and derive background from its caller.
    async with pool.acquire() as conn:
        result = await queries.get_wake_priority_context(conn, session.id)
    assert result is not None
    _acct, is_background = result
    assert is_background is True  # caller.kind == "run" => background


# ─── 6. fragment guards ───────────────────────────────────────────────────────


def test_floor_bounded_requires_awaited_only() -> None:
    with pytest.raises(ValueError, match="awaited_only=True"):
        queries.open_request_anti_join(
            sid="$1", acct="$2", awaited_only=False, floor_bounded=True
        )


def test_floor_bounded_rejects_batch_any_sid() -> None:
    with pytest.raises(ValueError, match="scalar"):
        queries.open_request_anti_join(
            sid="ANY($1::text[])", acct="$2", awaited_only=True, floor_bounded=True
        )


def test_floor_bounded_false_allows_batch_any_sid() -> None:
    # Sanity: the guard is specific to floor_bounded=True, not a blanket ban.
    sql = queries.open_request_anti_join(
        sid="ANY($1::text[])", acct="$2", awaited_only=True, floor_bounded=False
    )
    assert "open_request_scan_floor" not in sql


# ─── 7. migration plan-shape: partial index exists, valid, and is used ───────


@pytest.fixture
def postgres() -> Iterator[object]:
    if not _docker_available():
        pytest.skip("Docker not available")
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("postgres:16-alpine") as pg:
        yield pg


async def _execute(db_url: str, sql: str, *args: Any) -> None:
    conn = await asyncpg.connect(db_url)
    try:
        await conn.execute(sql, *args)
    finally:
        await conn.close()


async def _fetchval(db_url: str, sql: str, *args: Any) -> Any:
    conn = await asyncpg.connect(db_url)
    try:
        return await conn.fetchval(sql, *args)
    finally:
        await conn.close()


def _collect_nodes(plan_node: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = [plan_node]
    for child in plan_node.get("Plans", []):
        nodes.extend(_collect_nodes(child))
    return nodes


@needs_docker
@pytest.mark.integration
def test_migration_creates_valid_partial_index(postgres: object) -> None:
    db_url = _alembic_url(postgres)
    up = _run_alembic(["upgrade", "head"], db_url)
    assert up.returncode == 0, f"upgrade to head failed:\n{up.stderr}\n{up.stdout}"

    async def _check() -> tuple[str | None, bool | None]:
        conn = await asyncpg.connect(db_url)
        try:
            indexdef = await conn.fetchval(
                "SELECT indexdef FROM pg_indexes WHERE tablename = 'events' "
                "AND indexname = 'events_request_opened_seq_idx'"
            )
            valid = await conn.fetchval(
                "SELECT indisvalid FROM pg_index "
                "JOIN pg_class ON pg_class.oid = pg_index.indexrelid "
                "WHERE pg_class.relname = 'events_request_opened_seq_idx'"
            )
            return indexdef, valid
        finally:
            await conn.close()

    indexdef, valid = asyncio.run(_check())
    assert indexdef is not None, "events_request_opened_seq_idx missing after upgrade"
    assert "session_id" in indexdef and "seq" in indexdef, indexdef
    assert "request_opened" in indexdef, indexdef
    assert valid is True, "events_request_opened_seq_idx is invalid (failed CONCURRENTLY build)"

    # sessions.open_request_scan_floor exists, NOT NULL, default 0.
    async def _col() -> tuple[bool, str | None]:
        conn = await asyncpg.connect(db_url)
        try:
            row = await conn.fetchrow(
                "SELECT is_nullable, column_default FROM information_schema.columns "
                "WHERE table_name = 'sessions' AND column_name = 'open_request_scan_floor'"
            )
            assert row is not None
            return row["is_nullable"] == "NO", row["column_default"]
        finally:
            await conn.close()

    not_null, default = asyncio.run(_col())
    assert not_null is True
    assert default is not None and "0" in default


_CHAIN_SQL = """
INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
VALUES ('acc_root', NULL, TRUE, 'root');
INSERT INTO environments (id, name, account_id)
VALUES ('env_a', 'env-a', 'acc_root');
INSERT INTO agents (id, name, model, account_id)
VALUES ('agent_a', 'agent-a', 'test/model', 'acc_root');
INSERT INTO sessions (id, agent_id, environment_id, workspace_volume_path, account_id,
                      last_event_seq, open_request_scan_floor)
VALUES ('sess_a', 'agent_a', 'env_a', '/tmp/ws-a', 'acc_root', 1000, 990);
"""


@needs_docker
@pytest.mark.integration
def test_floor_bounded_query_uses_partial_index_not_full_scan(postgres: object) -> None:
    """The master-failing pin: EXPLAIN of the floor-bounded ``get_open_obligations``
    SQL must seek ``events_request_opened_seq_idx`` with a seq-range condition,
    not scan every ``request_opened`` row."""
    db_url = _alembic_url(postgres)
    up = _run_alembic(["upgrade", "head"], db_url)
    assert up.returncode == 0, f"upgrade to head failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(_execute(db_url, _CHAIN_SQL))

    # Seed 1000 request_opened+request_response pairs below the floor (seq
    # 1..1980) plus one open edge at/after the floor (seq 991) so the planner
    # has a real reason to prefer the index over a full scan.
    async def _seed_events() -> None:
        conn = await asyncpg.connect(db_url)
        try:
            rows = []
            seq = 1
            for i in range(495):
                rows.append(
                    (
                        f"evt_open_{i}",
                        "sess_a",
                        seq,
                        "lifecycle",
                        json.dumps(
                            {
                                "event": "request_opened",
                                "request_id": f"req_{i}",
                                "caller": {"kind": "run", "id": "run_x"},
                                "awaited": True,
                            }
                        ),
                        "acc_root",
                    )
                )
                seq += 1
                rows.append(
                    (
                        f"evt_resp_{i}",
                        "sess_a",
                        seq,
                        "lifecycle",
                        json.dumps({"event": "request_response", "request_id": f"req_{i}"}),
                        "acc_root",
                    )
                )
                seq += 1
            # One open edge at seq 991 (>= floor 990).
            rows.append(
                (
                    "evt_open_current",
                    "sess_a",
                    991,
                    "lifecycle",
                    json.dumps(
                        {
                            "event": "request_opened",
                            "request_id": "req_current",
                            "caller": {"kind": "run", "id": "run_x"},
                            "awaited": True,
                        }
                    ),
                    "acc_root",
                )
            )
            await conn.executemany(
                "INSERT INTO events (id, session_id, seq, kind, data, account_id) "
                "VALUES ($1, $2, $3, $4, $5::jsonb, $6)",
                rows,
            )
        finally:
            await conn.close()

    asyncio.run(_seed_events())

    query = (
        "SELECT req.data->>'request_id' AS rid FROM events req WHERE "
        + queries.open_request_anti_join(
            sid="$1", acct="$2", awaited_only=True, floor_bounded=True
        )
        + "ORDER BY req.seq ASC"
    )

    async def _plan() -> dict[str, Any]:
        conn = await asyncpg.connect(db_url)
        try:
            await conn.execute("SET enable_seqscan = off")
            result = await conn.fetchval(
                f"EXPLAIN (FORMAT JSON) {query}", "sess_a", "acc_root"
            )
            if isinstance(result, str):
                result = json.loads(result)
            return result[0]["Plan"]  # type: ignore[no-any-return]
        finally:
            await conn.close()

    plan = asyncio.run(_plan())
    nodes = _collect_nodes(plan)

    events_scans = [n for n in nodes if n.get("Relation Name") == "events"]
    assert events_scans, f"no scan node over events in plan: {plan}"

    index_scans = [
        n
        for n in nodes
        if n.get("Node Type") in ("Index Scan", "Index Only Scan", "Bitmap Index Scan")
        and n.get("Index Name") == "events_request_opened_seq_idx"
    ]
    assert index_scans, (
        "plan does not seek events_request_opened_seq_idx; "
        f"events scan nodes: {[(n.get('Node Type'), n.get('Index Name')) for n in events_scans]}"
    )


# ─── 8. end-to-end through compute_step_prelude ───────────────────────────────


@pytest.fixture
def _stub_tool_provider() -> Iterator[None]:
    """Mirrors ``test_obligations_tail_block.py``'s stub: ``compute_step_prelude``
    calls into MCP discovery, which we don't need to exercise here."""
    from unittest import mock
    from unittest.mock import AsyncMock

    from aios.harness import runtime

    prev = runtime.tool_provider
    tp = mock.Mock()
    tp.list_tools_for_session = AsyncMock(return_value=[])
    runtime.tool_provider = tp
    try:
        yield
    finally:
        runtime.tool_provider = prev


async def test_step_prelude_ratchets_floor_end_to_end(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str],
    _stub_tool_provider: None,
) -> None:
    """Drives the real step-prelude path so an unwired or exception-swallowed
    advance shows up here even though every query-level test above would pass
    at floor 0 (floor 0 is byte-identical to the pre-#1747 unbounded scan)."""
    from aios.harness.step_context import compute_step_prelude
    from aios.services import agents as agents_service
    from aios.services import sessions as sessions_service

    pool, account_id, _agent_id, env_id = pool_env
    _agent, _env, seeded_session = await seed_agent_env_session(
        pool, account_id=account_id, prefix="t8"
    )

    await _open(
        pool,
        session_id=seeded_session.id,
        account_id=account_id,
        environment_id=env_id,
        request_id="A",
    )
    await _answer(pool, session_id=seeded_session.id, account_id=account_id, request_id="A")

    assert await _floor(pool, session_id=seeded_session.id) == 0

    session = await sessions_service.get_session_basic(
        pool, seeded_session.id, account_id=account_id
    )
    agent = await agents_service.load_for_session(pool, session, account_id=account_id)

    await compute_step_prelude(
        pool,
        seeded_session.id,
        account_id=account_id,
        session=session,
        agent=agent,
        channels=[],
        memory_store_echoes=[],
    )

    floor_after = await _floor(pool, session_id=seeded_session.id)
    assert floor_after > 0, "compute_step_prelude must have ratcheted the floor"

    seq_a = await _seq_of(pool, session_id=seeded_session.id, request_id="A")
    assert floor_after > seq_a, "floor must have advanced past the answered edge A"

    # A subsequent read is unaffected in content — still correctly empty —
    # confirming the ratchet is perf-only, not a correctness change.
    async with pool.acquire() as conn:
        ids = await queries.get_open_request_ids(conn, seeded_session.id, account_id=account_id)
    assert ids == []
