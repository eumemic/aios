"""Migration 0134 adds ``events_tool_confirmed_allow_recent_idx`` — the
``created_at``-keyed partial index that lets the sweep's cross-session
confirmed-rows detector prune at the index rather than heap-fetching every
confirmed-allow row ever (#1740).

``sweep.CONFIRMED_ROWS_SQL`` (composed from
``queries.confirmed_unresolved_predicate``) has no ``session_id`` equality to
seek on (it is cross-session), so the only existing index covering its
predicate — ``events_tool_confirmed_allow_idx`` (migration 0065, keyed
``(session_id, tool_call_id)``) — cannot help it. Before this migration the
planner's only access path was a scan of every confirmed-allow row ever
written, applying ``confirmed_dispatch_max_age_seconds`` only after heap
fetch. These tests pin the fix two ways: the index exists in ``pg_indexes``
after ``alembic upgrade head`` with the right key column and partial
predicate, and an ``EXPLAIN (FORMAT JSON)`` of the *exact*
``CONFIRMED_ROWS_SQL`` text yields a plan that prunes at the new index (an
``Index Cond`` mentioning ``created_at``), not a heap ``Filter`` after a full
scan. On master (index absent) the EXPLAIN assertion fails.

The plan-shape test seeds a production-shaped population (tens of thousands
of confirmed-allow rows on distinct sessions, ``created_at`` spread over 180
days, ``ANALYZE``d) rather than the bare 1-2 row chain: on a near-empty table
the planner has no cost signal to prefer ``events_tool_confirmed_allow_recent_idx``
over the pre-existing ``events_tool_confirmed_allow_idx`` (migration 0065,
keyed ``session_id``) and the choice is arbitrary either way, which is why an
earlier version of this test needed ``SET enable_seqscan = off`` to force the
new index in the plan — that forcing is gone; with a realistic rowcount and
age distribution the planner picks the ``created_at``-keyed index on cost
alone, because it prunes the overwhelming majority of confirmed-allow history
before ever reaching the heap.

Patterned exactly on ``tests/integration/test_migrations_0128_inbound_budget_index.py``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from typing import Any

import asyncpg
import pytest

from aios.db.queries.events import confirmed_unresolved_predicate
from aios.harness.sweep import CONFIRMED_ROWS_SQL
from tests.conftest import _docker_available, needs_docker
from tests.integration.test_migrations import _alembic_url, _run_alembic

_INDEX_NAME = "events_tool_confirmed_allow_recent_idx"

# A minimal account/agent/env/session chain so confirmed-allow lifecycle
# events have a session to hang off, plus the assistant/tool-result rows the
# predicate's JOIN and NOT EXISTS guard reference.
_CHAIN_SQL = """
INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
VALUES ('acc_root', NULL, TRUE, 'root');
INSERT INTO environments (id, name, account_id)
VALUES ('env_a', 'env-a', 'acc_root');
INSERT INTO agents (id, name, model, account_id)
VALUES ('agent_a', 'agent-a', 'test/model', 'acc_root');
INSERT INTO sessions (id, agent_id, environment_id, workspace_volume_path, account_id,
                      last_event_seq)
VALUES ('sess_a', 'agent_a', 'env_a', '/tmp/ws-a', 'acc_root', 3);
"""

_EVENTS_SQL = """
INSERT INTO events (id, session_id, seq, kind, data, role, account_id)
VALUES
  ('evt_asst', 'sess_a', 1, 'message',
   '{"role": "assistant", "content": "", "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]}'::jsonb,
   'assistant', 'acc_root'),
  ('evt_confirm', 'sess_a', 2, 'lifecycle',
   '{"event": "tool_confirmed", "result": "allow", "tool_call_id": "tc1"}'::jsonb,
   NULL, 'acc_root');
"""

# Realistic noise: a large population of *old* confirmed-allow rows (each on
# its own throwaway session, ``created_at`` spread over the last 180 days so
# only a small recent slice matches the age window). This is what makes the
# planner's choice between ``events_tool_confirmed_allow_idx`` (keyed
# ``session_id``, no ``created_at`` correlation) and
# ``events_tool_confirmed_allow_recent_idx`` (keyed ``created_at``) a genuine
# cost decision rather than an arbitrary tie on a near-empty table — on
# production-shaped data the age-keyed index is cheaper because it prunes the
# overwhelming majority of confirmed-allow history before the heap fetch.
_NOISE_ROW_COUNT = 100_000

_NOISE_SQL = f"""
INSERT INTO sessions (id, agent_id, environment_id, workspace_volume_path, account_id,
                      last_event_seq)
SELECT 'sess_noise_' || i, 'agent_a', 'env_a', '/tmp/ws-noise', 'acc_root', 1
FROM generate_series(1, {_NOISE_ROW_COUNT}) AS i;

INSERT INTO events (id, session_id, seq, kind, data, role, account_id, created_at)
SELECT 'evt_noise_' || i, 'sess_noise_' || i, 1, 'lifecycle',
       jsonb_build_object('event', 'tool_confirmed', 'result', 'allow',
                          'tool_call_id', 'tc_noise_' || i),
       NULL, 'acc_root', now() - (random() * interval '180 days')
FROM generate_series(1, {_NOISE_ROW_COUNT}) AS i;
"""

# The EXACT CONFIRMED_ROWS_SQL text the sweep runs, bound to a 1h age window —
# same rendering ``find_sessions_needing_inference`` performs (scope_clause
# empty for the unscoped sweep pass, age_param as $1).
_CONFIRMED_ROWS_SQL_TEXT = CONFIRMED_ROWS_SQL.format(scope_clause="", age_param="$1")


@pytest.fixture
def postgres() -> Iterator[object]:
    """Fresh function-scoped Postgres."""
    if not _docker_available():
        pytest.skip("Docker not available")
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("postgres:16-alpine") as pg:
        yield pg


async def _execute(db_url: str, sql: str) -> None:
    conn = await asyncpg.connect(db_url)
    try:
        await conn.execute(sql)
    finally:
        await conn.close()


async def _fetchval(db_url: str, sql: str, *args: Any) -> Any:
    conn = await asyncpg.connect(db_url)
    try:
        return await conn.fetchval(sql, *args)
    finally:
        await conn.close()


def _collect_nodes(plan_node: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the EXPLAIN plan tree into a list of nodes (pre-order)."""
    nodes = [plan_node]
    for child in plan_node.get("Plans", []):
        nodes.extend(_collect_nodes(child))
    return nodes


@needs_docker
@pytest.mark.integration
def test_clean_database_creates_confirmed_allow_recent_index(postgres: object) -> None:
    db_url = _alembic_url(postgres)

    up = _run_alembic(["upgrade", "head"], db_url)
    assert up.returncode == 0, f"upgrade to head failed:\n{up.stderr}\n{up.stdout}"

    indexdef = asyncio.run(
        _fetchval(
            db_url,
            "SELECT indexdef FROM pg_indexes WHERE tablename = 'events' AND indexname = $1",
            _INDEX_NAME,
        )
    )
    assert indexdef is not None, f"{_INDEX_NAME} missing after upgrade"
    indexdef = str(indexdef)
    # Single key column: created_at.
    assert "(created_at)" in indexdef, indexdef
    # The three partial-predicate clauses (mirrors 0065's WHERE).
    assert "WHERE" in indexdef, indexdef
    assert "kind = 'lifecycle'" in indexdef, indexdef
    assert "'event'::text) = 'tool_confirmed'" in indexdef, indexdef
    assert "'result'::text) = 'allow'" in indexdef, indexdef

    # 0065 must still exist — additive, not a replacement.
    old_indexdef = asyncio.run(
        _fetchval(
            db_url,
            "SELECT indexdef FROM pg_indexes WHERE tablename = 'events' "
            "AND indexname = 'events_tool_confirmed_allow_idx'",
        )
    )
    assert old_indexdef is not None, "migration 0065's index must remain (additive)"


@needs_docker
@pytest.mark.integration
def test_confirmed_rows_sql_uses_index_not_seq_scan(postgres: object) -> None:
    """The master-failing pin: EXPLAIN of the exact ``CONFIRMED_ROWS_SQL`` text
    must prune at the new index (``Index Cond`` on ``created_at``), never a
    ``Seq Scan`` over ``events`` for the ``lc`` node. On master (no index) the
    only access path is a ``Seq Scan`` and this assertion fails."""
    db_url = _alembic_url(postgres)

    up = _run_alembic(["upgrade", "head"], db_url)
    assert up.returncode == 0, f"upgrade to head failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(_execute(db_url, _CHAIN_SQL + _EVENTS_SQL + _NOISE_SQL))
    asyncio.run(_execute(db_url, "ANALYZE"))

    async def _plan() -> dict[str, Any]:
        conn = await asyncpg.connect(db_url)
        try:
            result = await conn.fetchval(
                f"EXPLAIN (FORMAT JSON) {_CONFIRMED_ROWS_SQL_TEXT}",
                3600,
            )
            if isinstance(result, str):
                result = json.loads(result)
            return result[0]["Plan"]  # type: ignore[no-any-return]
        finally:
            await conn.close()

    plan = asyncio.run(_plan())
    nodes = _collect_nodes(plan)

    # The ``lc``-aliased scan of events — identify it by its Index/Relation
    # name (there is also a nested-loop `s` join and a NOT EXISTS `tr` subplan,
    # both over other aliases of events/sessions).
    events_scans = [n for n in nodes if n.get("Relation Name") == "events"]
    assert events_scans, f"no scan node over events in plan: {plan}"

    seq_scans = [n for n in events_scans if n.get("Node Type") == "Seq Scan"]
    assert not seq_scans, (
        f"CONFIRMED_ROWS_SQL still seq-scans events (missing {_INDEX_NAME}?): "
        f"{[n.get('Node Type') for n in events_scans]}"
    )

    index_scans = [
        n
        for n in events_scans
        if n.get("Node Type") in ("Index Scan", "Index Only Scan", "Bitmap Index Scan")
        and n.get("Index Name") == _INDEX_NAME
    ]
    bitmap_index_scans = [
        n
        for n in nodes
        if n.get("Node Type") == "Bitmap Index Scan" and n.get("Index Name") == _INDEX_NAME
    ]
    assert index_scans or bitmap_index_scans, (
        f"plan does not seek {_INDEX_NAME} over events; "
        f"events scan nodes: {[(n.get('Node Type'), n.get('Index Name')) for n in events_scans]}"
    )

    # The age bound must prune AT the index (an Index Cond referencing
    # created_at), not merely ride along as a post-fetch Filter — that is the
    # whole point of dropping the ``IS NULL`` OR-arm (#1740).
    matched = index_scans or [
        n
        for n in nodes
        if n.get("Node Type") == "Bitmap Index Scan" and n.get("Index Name") == _INDEX_NAME
    ]
    index_conds = " ".join(str(n.get("Index Cond", "")) for n in matched)
    assert "created_at" in index_conds, (
        f"index scan on {_INDEX_NAME} does not prune on created_at "
        f"(Index Cond={index_conds!r}) — age bound is not sargable"
    )


def test_predicate_age_clause_has_no_is_null_or_arm() -> None:
    """Structural pin: the shared predicate's age clause must be a plain
    sargable range comparison — no ``IS NULL`` OR-arm — so it is prunable at
    a generic plan regardless of the specific bound value (#1740)."""
    pred = confirmed_unresolved_predicate("lc", "$1")
    assert "IS NULL" not in pred, pred
    assert "created_at >= now() - make_interval(secs => $1::bigint)" in pred, pred
