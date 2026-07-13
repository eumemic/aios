"""Migration 0128 adds ``events_inbound_budget_idx`` — the composite partial
index behind the per-counterparty inbound rate budget's rolling-window count
(#1557).

``_count_recent_inbounds`` leads its ``WHERE`` with the two equality predicates
``(account_id, orig_channel)``, and *no* existing ``events`` index leads with
that prefix, so before this migration the planner's only access path was a
sequential scan of the entire ``events`` log — paid synchronously on the hot
inbound admission path. These tests pin the fix two ways: the index exists in
``pg_indexes`` after ``alembic upgrade head``, and an ``EXPLAIN (FORMAT JSON)``
of the *exact* ``_count_recent_inbounds`` query (with real bind values) yields a
plan whose scan node over ``events`` is an index scan on the new index rather
than a ``Seq Scan``. On master (no index) the EXPLAIN assertion fails with a
``Seq Scan`` — this is the master-failing pin.

Mirrors the testcontainer-Postgres shape of
``test_migrations_0097_unique_tool_result.py`` and the ``EXPLAIN (FORMAT
JSON)`` plan-shape assertion style of ``tests/e2e/test_sweep_perf.py``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from typing import Any

import asyncpg
import pytest

from tests.conftest import _docker_available, needs_docker
from tests.integration.test_migrations import _alembic_url, _run_alembic

# A minimal account/agent/env/session chain so admitted-inbound ``role=user``
# events (and a wake-bearing lifecycle) have a session to hang off. The two
# inbound events share one ``orig_channel`` so the budget window has rows to
# count; the noise rows (assistant/tool/span/no-wake-lifecycle) exist so a
# planner running on real data would have a reason to seek rather than scan.
_CHAIN_SQL = """
INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
VALUES ('acc_root', NULL, TRUE, 'root');
INSERT INTO environments (id, name, account_id)
VALUES ('env_a', 'env-a', 'acc_root');
INSERT INTO agents (id, name, model, account_id)
VALUES ('agent_a', 'agent-a', 'test/model', 'acc_root');
INSERT INTO sessions (id, agent_id, environment_id, workspace_volume_path, account_id,
                      last_event_seq)
VALUES ('sess_a', 'agent_a', 'env_a', '/tmp/ws-a', 'acc_root', 4);
"""

_EVENTS_SQL = """
INSERT INTO events (id, session_id, seq, kind, data, role, account_id, orig_channel)
VALUES
  ('evt_in1', 'sess_a', 1, 'message',
   '{"role": "user", "content": "hi"}'::jsonb,
   'user', 'acc_root', 'slack/ext1/chat1'),
  ('evt_in2', 'sess_a', 2, 'message',
   '{"role": "user", "content": "hi again"}'::jsonb,
   'user', 'acc_root', 'slack/ext1/chat1'),
  ('evt_wake', 'sess_a', 3, 'lifecycle',
   '{"event": "external", "wake": true}'::jsonb,
   NULL, 'acc_root', 'slack/ext1/chat1'),
  ('evt_asst', 'sess_a', 4, 'message',
   '{"role": "assistant", "content": "hello"}'::jsonb,
   'assistant', 'acc_root', 'slack/ext1/chat1');

-- Keep the fixture production-shaped now that migration 0148 adds a second
-- inference-bearing partial index led by session_id. A single-session agent can
-- receive many counterparties; those rows must make the orig_channel equality
-- seek genuinely cheaper than scanning the whole agent window and filtering.
INSERT INTO events (id, session_id, seq, kind, data, role, account_id, orig_channel)
SELECT 'evt_other_' || n, 'sess_a', n, 'message',
       '{"role": "user", "content": "other chat"}'::jsonb,
       'user', 'acc_root', 'slack/ext1/other-' || n
FROM generate_series(5, 1004) AS n;
"""

# The EXACT ``_count_recent_inbounds`` query (predicate copied verbatim from
# ``inbound_budget._INFERENCE_BEARING_PREDICATE``), so the plan we assert on is
# the plan production runs — not a paraphrase.
_COUNT_SQL = """
SELECT count(*)
FROM events
WHERE account_id = $1
  AND orig_channel = $2
  AND ( (kind = 'message'   AND data->>'role' = 'user')
     OR (kind = 'lifecycle' AND (data->>'wake')::boolean IS TRUE) )
  AND created_at > now() - make_interval(secs => $3::bigint)
"""

_INDEX_NAME = "events_inbound_budget_idx"


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
def test_clean_database_creates_inbound_budget_index(postgres: object) -> None:
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
    # Key columns in order, and partial predicate matching the query's rows.
    assert "account_id, orig_channel, created_at" in indexdef, indexdef
    assert "WHERE" in indexdef and "kind = 'message'" in indexdef, indexdef
    assert "'role'::text) = 'user'" in indexdef, indexdef
    assert "kind = 'lifecycle'" in indexdef and "'wake'" in indexdef, indexdef


@needs_docker
@pytest.mark.integration
def test_count_query_uses_index_not_seq_scan(postgres: object) -> None:
    """The master-failing pin: EXPLAIN of the exact ``_count_recent_inbounds``
    query must seek the new index over ``events``, never a ``Seq Scan``. On
    master (no index) the only ``events`` scan is a ``Seq Scan`` and this
    assertion fails."""
    db_url = _alembic_url(postgres)

    up = _run_alembic(["upgrade", "head"], db_url)
    assert up.returncode == 0, f"upgrade to head failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(_execute(db_url, _CHAIN_SQL + _EVENTS_SQL))

    async def _plan() -> dict[str, Any]:
        conn = await asyncpg.connect(db_url)
        try:
            # Disable seq/bitmap-heap fallbacks the tiny test table would
            # otherwise pick on cost alone, so the plan reflects the index's
            # *applicability* to the predicate (the property #1557 is about),
            # not the planner's row-count heuristics on a 4-row table.
            await conn.execute("ANALYZE events")
            await conn.execute("SET enable_seqscan = off")
            result = await conn.fetchval(
                f"EXPLAIN (FORMAT JSON) {_COUNT_SQL}",
                "acc_root",
                "slack/ext1/chat1",
                3600,
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

    seq_scans = [n for n in events_scans if n.get("Node Type") == "Seq Scan"]
    assert not seq_scans, (
        f"query still seq-scans events (missing {_INDEX_NAME}?): "
        f"{[n.get('Node Type') for n in events_scans]}"
    )

    index_scans = [
        n
        for n in events_scans
        if n.get("Node Type") in ("Index Scan", "Index Only Scan", "Bitmap Index Scan")
        and n.get("Index Name") == _INDEX_NAME
    ]
    # Bitmap Index Scan lives on the node with the Index Name; the parent
    # Bitmap Heap Scan carries the Relation Name. Cover both shapes.
    bitmap_index_scans = [
        n
        for n in nodes
        if n.get("Node Type") == "Bitmap Index Scan" and n.get("Index Name") == _INDEX_NAME
    ]
    assert index_scans or bitmap_index_scans, (
        f"plan does not seek {_INDEX_NAME} over events; "
        f"events scan nodes: {[(n.get('Node Type'), n.get('Index Name')) for n in events_scans]}"
    )
