"""Migration 0135 adds ``events_session_lifecycle_seq_idx`` — the partial
index behind the lifecycle arm of ``read_windowed_context_events`` (#1741).

Before this migration the lifecycle arm (``kind = 'lifecycle' AND
data->>'event' = ANY(...)``) had no supporting partial index:
``events_turn_error_idx`` (migration 0062) carries the narrower
``data->>'stop_reason' = 'error'`` predicate, so the planner falls back to a
non-partial index that only seeks ``session_id`` (``events_session_created_at_idx``
from migration 0022, or the ``(session_id, seq)`` unique index) and heap-filters
``kind = 'lifecycle'`` across the whole session slate. These tests pin the fix
two ways: the index exists in ``pg_indexes`` after ``alembic upgrade head``, and
an ``EXPLAIN (FORMAT JSON)`` of the exact ``drop=None`` lifecycle-arm query
(with real bind values) shows an ``events`` scan carrying no residual
``Filter: (kind = 'lifecycle')`` — i.e. the partial index's own predicate
absorbed it. On master (no index) the plan heap-filters kind, so this is the
master-failing pin. Also asserts round-trip ``upgrade``/``downgrade`` both
succeed with the ``CREATE/DROP INDEX CONCURRENTLY`` form.

The plan-shape test seeds a **production-shaped** slate — messages dominate
(``kind='message'``), lifecycle rows are the minority the partial index prunes
to — rather than an all-lifecycle table. On an all-lifecycle fixture the
partial ``WHERE kind='lifecycle'`` index covers ~100% of the session, so it has
no selectivity edge over the pre-existing ``(session_id, created_at)`` index and
the planner picks either within cost noise (it chose ``created_at`` + a seq
``Sort``, defeating the assertion). Only when messages dominate does the
lifecycle-only partial index become the genuine cost winner — scanning just the
lifecycle rows in ``seq`` order (no ``Sort``, no ``kind`` heap-filter) — which
is exactly the property #1741 is about and the shape a real heavy session has.
Mirrors the sibling ``test_migrations_0134_confirmed_allow_recent_index.py`` /
e2e ``TestLifecycleArmPlanShapeGate`` production-shaped-fixture approach.

Mirrors ``test_migrations_0128_inbound_budget_index.py``'s testcontainer +
``EXPLAIN (FORMAT JSON)`` plan-shape style.
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

_INDEX_NAME = "events_session_lifecycle_seq_idx"

# Migration 0145 adds a strictly-narrower partial index over the SAME
# ``(session_id, seq)`` columns but with the additional
# ``AND data->>'event' IN (<MODEL_VISIBLE_LIFECYCLE_EVENTS>)`` predicate — the
# model-visible subset the lifecycle arm actually queries. It is not
# byte-redundant with 0135's generic index (its predicate is a strict subset),
# and the planner correctly PREFERS it for this query because it matches the
# query's own ``data->>'event' = ANY($3)`` allowlist and scans far fewer rows.
# Either partial index serves the lifecycle arm index-only (no seq scan, no
# ``kind`` heap-filter), which is the invariant this test guards.
_MODEL_VISIBLE_INDEX_NAME = "events_session_model_visible_lifecycle_seq_idx"
_LIFECYCLE_ARM_SERVING_INDEXES = frozenset({_INDEX_NAME, _MODEL_VISIBLE_INDEX_NAME})

# A minimal account/agent/env/session chain, then many lifecycle rows so a
# real planner (not a 1-row toy table) has a reason to seek rather than scan.
_N_EVENTS = 10_000  # total slate; ~1/5 lifecycle, ~4/5 message

_CHAIN_SQL = f"""
INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
VALUES ('acc_root', NULL, TRUE, 'root');
INSERT INTO environments (id, name, account_id)
VALUES ('env_a', 'env-a', 'acc_root');
INSERT INTO agents (id, name, model, account_id)
VALUES ('agent_a', 'agent-a', 'test/model', 'acc_root');
INSERT INTO sessions (id, agent_id, environment_id, workspace_volume_path, account_id,
                      last_event_seq)
VALUES ('sess_a', 'agent_a', 'env_a', '/tmp/ws-a', 'acc_root', {_N_EVENTS});
"""

# A production-shaped slate: {_N_EVENTS} events interleaved by seq, of which
# ~4/5 are ``kind='message'`` (the dominant kind in a real session) and ~1/5
# are ``kind='lifecycle'`` — mostly request_opened/request_response (the
# per-inference noise #1741 calls out; NOT in MODEL_VISIBLE_LIFECYCLE_EVENTS),
# plus a handful of model-visible sandbox_fs_reset notices. The message
# majority is what makes the lifecycle-ONLY partial index the genuine cost
# winner for the lifecycle arm (see module docstring): a whole-session index
# would have to scan the messages too and then Sort by seq.
_EVENTS_SQL = f"""
INSERT INTO events (id, session_id, seq, kind, data, role, account_id, cumulative_tokens)
SELECT
    'evt_' || g,
    'sess_a',
    g,
    CASE WHEN g % 5 = 0 THEN 'lifecycle' ELSE 'message' END,
    CASE
        WHEN g % 5 <> 0
             THEN '{{"role": "user", "content": "hi"}}'::jsonb
        WHEN g % 1000 = 0
             THEN '{{"event": "sandbox_fs_reset", "reason": "test"}}'::jsonb
        WHEN g % 10 = 0
             THEN '{{"event": "request_opened"}}'::jsonb
        ELSE '{{"event": "request_response"}}'::jsonb
    END,
    CASE WHEN g % 5 = 0 THEN NULL ELSE 'user' END,
    'acc_root',
    CASE WHEN g % 5 = 0 THEN NULL ELSE g * 10 END
FROM generate_series(1, {_N_EVENTS}) AS g;
"""

# The exact ``drop=None`` lifecycle arm of ``read_windowed_context_events``
# (query text copied verbatim from ``src/aios/db/queries/events.py``).
_LIFECYCLE_ARM_SQL = """
SELECT * FROM events
WHERE session_id = $1 AND account_id = $2
AND kind = 'lifecycle' AND data->>'event' = ANY($3)
ORDER BY seq ASC
"""


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
def test_upgrade_creates_lifecycle_index(postgres: object) -> None:
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
    assert "session_id, seq" in indexdef, indexdef
    assert "WHERE" in indexdef and "kind = 'lifecycle'" in indexdef, indexdef
    # Generic predicate only — no hardcoded event-name allowlist baked in.
    assert "event" not in indexdef.split("WHERE")[1], indexdef


@needs_docker
@pytest.mark.integration
def test_upgrade_downgrade_roundtrip(postgres: object) -> None:
    db_url = _alembic_url(postgres)

    up = _run_alembic(["upgrade", "head"], db_url)
    assert up.returncode == 0, f"upgrade to head failed:\n{up.stderr}\n{up.stdout}"

    # Target this migration's own down_revision explicitly rather than the
    # relative "-1" — head has since grown a sibling migration (0136,
    # renumbered from 0132/0134 per #1746) chained after this one, so "-1"
    # from head would only undo 0136 and leave this index in place.
    down = _run_alembic(["downgrade", "0134"], db_url)
    assert down.returncode == 0, f"downgrade failed:\n{down.stderr}\n{down.stdout}"

    indexdef = asyncio.run(
        _fetchval(
            db_url,
            "SELECT indexdef FROM pg_indexes WHERE tablename = 'events' AND indexname = $1",
            _INDEX_NAME,
        )
    )
    assert indexdef is None, f"{_INDEX_NAME} still present after downgrade"

    up_again = _run_alembic(["upgrade", "head"], db_url)
    assert up_again.returncode == 0, (
        f"re-upgrade after downgrade failed:\n{up_again.stderr}\n{up_again.stdout}"
    )


@needs_docker
@pytest.mark.integration
def test_lifecycle_arm_uses_index_not_seq_scan(postgres: object) -> None:
    """The master-failing pin: EXPLAIN of the exact lifecycle arm must not
    heap-filter ``kind = 'lifecycle'`` over a whole-session scan. On master
    (no partial index) the ``events`` scan carries a residual
    ``Filter: (kind = 'lifecycle')`` over a non-partial session index
    (``events_session_created_at_idx`` + a seq ``Sort``, or the ``(session_id,
    seq)`` unique index); after the fix the lifecycle-only partial index
    absorbs the predicate and no such residual filter remains. The fixture is
    production-shaped (messages dominate) so this is a genuine cost decision:
    on an all-lifecycle table the partial index has no selectivity edge and
    the planner picks ``created_at`` within cost noise (see module docstring)."""
    db_url = _alembic_url(postgres)

    up = _run_alembic(["upgrade", "head"], db_url)
    assert up.returncode == 0, f"upgrade to head failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(_execute(db_url, _CHAIN_SQL + _EVENTS_SQL))

    from aios.models.events import MODEL_VISIBLE_LIFECYCLE_EVENTS

    allowlist = list(MODEL_VISIBLE_LIFECYCLE_EVENTS)

    async def _plan() -> dict[str, Any]:
        conn = await asyncpg.connect(db_url)
        try:
            await conn.execute("ANALYZE events")
            # Disable seq/bitmap-heap fallbacks so an index scan is compared
            # against an index scan: on the production-shaped slate the
            # lifecycle-only partial index is the genuine cost winner over the
            # whole-session ``created_at`` index (which must also Sort by seq),
            # so the plan reflects the index's applicability to the predicate
            # (the property #1741 is about), not seq/bitmap cost heuristics.
            await conn.execute("SET enable_seqscan = off")
            await conn.execute("SET enable_bitmapscan = off")
            result = await conn.fetchval(
                f"EXPLAIN (FORMAT JSON) {_LIFECYCLE_ARM_SQL}",
                "sess_a",
                "acc_root",
                allowlist,
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

    for scan in events_scans:
        filt = str(scan.get("Filter", ""))
        assert "kind" not in filt, (
            f"lifecycle arm still heap-filters kind over an unbounded scan "
            f"(missing a lifecycle-serving partial index "
            f"{sorted(_LIFECYCLE_ARM_SERVING_INDEXES)}?): {scan}"
        )

    index_names = {n.get("Index Name") for n in nodes if n.get("Index Name")}
    assert index_names & _LIFECYCLE_ARM_SERVING_INDEXES, (
        f"plan does not use a lifecycle-arm serving index "
        f"({sorted(_LIFECYCLE_ARM_SERVING_INDEXES)}); scan nodes: "
        f"{[(n.get('Node Type'), n.get('Index Name')) for n in events_scans]}"
    )
