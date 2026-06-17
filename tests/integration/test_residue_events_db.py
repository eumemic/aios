"""DB-backed tests for the residue_events gauge (#1328).

Exercises the append-only invariant, the CHECK vocabularies, the
stamped-at-source axis-2 ingest, the axis-1 observer ingest, idempotency, and the
uncorrelated run-table denominator — all against real Postgres, no live model.
(The pure-Python render / axis-segregation / migration-predicate tests live in
``tests/unit/``.)
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
import pytest

from aios.db.pool import create_pool
from aios.db.queries import residue

pytestmark = pytest.mark.integration

ACCOUNT = "acc_residue"


@pytest.fixture
async def pool(migrated_db_url: str, _reset_db_state: None) -> AsyncIterator[asyncpg.Pool[Any]]:
    p = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with p.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ($1, NULL, TRUE, 'residue-root')",
                ACCOUNT,
            )
            await conn.execute(
                "INSERT INTO environments (id, name, config, account_id) "
                "VALUES ('env_res', 'res-env', '{}'::jsonb, $1)",
                ACCOUNT,
            )
        yield p
    finally:
        await p.close()


def _window() -> datetime:
    return datetime.now(UTC) - timedelta(days=30)


async def _insert_axis2(conn: asyncpg.Connection[Any], **kw: Any) -> bool:
    base: dict[str, Any] = dict(
        account_id=ACCOUNT,
        source_gate_nonce="gn_1",
        gate_kind="design",
        result={"approve": True, "residue_kind": "design-judgment"},
        finder="chairman",
    )
    base.update(kw)
    return await residue.ingest_gate_resume_axis2(conn, **base)


# ─── AC1 (table exists) / AC3 (CHECK vocabularies reject bad inserts) ─────────


async def test_insert_basic_row_and_open_residue_kind(pool: asyncpg.Pool[Any]) -> None:
    async with pool.acquire() as conn:
        # An ``other`` row inserts cleanly — residue_kind is an OPEN enum (no CHECK).
        inserted = await residue.insert_residue_event(
            conn,
            account_id=ACCOUNT,
            axis=residue.AXIS_CLASS_MIGRATION,
            finder="chairman",
            residue_kind="other",
            kind_source=residue.KIND_SOURCE_MANUAL,
            signature={"note": "irreducible"},
            idempotency_key="k_other",
        )
        assert inserted is True


async def test_axis_check_rejects_out_of_range(pool: asyncpg.Pool[Any]) -> None:
    async with pool.acquire() as conn:
        with pytest.raises(asyncpg.IntegrityConstraintViolationError):
            await residue.insert_residue_event(
                conn,
                account_id=ACCOUNT,
                axis=3,  # not in (1,2)
                finder="chairman",
                residue_kind="other",
                kind_source=residue.KIND_SOURCE_MANUAL,
                signature={},
                idempotency_key="k_axis3",
            )


async def test_finder_check_rejects_unknown_finder(pool: asyncpg.Pool[Any]) -> None:
    async with pool.acquire() as conn:
        with pytest.raises(asyncpg.IntegrityConstraintViolationError):
            await residue.insert_residue_event(
                conn,
                account_id=ACCOUNT,
                axis=1,
                finder="some-new-finder",
                residue_kind="other",
                kind_source=residue.KIND_SOURCE_OBSERVER,
                signature={},
                idempotency_key="k_badfinder",
            )


async def test_kind_source_check_rejects_unknown_source(pool: asyncpg.Pool[Any]) -> None:
    async with pool.acquire() as conn:
        with pytest.raises(asyncpg.IntegrityConstraintViolationError):
            await residue.insert_residue_event(
                conn,
                account_id=ACCOUNT,
                axis=1,
                finder="chairman",
                residue_kind="other",
                kind_source="prose-inference",  # forbidden
                signature={},
                idempotency_key="k_badsrc",
            )


# ─── AC2 (append-only: UPDATE and DELETE both RAISE via the trigger) ──────────


async def test_update_is_rejected_by_trigger(pool: asyncpg.Pool[Any]) -> None:
    async with pool.acquire() as conn:
        await residue.insert_residue_event(
            conn,
            account_id=ACCOUNT,
            axis=2,
            finder="chairman",
            residue_kind="design-judgment",
            kind_source=residue.KIND_SOURCE_GATE,
            signature={},
            idempotency_key="k_upd",
        )
        with pytest.raises(asyncpg.PostgresError) as exc:
            await conn.execute(
                "UPDATE residue_events SET finder = 'external-world' WHERE account_id = $1",
                ACCOUNT,
            )
        assert "append-only" in str(exc.value)


async def test_delete_is_rejected_by_trigger(pool: asyncpg.Pool[Any]) -> None:
    async with pool.acquire() as conn:
        await residue.insert_residue_event(
            conn,
            account_id=ACCOUNT,
            axis=2,
            finder="chairman",
            residue_kind="design-judgment",
            kind_source=residue.KIND_SOURCE_GATE,
            signature={},
            idempotency_key="k_del",
        )
        with pytest.raises(asyncpg.PostgresError) as exc:
            await conn.execute("DELETE FROM residue_events WHERE account_id = $1", ACCOUNT)
        assert "append-only" in str(exc.value)


# ─── AC7 (axis-2 ingest copies residue_kind verbatim; kind_source gate) ───────


async def test_axis2_ingest_copies_stamp_verbatim(pool: asyncpg.Pool[Any]) -> None:
    async with pool.acquire() as conn:
        inserted = await _insert_axis2(
            conn,
            result={"approve": True, "residue_kind": "design-judgment"},
        )
        assert inserted is True
        row = await conn.fetchrow(
            "SELECT axis, residue_kind, kind_source, finder FROM residue_events "
            "WHERE account_id = $1",
            ACCOUNT,
        )
        assert row["axis"] == residue.AXIS_CLASS_MIGRATION
        assert row["residue_kind"] == "design-judgment"  # copied, never inferred
        assert row["kind_source"] == residue.KIND_SOURCE_GATE
        assert row["finder"] == "chairman"


async def test_axis2_ingest_skips_non_human_in_loop_kind(pool: asyncpg.Pool[Any]) -> None:
    async with pool.acquire() as conn:
        # A verify/merge_guard gate is NOT a residue event.
        inserted = await _insert_axis2(conn, gate_kind="verify", result={"override": False})
        assert inserted is False
        n = await conn.fetchval("SELECT count(*) FROM residue_events")
        assert n == 0


async def test_axis2_ingest_missing_stamp_refuses_to_fabricate(pool: asyncpg.Pool[Any]) -> None:
    async with pool.acquire() as conn:
        with pytest.raises(ValueError):
            await _insert_axis2(conn, result={"approve": True})  # no residue_kind


# ─── AC10 (idempotency: replaying the same gate inserts ONE row) ──────────────


async def test_axis2_ingest_is_idempotent(pool: asyncpg.Pool[Any]) -> None:
    async with pool.acquire() as conn:
        first = await _insert_axis2(conn, source_gate_nonce="gn_dup")
        second = await _insert_axis2(conn, source_gate_nonce="gn_dup")
        assert first is True
        assert second is False  # ON CONFLICT DO NOTHING
        n = await conn.fetchval("SELECT count(*) FROM residue_events")
        assert n == 1


async def test_observer_ingest_is_idempotent(pool: asyncpg.Pool[Any]) -> None:
    async with pool.acquire() as conn:
        sig = {"anomaly": "cost_band", "run_id": "wfr_x"}
        a = await residue.ingest_observer_verdict_axis1(
            conn, account_id=ACCOUNT, source_run_id="wfr_x", verdict="anomaly", signature=sig
        )
        b = await residue.ingest_observer_verdict_axis1(
            conn, account_id=ACCOUNT, source_run_id="wfr_x", verdict="anomaly", signature=sig
        )
        assert a is True and b is False
        n = await conn.fetchval("SELECT count(*) FROM residue_events")
        assert n == 1


# ─── AC12 (axis-1 observer ingest: anomaly row; cannot-determine; ok→no row) ──


async def test_observer_anomaly_writes_axis1_row(pool: asyncpg.Pool[Any]) -> None:
    async with pool.acquire() as conn:
        sig = {"anomaly": "token_band", "run_id": "wfr_a", "band": [0, 100]}
        inserted = await residue.ingest_observer_verdict_axis1(
            conn, account_id=ACCOUNT, source_run_id="wfr_a", verdict="anomaly", signature=sig
        )
        assert inserted is True
        row = await conn.fetchrow(
            "SELECT axis, finder, kind_source, residue_kind, signature "
            "FROM residue_events WHERE account_id = $1",
            ACCOUNT,
        )
        assert row["axis"] == residue.AXIS_OBSERVER
        assert row["finder"] == "internal-armed-check"
        assert row["kind_source"] == residue.KIND_SOURCE_OBSERVER
        assert row["residue_kind"] == "uncorrelated-detection"
        assert row["signature"]["anomaly"] == "token_band"  # fingerprint populated


async def test_observer_cannot_determine_writes_fail_loud_row_not_ok(
    pool: asyncpg.Pool[Any],
) -> None:
    async with pool.acquire() as conn:
        inserted = await residue.ingest_observer_verdict_axis1(
            conn,
            account_id=ACCOUNT,
            source_run_id="wfr_null",
            verdict="cannot-determine",
            signature={"reason": "telemetry null"},
        )
        assert inserted is True
        row = await conn.fetchrow(
            "SELECT residue_kind FROM residue_events WHERE account_id = $1", ACCOUNT
        )
        # A null-telemetry verdict is a cannot-determine row — NEVER an ok/clean row.
        assert row["residue_kind"] == residue.CANNOT_DETERMINE_KIND


async def test_observer_ok_writes_no_row(pool: asyncpg.Pool[Any]) -> None:
    async with pool.acquire() as conn:
        inserted = await residue.ingest_observer_verdict_axis1(
            conn, account_id=ACCOUNT, source_run_id="wfr_ok", verdict="ok", signature={}
        )
        assert inserted is False
        n = await conn.fetchval("SELECT count(*) FROM residue_events")
        assert n == 0


# ─── AC8 (run-table denominator from the uncorrelated wf_runs substrate) ──────


async def _seed_run(
    conn: asyncpg.Connection[Any], run_id: str, status: str, *, age_days: int = 0
) -> None:
    # First a workflow row to satisfy the FK.
    await conn.execute(
        "INSERT INTO workflows (id, account_id, name, script) VALUES "
        "($1, $2, $3, 'async def main(input):\n    return None') "
        "ON CONFLICT DO NOTHING",
        "wf_res",
        ACCOUNT,
        "res-wf",
    )
    created = datetime.now(UTC) - timedelta(days=age_days)
    await conn.execute(
        "INSERT INTO wf_runs (id, workflow_id, account_id, environment_id, script, "
        "script_sha, host_semantics_epoch, status, created_at) "
        "VALUES ($1, 'wf_res', $2, 'env_res', 'src', 'sha', 0, $3, $4)",
        run_id,
        ACCOUNT,
        status,
        created,
    )


async def test_run_table_denominator_counts_terminal_runs_only(
    pool: asyncpg.Pool[Any],
) -> None:
    async with pool.acquire() as conn:
        await _seed_run(conn, "wfr_1", "completed")
        await _seed_run(conn, "wfr_2", "errored")
        await _seed_run(conn, "wfr_3", "cancelled")
        await _seed_run(conn, "wfr_4", "running")  # non-terminal
        await _seed_run(conn, "wfr_5", "suspended")  # non-terminal
        denom = await residue.run_table_denominator(
            conn, account_id=ACCOUNT, window_start=_window()
        )
        assert denom == 3  # only the 3 terminal runs


async def test_denominator_independent_of_ops_agent_event_count(
    pool: asyncpg.Pool[Any],
) -> None:
    """The denominator comes from wf_runs, NOT from how many residue rows the
    ops-agent ingested — fewer residue rows than runs, denominator still == runs."""
    async with pool.acquire() as conn:
        for i in range(5):
            await _seed_run(conn, f"wfr_t{i}", "completed")
        # Ingest only ONE residue classification (far fewer than 5 runs).
        await residue.ingest_observer_verdict_axis1(
            conn,
            account_id=ACCOUNT,
            source_run_id="wfr_t0",
            verdict="anomaly",
            signature={"x": 1},
        )
        denom = await residue.run_table_denominator(
            conn, account_id=ACCOUNT, window_start=_window()
        )
        assert denom == 5  # the run count, not the residue-row count


async def test_run_table_denominator_respects_window(pool: asyncpg.Pool[Any]) -> None:
    async with pool.acquire() as conn:
        await _seed_run(conn, "wfr_recent", "completed", age_days=1)
        await _seed_run(conn, "wfr_old", "completed", age_days=90)  # before the window
        denom = await residue.run_table_denominator(
            conn, account_id=ACCOUNT, window_start=_window()
        )
        assert denom == 1


# ─── found-by-finder is axis-scoped (never cross-axis) ────────────────────────


async def test_found_by_finder_is_axis_scoped(pool: asyncpg.Pool[Any]) -> None:
    async with pool.acquire() as conn:
        await residue.insert_residue_event(
            conn,
            account_id=ACCOUNT,
            axis=1,
            finder="internal-armed-check",
            residue_kind="uncorrelated-detection",
            kind_source=residue.KIND_SOURCE_OBSERVER,
            signature={},
            idempotency_key="a1",
        )
        await residue.insert_residue_event(
            conn,
            account_id=ACCOUNT,
            axis=2,
            finder="chairman",
            residue_kind="design-judgment",
            kind_source=residue.KIND_SOURCE_GATE,
            signature={},
            idempotency_key="a2",
        )
        axis1 = await residue.found_by_finder(
            conn, account_id=ACCOUNT, axis=1, window_start=_window()
        )
        axis2 = await residue.found_by_finder(
            conn, account_id=ACCOUNT, axis=2, window_start=_window()
        )
        assert axis1 == {"internal-armed-check": 1}
        assert axis2 == {"chairman": 1}  # the two axes never mingle


async def test_found_by_finder_excludes_cannot_determine(pool: asyncpg.Pool[Any]) -> None:
    async with pool.acquire() as conn:
        await residue.ingest_observer_verdict_axis1(
            conn,
            account_id=ACCOUNT,
            source_run_id="wfr_cd",
            verdict="cannot-determine",
            signature={},
        )
        breakdown = await residue.found_by_finder(
            conn, account_id=ACCOUNT, axis=1, window_start=_window()
        )
        # cannot-determine is surfaced on its own line, not as a clean finder hit.
        assert breakdown == {}
        cd = await residue.kind_count(
            conn,
            account_id=ACCOUNT,
            axis=1,
            residue_kind=residue.CANNOT_DETERMINE_KIND,
            window_start=_window(),
        )
        assert cd == 1
