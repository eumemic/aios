"""call_llm run-level inference-cost meter (#1633), against a real Postgres.

Proves migration 0125 applies and that the per-run ``call_llm_cost_microusd``
meter is read/charged by the dedicated query helpers — the ledger the over-budget
gate sums alongside the child-session rollup so a budget-exhausted run refuses
further ``call_llm``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db.pool import register_jsonb_codec
from aios.db.queries import workflows as wf_queries
from aios.workflows.determinism import HOST_SEMANTICS_EPOCH


@pytest.fixture
async def wf_conn(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[asyncpg.Connection[Any]]:
    conn = await asyncpg.connect(migrated_db_url)
    await register_jsonb_codec(conn)
    try:
        await conn.execute(
            "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
            "VALUES ('acc_root', NULL, TRUE, 'tenant-root')"
        )
        await conn.execute(
            "INSERT INTO environments (id, name, config, account_id) "
            "VALUES ('env_root', 'wf-env', '{}'::jsonb, 'acc_root')"
        )
        yield conn
    finally:
        await conn.close()


async def _seed_run(conn: asyncpg.Connection[Any], *, budget_usd: float | None = None) -> str:
    wf = await wf_queries.insert_workflow(
        conn, account_id="acc_root", name="demo", script="async def main(input):\n    return 1\n"
    )
    run = await wf_queries.insert_wf_run(
        conn,
        account_id="acc_root",
        workflow_id=wf.id,
        environment_id="env_root",
        script=wf.script,
        host_semantics_epoch=HOST_SEMANTICS_EPOCH,
        script_sha="deadbeef",
        budget_usd=budget_usd,
        depth=10,
    )
    return run.id


async def test_meter_defaults_to_zero(wf_conn: asyncpg.Connection[Any]) -> None:
    run_id = await _seed_run(wf_conn)
    # A fresh run has spent nothing on call_llm — the column is NOT NULL DEFAULT 0.
    assert (
        await wf_queries.get_run_call_llm_cost_microusd(
            wf_conn, run_id, account_id="acc_root"
        )
        == 0
    )
    # And it round-trips onto the read model.
    run = await wf_queries.get_wf_run(wf_conn, run_id, account_id="acc_root")
    assert run.call_llm_cost_microusd == 0


async def test_charge_accumulates_atomically(wf_conn: asyncpg.Connection[Any]) -> None:
    run_id = await _seed_run(wf_conn)
    await wf_queries.add_run_call_llm_cost_microusd(
        wf_conn, run_id, 1500, account_id="acc_root"
    )
    await wf_queries.add_run_call_llm_cost_microusd(
        wf_conn, run_id, 2500, account_id="acc_root"
    )
    # Each charge is col = col + delta — the two accumulate, never clobber.
    assert (
        await wf_queries.get_run_call_llm_cost_microusd(
            wf_conn, run_id, account_id="acc_root"
        )
        == 4000
    )


async def test_zero_or_negative_charge_is_noop(wf_conn: asyncpg.Connection[Any]) -> None:
    run_id = await _seed_run(wf_conn)
    # An unreported LiteLLM cost (None → 0) or a degenerate negative never writes.
    await wf_queries.add_run_call_llm_cost_microusd(wf_conn, run_id, 0, account_id="acc_root")
    await wf_queries.add_run_call_llm_cost_microusd(wf_conn, run_id, -5, account_id="acc_root")
    assert (
        await wf_queries.get_run_call_llm_cost_microusd(
            wf_conn, run_id, account_id="acc_root"
        )
        == 0
    )


async def test_meter_is_account_scoped(wf_conn: asyncpg.Connection[Any]) -> None:
    run_id = await _seed_run(wf_conn)
    await wf_queries.add_run_call_llm_cost_microusd(
        wf_conn, run_id, 9000, account_id="acc_root"
    )
    # A foreign-account read never sees the spend (and a foreign-account charge no-ops).
    assert (
        await wf_queries.get_run_call_llm_cost_microusd(
            wf_conn, run_id, account_id="acc_other"
        )
        == 0
    )
    await wf_queries.add_run_call_llm_cost_microusd(
        wf_conn, run_id, 1000, account_id="acc_other"
    )
    assert (
        await wf_queries.get_run_call_llm_cost_microusd(
            wf_conn, run_id, account_id="acc_root"
        )
        == 9000
    )


async def test_budget_gate_sums_meter_with_child_rollup(
    wf_conn: asyncpg.Connection[Any],
) -> None:
    # The acceptance invariant at the query layer: the over-budget decision is the SUM
    # of the child-session rollup and the run's own call_llm meter. With no child
    # sessions, a $0.50 budget and $0.60 of call_llm spend is over budget.
    run_id = await _seed_run(wf_conn, budget_usd=0.50)
    await wf_queries.add_run_call_llm_cost_microusd(
        wf_conn, run_id, 600_000, account_id="acc_root"
    )
    children = await wf_queries.run_children_usage(wf_conn, run_id, account_id="acc_root")
    meter = await wf_queries.get_run_call_llm_cost_microusd(
        wf_conn, run_id, account_id="acc_root"
    )
    run = await wf_queries.get_wf_run(wf_conn, run_id, account_id="acc_root")
    assert run.budget_usd is not None
    budget_total_microusd = round(run.budget_usd * 1_000_000)
    spent = children.cost_microusd + meter
    assert spent == 600_000
    assert spent >= budget_total_microusd  # → the gate refuses further call_llm
