from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from aios.db.queries.triggers import (
    mark_trigger_run_woken_by_workflow_session,
    record_trigger_run,
)


@pytest.mark.asyncio
async def test_workflow_wake_is_persisted_before_audit_update() -> None:
    conn = AsyncMock()
    conn.fetch.return_value = []

    await mark_trigger_run_woken_by_workflow_session(conn, "sess_child")

    assert conn.method_calls[0][0] == "execute"
    marker_sql = conn.execute.await_args.args[0]
    assert "INSERT INTO workflow_run_owner_wakes" in marker_sql
    assert "ON CONFLICT (workflow_run_id) DO NOTHING" in marker_sql

    assert conn.method_calls[1][0] == "fetch"
    update_sql = conn.fetch.await_args.args[0]
    assert "UPDATE trigger_runs" in update_sql
    assert "s.parent_run_id = tr.result_id" in update_sql


@pytest.mark.asyncio
async def test_cron_audit_reconciles_an_early_workflow_wake() -> None:
    conn = AsyncMock()

    await record_trigger_run(
        conn,
        trigger_id="trg_test",
        account_id="acc_test",
        owner_session_id="sess_owner",
        trigger_name="watchdog",
        trigger_context="cron",
        status="ok",
        error_summary=None,
        result_id="wfr_test",
        started_at=datetime.now(UTC),
    )

    insert_sql = conn.execute.await_args.args[0]
    assert "SELECT 1 FROM workflow_run_owner_wakes" in insert_sql
    # ``result_id`` is inserted into an unconstrained text column and otherwise
    # compared only with NULL when this branch short-circuits. Keep an explicit
    # type anchor so asyncpg/Postgres can prepare non-workflow timer fires too.
    assert "$9::text IS NOT NULL" in insert_sql
    assert "workflow_run_id = $9::text" in insert_sql
