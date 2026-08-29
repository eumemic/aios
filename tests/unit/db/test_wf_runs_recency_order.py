"""Regression coverage for the workflow-run collection ordering contract."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from aios.db.queries import workflows


@pytest.mark.asyncio
async def test_list_wf_runs_orders_by_created_at_with_stable_keyset() -> None:
    conn = AsyncMock()
    conn.fetch.return_value = []

    await workflows.list_wf_runs(
        conn,
        account_id="acc_1",
        workflow_id="wf_1",
        after="wfr_anchor",
        limit=10,
    )

    sql, *args = conn.fetch.await_args.args
    normalized = " ".join(sql.split())
    assert "(created_at, id) < (SELECT created_at, id FROM wf_runs" in normalized
    assert "ORDER BY created_at DESC, id DESC" in normalized
    assert args == ["acc_1", "wf_1", "wfr_anchor", 10]
