"""Regression coverage for the workflow-run collection ordering contract."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from aios.db.queries import workflows
from aios.models.workflows import RunReader


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
        reader=None,
    )

    sql, *args = conn.fetch.await_args.args
    normalized = " ".join(sql.split())
    assert "(created_at, id) < (SELECT created_at, id FROM wf_runs" in normalized
    assert "ORDER BY created_at DESC, id DESC" in normalized
    assert args == ["acc_1", "wf_1", "wfr_anchor", 10]


@pytest.mark.asyncio
async def test_list_wf_runs_has_no_default_reader() -> None:
    """The unfiltered (operator) view must be asked for by name: a caller that omits
    ``reader`` is refused rather than handed every run (#2513)."""
    conn = AsyncMock()
    conn.fetch.return_value = []
    with pytest.raises(TypeError, match="reader"):
        await workflows.list_wf_runs(conn, account_id="acc_1")  # type: ignore[call-arg]
    conn.fetch.assert_not_awaited()


@pytest.mark.asyncio
async def test_list_wf_runs_reader_filters_the_page_and_the_cursor() -> None:
    conn = AsyncMock()
    conn.fetch.return_value = []

    await workflows.list_wf_runs(
        conn, account_id="acc_1", after="wfr_anchor", reader=RunReader("ses_1")
    )

    sql, *args = conn.fetch.await_args.args
    normalized = " ".join(sql.split())
    visible = "(visibility = 'account' OR launcher_session_id = $2)"
    assert normalized.count(visible) == 2  # the page and the cursor subselect
    assert args == ["acc_1", "ses_1", "wfr_anchor", 50]
