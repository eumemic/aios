"""B1.9 — worker wiring: the wake_workflow task + the run re-enqueue sweep."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock

import asyncpg
import pytest

from aios.db.pool import create_pool
from aios.db.queries import workflows as wf_queries
from aios.workflows.sweep import wake_runs_needing_step

pytestmark = pytest.mark.integration


@pytest.fixture
async def sweep_pool(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[asyncpg.Pool[Any]]:
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ('acc_sw', NULL, TRUE, 'sweep-root')"
            )
        yield pool
    finally:
        await pool.close()


def test_wake_workflow_task_registered_on_workflows_queue() -> None:
    import aios.harness.tasks  # noqa: F401  — importing registers the @app.task
    from aios.harness.procrastinate_app import app

    assert app.tasks["harness.wake_workflow"].queue == "workflows"


async def test_sweep_wakes_only_nonterminal_runs(sweep_pool: asyncpg.Pool[Any]) -> None:
    pool = sweep_pool
    runs: dict[str, str] = {}
    async with pool.acquire() as conn:
        wf = await wf_queries.insert_workflow(
            conn, account_id="acc_sw", name="w", script="async def main(input):\n    return 1"
        )
        for status in ("pending", "suspended", "completed", "errored"):
            run = await wf_queries.insert_wf_run(
                conn, account_id="acc_sw", workflow_id=wf.id, script="x", script_sha="x"
            )
            if status != "pending":
                await wf_queries.set_run_status(conn, run.id, status, account_id="acc_sw")
            runs[status] = run.id

    with mock.patch("aios.workflows.sweep.defer_run_wake", new=AsyncMock()) as deferred:
        swept = await wake_runs_needing_step(pool)

    assert swept == 2  # only pending + suspended
    woken = {call.args[0] for call in deferred.call_args_list}
    assert woken == {runs["pending"], runs["suspended"]}
