"""``create_run(version=N)`` — re-run a historical workflow version + the
run→version composite-FK binding (#1321, Phase 2), end to end against Postgres.

A run launched with a pinned ``version`` snapshots that specific historical
version's script + declared surface (clamped against the CURRENT launcher),
binds ``source_version`` to it, and still execs its own inline ``script`` copy.
The default (``version=None``) resolves to the workflow's current version. An
archived workflow refuses ANY version. The bound ``source_version`` surfaces on
the read model.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock

import asyncpg
import pytest

from aios.db.pool import create_pool
from aios.db.queries import workflows as wf_queries
from aios.errors import ConflictError, NotFoundError
from aios.harness import runtime
from aios.models.agents import ToolSpec
from aios.workflows import run_tools, service

pytestmark = pytest.mark.integration


@pytest.fixture
async def wf_pool(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[asyncpg.Pool[Any]]:
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    prev = runtime.pool
    runtime.pool = pool
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ('acc_sv', NULL, TRUE, 'sv-root')"
            )
            await conn.execute(
                "INSERT INTO environments (id, name, config, account_id) "
                "VALUES ('env_sv', 'sv-env', '{}'::jsonb, 'acc_sv')"
            )
        run_tools._INFLIGHT.clear()
        with mock.patch("aios.workflows.service.defer_run_wake", new=AsyncMock()):
            yield pool
    finally:
        run_tools._INFLIGHT.clear()
        runtime.pool = prev
        await pool.close()


async def _v3_workflow(pool: asyncpg.Pool[Any]) -> str:
    """Create a workflow and edit it twice → a v1/v2/v3 history with distinct
    scripts (and a bash tool added only at v2)."""
    async with pool.acquire() as conn:
        wf = await wf_queries.insert_workflow(
            conn,
            account_id="acc_sv",
            name="evolving",
            script="async def main(input):\n    return 'v1'\n",
        )
        await wf_queries.update_workflow(
            conn,
            wf.id,
            account_id="acc_sv",
            expected_version=1,
            script="async def main(input):\n    return 'v2'\n",
            tools=[ToolSpec(type="bash")],
        )
        await wf_queries.update_workflow(
            conn,
            wf.id,
            account_id="acc_sv",
            expected_version=2,
            script="async def main(input):\n    return 'v3'\n",
            tools=[],
        )
    return wf.id


async def test_pinned_version_snapshots_that_version(wf_pool: asyncpg.Pool[Any]) -> None:
    wf_id = await _v3_workflow(wf_pool)

    run = await service.create_run(
        wf_pool,
        account_id="acc_sv",
        workflow_id=wf_id,
        environment_id="env_sv",
        version=2,
    )
    # The run execs v2's script + carries v2's surface (the bash tool added at v2),
    # and binds source_version to 2 — even though the head is now v3.
    assert run.source_version == 2
    assert run.script == "async def main(input):\n    return 'v2'\n"
    assert run.script_sha == hashlib.sha256(run.script.encode("utf-8")).hexdigest()
    assert [t.type for t in run.tools] == ["bash"]


async def test_default_resolves_to_current_version(wf_pool: asyncpg.Pool[Any]) -> None:
    wf_id = await _v3_workflow(wf_pool)

    run = await service.create_run(
        wf_pool,
        account_id="acc_sv",
        workflow_id=wf_id,
        environment_id="env_sv",
    )
    assert run.source_version == 3
    assert run.script == "async def main(input):\n    return 'v3'\n"


async def test_missing_version_404s(wf_pool: asyncpg.Pool[Any]) -> None:
    wf_id = await _v3_workflow(wf_pool)
    with pytest.raises(NotFoundError):
        await service.create_run(
            wf_pool,
            account_id="acc_sv",
            workflow_id=wf_id,
            environment_id="env_sv",
            version=99,
        )


async def test_archived_workflow_refuses_any_version(wf_pool: asyncpg.Pool[Any]) -> None:
    wf_id = await _v3_workflow(wf_pool)
    async with wf_pool.acquire() as conn:
        await wf_queries.archive_workflow(conn, wf_id, account_id="acc_sv")

    # The archived gate is driven by the LIVE row, so even a historical version
    # (whose snapshot has no archived_at) is refused.
    with pytest.raises(ConflictError, match="archived"):
        await service.create_run(
            wf_pool,
            account_id="acc_sv",
            workflow_id=wf_id,
            environment_id="env_sv",
            version=1,
        )


async def test_source_version_resolves_via_fk_and_read_model(
    wf_pool: asyncpg.Pool[Any],
) -> None:
    wf_id = await _v3_workflow(wf_pool)
    run = await service.create_run(
        wf_pool,
        account_id="acc_sv",
        workflow_id=wf_id,
        environment_id="env_sv",
        version=1,
    )

    # The FK resolved (the insert would have raised otherwise); confirm the bound
    # pointer is readable back on the public read model.
    async with wf_pool.acquire() as conn:
        fetched = await wf_queries.get_wf_run(conn, run.id, account_id="acc_sv")
        # The composite FK pointer resolves to the real version row.
        joined = await conn.fetchrow(
            "SELECT wv.version FROM wf_runs r "
            "JOIN workflow_versions wv "
            "  ON wv.workflow_id = r.workflow_id "
            " AND wv.version = r.source_version "
            " AND wv.account_id = r.account_id "
            "WHERE r.id = $1",
            run.id,
        )
    assert fetched.source_version == 1
    assert joined is not None and joined["version"] == 1
