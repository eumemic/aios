"""Integration test: run-spawned ``agent()`` children are listable by run_id,
even after they soft-archive on idle, with a terminal ``archived`` status and
their token usage.

Regression for #831: ``agent()`` children (``origin=background``,
``archive_when_idle=True``) are recovered only via journal-scraping because
``list_sessions`` hardcoded ``archived_at IS NULL`` and the derived status was
only ``{active, idle}``. An account running workflows must be able to enumerate
its judgment nodes and sum their token spend as a first-class query.
"""

from __future__ import annotations

from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.db.queries import workflows as wf_queries
from aios.workflows.determinism import HOST_SEMANTICS_EPOCH
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration


async def _seed_run(conn: asyncpg.Connection[Any], account_id: str) -> str:
    wf = await wf_queries.insert_workflow(
        conn,
        account_id=account_id,
        name="demo",
        script="async def main(input):\n    return 1\n",
    )
    run = await wf_queries.insert_wf_run(
        conn,
        account_id=account_id,
        workflow_id=wf.id,
        environment_id="env_lsrc",
        script=wf.script,
        host_semantics_epoch=HOST_SEMANTICS_EPOCH,
        script_sha="deadbeef",
    )
    return run.id


class TestListSessionsRunChildren:
    async def test_archived_children_listable_by_run_id_with_status_and_usage(
        self, migrated_db_url: str, _reset_db_state: None
    ) -> None:
        pool: asyncpg.Pool[Any] = await create_pool(migrated_db_url, min_size=1, max_size=4)
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO accounts (id, parent_account_id, can_mint_children, "
                    "display_name) VALUES ('acc_lsrc', NULL, TRUE, 'run-children-test')"
                )
                await conn.execute(
                    "INSERT INTO environments (id, name, config, account_id) "
                    "VALUES ('env_lsrc', 'lsrc-env', '{}'::jsonb, 'acc_lsrc')"
                )

            agent, _env, _parent = await seed_agent_env_session(
                pool, account_id="acc_lsrc", prefix="lsrc"
            )

            async with pool.acquire() as conn:
                run_id = await _seed_run(conn, "acc_lsrc")
                child = await queries.insert_child_session(
                    conn,
                    session_id="ses_lsrc_child",
                    account_id="acc_lsrc",
                    agent_id=agent.id,
                    environment_id="env_lsrc",
                    agent_version=1,
                    model="openrouter/test",
                    parent_run_id=run_id,
                    tools=[],
                    mcp_servers=[],
                    http_servers=[],
                )
                assert child is not None
                # Record token spend (the per-beat cost the v0 PoC had to scrape).
                await queries.increment_session_usage(
                    conn,
                    child.id,
                    account_id="acc_lsrc",
                    input_tokens=12_010,
                    output_tokens=5_133,
                )

            # While alive, the child lists by run_id (status active/idle).
            async with pool.acquire() as conn:
                live = await queries.list_sessions(
                    conn, account_id="acc_lsrc", parent_run_id=run_id
                )
            assert [s.id for s in live] == [child.id]

            # The child reclaims itself on idle (archive_when_idle).
            async with pool.acquire() as conn:
                archived = await queries.reclaim_session_if_idle(
                    conn, child.id, account_id="acc_lsrc"
                )
            assert archived is True

            # The archived child STILL lists by run_id, with terminal status and usage.
            async with pool.acquire() as conn:
                after = await queries.list_sessions(
                    conn, account_id="acc_lsrc", parent_run_id=run_id
                )
            assert [s.id for s in after] == [child.id]
            got = after[0]
            assert got.status == "archived"
            assert got.archived_at is not None
            assert got.usage.input_tokens == 12_010
            assert got.usage.output_tokens == 5_133

        finally:
            await pool.close()

    async def test_status_archived_filter_includes_archived_only(
        self, migrated_db_url: str, _reset_db_state: None
    ) -> None:
        pool: asyncpg.Pool[Any] = await create_pool(migrated_db_url, min_size=1, max_size=4)
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO accounts (id, parent_account_id, can_mint_children, "
                    "display_name) VALUES ('acc_lsrc2', NULL, TRUE, 'run-children-test2')"
                )
            # One live foreground session...
            _agent, _env, live = await seed_agent_env_session(
                pool, account_id="acc_lsrc2", prefix="lsrc2"
            )
            # ...and one archived session.
            _agent2, _env2, gone = await seed_agent_env_session(
                pool, account_id="acc_lsrc2", prefix="lsrc2b"
            )
            async with pool.acquire() as conn:
                await queries.archive_session(conn, gone.id, account_id="acc_lsrc2")

            # Default listing omits the archived session.
            async with pool.acquire() as conn:
                default = await queries.list_sessions(conn, account_id="acc_lsrc2")
            default_ids = {s.id for s in default}
            assert live.id in default_ids
            assert gone.id not in default_ids

            # status=archived returns the archived session, not the live one.
            async with pool.acquire() as conn:
                archived = await queries.list_sessions(
                    conn, account_id="acc_lsrc2", status="archived"
                )
            archived_ids = {s.id for s in archived}
            assert gone.id in archived_ids
            assert live.id not in archived_ids
            assert all(s.status == "archived" for s in archived)
        finally:
            await pool.close()
