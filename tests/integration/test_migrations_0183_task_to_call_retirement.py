"""Migration 0183 rewrites ``stop_task``/``list_tasks`` on every surface, including
the insert-only ``workflow_versions`` table, then runs the residue guard.

``workflow_versions`` carries the 0112 ``workflow_versions_no_update`` trigger,
which raises on every ``UPDATE``. A backfill that rewrites it without suspending
the guard aborts the whole migration on any database holding a legacy task verb
in a workflow version — so the DB can never reach 0183. This seeds such a row at
0182 and proves 0183 converges, the row is canonicalised, and the immutability
guard is back in force afterwards.

0183 is ONE revision (backfill, then contract guard) in one transaction, so a
failure anywhere in it must leave nothing applied: the DB stays at 0182, the
legacy rows are untouched, the immutability trigger is still enabled, and a
re-run after the cause is removed converges.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, cast

import asyncpg
import pytest

from tests.conftest import needs_docker
from tests.helpers.alembic import run_alembic

pytestmark = pytest.mark.integration

_SEED_SQL = """
INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
VALUES ('acc_root', NULL, TRUE, 'root');
INSERT INTO workflows (id, account_id, name, version, script, tools)
VALUES ('wf_tasks', 'acc_root', 'tasks', 1, 'S',
        '[{"type":"bash"},{"type":"stop_task"},{"type":"list_tasks"}]'::jsonb);
INSERT INTO workflow_versions (workflow_id, account_id, version, name, script, tools)
VALUES ('wf_tasks', 'acc_root', 1, 'tasks', 'S',
        '[{"type":"bash"},{"type":"stop_task"},{"type":"list_tasks"}]'::jsonb);
"""


async def _execute(db_url: str, sql: str) -> None:
    conn = await asyncpg.connect(db_url)
    try:
        await conn.execute(sql)
    finally:
        await conn.close()


async def _version_tools(db_url: str) -> list[dict[str, Any]]:
    conn = await asyncpg.connect(db_url)
    try:
        raw = await conn.fetchval(
            "SELECT tools FROM workflow_versions WHERE workflow_id = 'wf_tasks' AND version = 1"
        )
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        return cast("list[dict[str, Any]]", parsed)
    finally:
        await conn.close()


async def _alembic_version(db_url: str) -> str:
    conn = await asyncpg.connect(db_url)
    try:
        return cast("str", await conn.fetchval("SELECT version_num FROM alembic_version"))
    finally:
        await conn.close()


async def _trigger_state(db_url: str) -> str:
    conn = await asyncpg.connect(db_url)
    try:
        return cast(
            "str",
            await conn.fetchval(
                "SELECT tgenabled::text FROM pg_trigger WHERE tgname = 'workflow_versions_no_update'"
            ),
        )
    finally:
        await conn.close()


# Injected failure AFTER the workflow_versions rewrite (wf_runs is rewritten later
# in the same upgrade()): any UPDATE on wf_runs raises. Seeding one legacy wf_runs
# row makes the backfill reach it and abort mid-migration.
_INJECT_FAILURE_SQL = """
CREATE FUNCTION _test_fail_0183() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN RAISE EXCEPTION 'injected 0183 failure'; END $$;
CREATE TRIGGER _test_fail_0183 BEFORE UPDATE ON wf_runs
    FOR EACH ROW EXECUTE FUNCTION _test_fail_0183();
"""

_SEED_WF_RUN_SQL = """
INSERT INTO wf_runs (id, account_id, script, script_sha, environment_id,
                     host_semantics_epoch, tools)
VALUES ('wfr_tasks', 'acc_root', 'S', 'sha', 'env_x', 0,
        '[{"type":"stop_task"}]'::jsonb);
"""

_REMOVE_FAILURE_SQL = """
DROP TRIGGER _test_fail_0183 ON wf_runs;
DROP FUNCTION _test_fail_0183();
"""


async def _update_is_rejected(db_url: str) -> bool:
    conn = await asyncpg.connect(db_url)
    try:
        await conn.execute(
            "UPDATE workflow_versions SET script = 'X' WHERE workflow_id = 'wf_tasks'"
        )
    except asyncpg.RaiseError:
        return True
    finally:
        await conn.close()
    return False


@needs_docker
def test_backfill_rewrites_immutable_workflow_versions(migration_db_url: str) -> None:
    db_url = migration_db_url

    up = run_alembic(["upgrade", "0182"], db_url)
    assert up.returncode == 0, f"upgrade to 0182 failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(_execute(db_url, _SEED_SQL))

    up = run_alembic(["upgrade", "0183"], db_url)
    assert up.returncode == 0, f"upgrade to 0183 failed:\n{up.stderr}\n{up.stdout}"
    assert asyncio.run(_alembic_version(db_url)) == "0183"

    tools = asyncio.run(_version_tools(db_url))
    assert [t["type"] for t in tools] == ["bash", "cancel_call", "list_calls"]

    # The insert-only guard is re-enabled once the backfill is done.
    assert asyncio.run(_update_is_rejected(db_url))


@needs_docker
def test_failed_upgrade_leaves_nothing_applied_and_rerun_converges(
    migration_db_url: str,
) -> None:
    db_url = migration_db_url

    up = run_alembic(["upgrade", "0182"], db_url)
    assert up.returncode == 0, f"upgrade to 0182 failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(_execute(db_url, _SEED_SQL))
    asyncio.run(_execute(db_url, "SET session_replication_role = replica;" + _SEED_WF_RUN_SQL))
    asyncio.run(_execute(db_url, _INJECT_FAILURE_SQL))

    failed = run_alembic(["upgrade", "0183"], db_url)
    assert failed.returncode == 1
    assert "injected 0183 failure" in failed.stderr

    # Transactional: no rewrite, no version bump, and the DISABLE TRIGGER rolled back.
    assert asyncio.run(_alembic_version(db_url)) == "0182"
    tools = asyncio.run(_version_tools(db_url))
    assert [t["type"] for t in tools] == ["bash", "stop_task", "list_tasks"]
    assert asyncio.run(_trigger_state(db_url)) == "O"

    # Remove the cause and re-run: the same revision converges from the same state.
    asyncio.run(_execute(db_url, _REMOVE_FAILURE_SQL))
    up = run_alembic(["upgrade", "0183"], db_url)
    assert up.returncode == 0, f"re-run of 0183 failed:\n{up.stderr}\n{up.stdout}"
    assert asyncio.run(_alembic_version(db_url)) == "0183"
    tools = asyncio.run(_version_tools(db_url))
    assert [t["type"] for t in tools] == ["bash", "cancel_call", "list_calls"]
    assert asyncio.run(_trigger_state(db_url)) == "O"
