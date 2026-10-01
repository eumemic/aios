"""Migration 0184 rewrites ``stop_task``/``list_tasks`` on every surface, including
the insert-only ``workflow_versions`` table.

``workflow_versions`` carries the 0112 ``workflow_versions_no_update`` trigger,
which raises on every ``UPDATE``. A backfill that rewrites it without suspending
the guard aborts the whole migration on any database holding a legacy task verb
in a workflow version — so the DB can never reach 0184/0185. This seeds such a
row at 0183 and proves 0184 → 0185 converge, the row is canonicalised, and the
immutability guard is back in force afterwards.
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

    up = run_alembic(["upgrade", "0183"], db_url)
    assert up.returncode == 0, f"upgrade to 0183 failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(_execute(db_url, _SEED_SQL))

    up = run_alembic(["upgrade", "0185"], db_url)
    assert up.returncode == 0, f"upgrade to 0185 failed:\n{up.stderr}\n{up.stdout}"

    tools = asyncio.run(_version_tools(db_url))
    assert [t["type"] for t in tools] == ["bash", "cancel_call", "list_calls"]

    # The insert-only guard is re-enabled once the backfill is done.
    assert asyncio.run(_update_is_rejected(db_url))
