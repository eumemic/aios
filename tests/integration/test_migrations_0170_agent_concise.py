"""Migration 0170 adds ``concise`` to agents + agent_versions.

Covers the additive-column contract on a real Postgres: a row seeded
BEFORE the migration comes out ``concise = false`` (existing agents keep
current behavior), an INSERT that omits the column lands at ``false``,
and both columns are boolean NOT NULL.
"""

from __future__ import annotations

import asyncio
from typing import Any

import asyncpg
import pytest

from tests.conftest import needs_docker
from tests.helpers.alembic import run_alembic

# Seeded BEFORE 0170 so the rows predate the column.
_SEED_SQL = """
INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
VALUES ('acc_root', NULL, TRUE, 'root');
INSERT INTO agents (id, name, model, version, account_id)
VALUES ('agt_old', 'old', 'openrouter/test', 1, 'acc_root');
INSERT INTO agent_versions (agent_id, version, model, account_id)
VALUES ('agt_old', 1, 'openrouter/test', 'acc_root');
"""


async def _fetchrow(db_url: str, sql: str, *args: Any) -> Any:
    conn = await asyncpg.connect(db_url)
    try:
        return await conn.fetchrow(sql, *args)
    finally:
        await conn.close()


async def _execute(db_url: str, sql: str) -> None:
    conn = await asyncpg.connect(db_url)
    try:
        await conn.execute(sql)
    finally:
        await conn.close()


@needs_docker
@pytest.mark.integration
def test_concise_column_added_default_false(migration_db_url: str) -> None:
    db_url = migration_db_url

    up = run_alembic(["upgrade", "0169"], db_url)
    assert up.returncode == 0, f"upgrade to 0169 failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(_execute(db_url, _SEED_SQL))

    up = run_alembic(["upgrade", "0170"], db_url)
    assert up.returncode == 0, f"upgrade to 0170 failed:\n{up.stderr}\n{up.stdout}"

    for table in ("agents", "agent_versions"):
        meta = asyncio.run(
            _fetchrow(
                db_url,
                """
                SELECT data_type AS dt, is_nullable AS nullable, column_default AS dflt
                  FROM information_schema.columns
                 WHERE table_name = $1 AND column_name = 'concise'
                """,
                table,
            )
        )
        assert meta is not None, f"{table}.concise column missing"
        assert meta["dt"] == "boolean"
        assert meta["nullable"] == "NO"
        assert meta["dflt"] == "false"

    # Pre-existing rows come out false — existing agents keep current behavior.
    row = asyncio.run(_fetchrow(db_url, "SELECT concise FROM agents WHERE id = 'agt_old'"))
    assert row["concise"] is False
    row = asyncio.run(
        _fetchrow(db_url, "SELECT concise FROM agent_versions WHERE agent_id = 'agt_old'")
    )
    assert row["concise"] is False

    # An INSERT that omits the column still lands at false (default retained).
    asyncio.run(
        _execute(
            db_url,
            """
            INSERT INTO agents (id, name, model, version, account_id)
            VALUES ('agt_new', 'new', 'openrouter/test', 1, 'acc_root')
            """,
        )
    )
    row = asyncio.run(_fetchrow(db_url, "SELECT concise FROM agents WHERE id = 'agt_new'"))
    assert row["concise"] is False
