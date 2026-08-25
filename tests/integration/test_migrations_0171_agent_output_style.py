"""Migration 0171 adds ``output_style`` to agents + agent_versions.

Covers the additive-column contract on a real Postgres: a row seeded
BEFORE the migration comes out ``output_style = 'default'`` (existing
agents keep current behavior), an INSERT that omits the column lands at
``'default'``, and both columns are text NOT NULL (no CHECK constraint --
the pydantic ``OutputStyle`` Literal is the single validation point,
0139/0111 precedent).
"""

from __future__ import annotations

import asyncio
from typing import Any

import asyncpg
import pytest

from tests.conftest import needs_docker
from tests.helpers.alembic import run_alembic

# Seeded BEFORE 0171 so the rows predate the column.
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


async def _output_style_columns(db_url: str) -> set[str]:
    conn = await asyncpg.connect(db_url)
    try:
        rows = await conn.fetch(
            """
            SELECT table_name
              FROM information_schema.columns
             WHERE table_schema = 'public'
               AND table_name IN ('agents', 'agent_versions')
               AND column_name = 'output_style'
            """
        )
        return {row["table_name"] for row in rows}
    finally:
        await conn.close()


@needs_docker
@pytest.mark.integration
def test_output_style_column_added_default_default(migration_db_url: str) -> None:
    db_url = migration_db_url

    up = run_alembic(["upgrade", "0169"], db_url)
    assert up.returncode == 0, f"upgrade to 0169 failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(_execute(db_url, _SEED_SQL))

    up = run_alembic(["upgrade", "0171"], db_url)
    assert up.returncode == 0, f"upgrade to 0171 failed:\n{up.stderr}\n{up.stdout}"

    for table in ("agents", "agent_versions"):
        meta = asyncio.run(
            _fetchrow(
                db_url,
                """
                SELECT data_type AS dt, is_nullable AS nullable, column_default AS dflt
                  FROM information_schema.columns
                 WHERE table_name = $1 AND column_name = 'output_style'
                """,
                table,
            )
        )
        assert meta is not None, f"{table}.output_style column missing"
        assert meta["dt"] == "text"
        assert meta["nullable"] == "NO"
        assert meta["dflt"] == "'default'::text"

    # Pre-existing rows come out 'default' — existing agents keep current behavior.
    row = asyncio.run(_fetchrow(db_url, "SELECT output_style FROM agents WHERE id = 'agt_old'"))
    assert row["output_style"] == "default"
    row = asyncio.run(
        _fetchrow(db_url, "SELECT output_style FROM agent_versions WHERE agent_id = 'agt_old'")
    )
    assert row["output_style"] == "default"

    # An INSERT that omits the column still lands at 'default' (default retained).
    asyncio.run(
        _execute(
            db_url,
            """
            INSERT INTO agents (id, name, model, version, account_id)
            VALUES ('agt_new', 'new', 'openrouter/test', 1, 'acc_root')
            """,
        )
    )
    row = asyncio.run(_fetchrow(db_url, "SELECT output_style FROM agents WHERE id = 'agt_new'"))
    assert row["output_style"] == "default"


@needs_docker
@pytest.mark.integration
@pytest.mark.parametrize("unsafe_table", ["agents", "agent_versions"])
def test_downgrade_refuses_each_non_default_output_style_without_dropping_columns(
    migration_db_url: str, unsafe_table: str
) -> None:
    """Each target independently blocks a lossy downgrade, before either DROP."""
    db_url = migration_db_url
    up = run_alembic(["upgrade", "0169"], db_url)
    assert up.returncode == 0, f"upgrade to 0169 failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(_execute(db_url, _SEED_SQL))
    up = run_alembic(["upgrade", "0171"], db_url)
    assert up.returncode == 0, f"upgrade to 0171 failed:\n{up.stderr}\n{up.stdout}"

    where = "id = 'agt_old'" if unsafe_table == "agents" else "agent_id = 'agt_old'"
    asyncio.run(
        _execute(db_url, f"UPDATE {unsafe_table} SET output_style = 'concise' WHERE {where}")
    )

    down = run_alembic(["downgrade", "0169"], db_url)
    assert down.returncode != 0, f"downgrade should refuse unsafe {unsafe_table}:\n{down.stdout}"
    assert "cannot downgrade 0171" in down.stderr
    assert unsafe_table in down.stderr

    # Alembic's failed transaction must preserve both targets, even when the
    # unsafe target is the second table checked/dropped.
    assert asyncio.run(_output_style_columns(db_url)) == {"agents", "agent_versions"}
    row = asyncio.run(_fetchrow(db_url, f"SELECT output_style FROM {unsafe_table} WHERE {where}"))
    assert row["output_style"] == "concise"
