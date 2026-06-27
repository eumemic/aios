"""Migration 0124 adds the ``tools_vocab_epoch`` stamp + per-surface partial index.

The per-row epoch stamp is the raw-restore belt (#1576, epic #1572): a row records
the latest backfill its blob satisfied, so a chain-bypassing ``pg_restore`` of an old
DB self-describes its staleness even when it bypasses ``aios migrate``. This exercises
the migration against a real Postgres:

* the column exists on ALL SEVEN surface tables, ``smallint NOT NULL DEFAULT 0``;
* a row seeded BEFORE the migration (an "old DB" blob) ends up at epoch 0 — i.e. it
  self-describes as stale (``epoch < latest-backfill-rev``), the raw-restore belt;
* a direct ``INSERT`` that omits the column still lands at 0 (default RETAINED → fail
  safe: stale, never silently current);
* the partial stale index exists per surface and the fast boot-scan predicate
  (``MIN(epoch) >= horizon`` / index-empty) behaves.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any

import asyncpg
import pytest

from tests.conftest import _docker_available, needs_docker
from tests.integration.test_migrations import _alembic_url, _run_alembic

# The seven surface tables and their backfill horizon (must match migration 0124).
_SURFACE_TABLES = (
    "agents",
    "agent_versions",
    "workflows",
    "workflow_versions",
    "wf_runs",
    "sessions",
    "connectors",
)
_EPOCH_HORIZON = 122

# Seeded BEFORE 0124 so the row is a pristine "old DB" blob with no epoch column.
_SEED_SQL = """
INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
VALUES ('acc_root', NULL, TRUE, 'root');
INSERT INTO workflows (id, account_id, name, version, script, tools)
VALUES ('wf_old', 'acc_root', 'old', 1, 'S', '[{"type":"bash"}]'::jsonb);
"""


@pytest.fixture
def postgres() -> Iterator[object]:
    if not _docker_available():
        pytest.skip("Docker not available")
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("postgres:16-alpine") as pg:
        yield pg


async def _fetchval(db_url: str, sql: str, *args: Any) -> Any:
    conn = await asyncpg.connect(db_url)
    try:
        return await conn.fetchval(sql, *args)
    finally:
        await conn.close()


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
def test_epoch_column_and_index_added_to_all_seven_surfaces(postgres: object) -> None:
    db_url = _alembic_url(postgres)

    # Seed an "old DB" row BEFORE 0124 so it has no epoch stamp.
    up = _run_alembic(["upgrade", "0123"], db_url)
    assert up.returncode == 0, f"upgrade to 0123 failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(_execute(db_url, _SEED_SQL))

    up = _run_alembic(["upgrade", "0124"], db_url)
    assert up.returncode == 0, f"upgrade to 0124 failed:\n{up.stderr}\n{up.stdout}"

    for table in _SURFACE_TABLES:
        # Column exists, smallint, NOT NULL, server default 0. Select the three
        # attributes as separate columns (NOT a ``(a, b, c)`` row constructor):
        # asyncpg has no decoder for an anonymous composite, so a row-tuple
        # SELECT raises ``no decoder for composite type element`` at decode time.
        col_row = asyncio.run(
            _fetchrow(
                db_url,
                """
                SELECT data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_name = $1 AND column_name = 'tools_vocab_epoch'
                """,
                table,
            )
        )
        assert col_row is not None, f"{table}.tools_vocab_epoch missing"
        coltype = col_row["data_type"]
        notnull = col_row["is_nullable"]
        default = col_row["column_default"]
        assert coltype == "smallint", f"{table}.tools_vocab_epoch is {coltype}"
        assert notnull == "NO", f"{table}.tools_vocab_epoch is nullable"
        assert default is not None and "0" in default, f"{table} default={default!r}"

        # Partial stale index exists.
        idx = asyncio.run(
            _fetchval(
                db_url,
                "SELECT indexdef FROM pg_indexes WHERE tablename = $1 AND indexname = $2",
                table,
                f"ix_{table}_tools_vocab_epoch_stale",
            )
        )
        assert idx is not None, f"{table} missing partial stale index"
        assert "tools_vocab_epoch" in idx
        assert f"< {_EPOCH_HORIZON}" in idx, f"{table} index predicate: {idx}"

    # The old-DB row self-describes as stale: epoch 0 < horizon.
    old_epoch = asyncio.run(
        _fetchval(db_url, "SELECT tools_vocab_epoch FROM workflows WHERE id = 'wf_old'")
    )
    assert old_epoch == 0
    assert old_epoch < _EPOCH_HORIZON  # the raw-restore belt fires


@needs_docker
@pytest.mark.integration
def test_default_retained_so_omitting_column_yields_stale_row(postgres: object) -> None:
    # A direct INSERT that does NOT name tools_vocab_epoch (a raw restore, a write
    # path that forgets it) must land at 0 — stale, never silently current.
    db_url = _alembic_url(postgres)
    up = _run_alembic(["upgrade", "0124"], db_url)
    assert up.returncode == 0, f"upgrade to 0124 failed:\n{up.stderr}\n{up.stdout}"

    asyncio.run(
        _execute(
            db_url,
            """
            INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
            VALUES ('acc_root', NULL, TRUE, 'root');
            INSERT INTO workflows (id, account_id, name, version, script, tools)
            VALUES ('wf_raw', 'acc_root', 'raw', 1, 'S', '[{"type":"bash"}]'::jsonb);
            """,
        )
    )
    epoch = asyncio.run(
        _fetchval(db_url, "SELECT tools_vocab_epoch FROM workflows WHERE id = 'wf_raw'")
    )
    assert epoch == 0  # fail safe: a column-omitting write is stale


@needs_docker
@pytest.mark.integration
def test_partial_index_powers_fast_boot_scan(postgres: object) -> None:
    # The boot residue scan fast-paths a table to "clean" when MIN(epoch) >= horizon
    # (the partial stale index is empty); a stale row keeps the table in the scan.
    db_url = _alembic_url(postgres)
    up = _run_alembic(["upgrade", "0124"], db_url)
    assert up.returncode == 0, f"upgrade to 0124 failed:\n{up.stderr}\n{up.stdout}"

    asyncio.run(
        _execute(
            db_url,
            """
            INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
            VALUES ('acc_root', NULL, TRUE, 'root');
            INSERT INTO workflows (id, account_id, name, version, script, tools, tools_vocab_epoch)
            VALUES ('wf_current', 'acc_root', 'cur', 1, 'S', '[]'::jsonb, 122);
            """,
        )
    )
    # Only-current table → no stale residue → boot fast-paths clean.
    stale_count = asyncio.run(
        _fetchval(
            db_url,
            "SELECT count(*) FROM workflows WHERE tools_vocab_epoch < $1",
            _EPOCH_HORIZON,
        )
    )
    assert stale_count == 0
    min_epoch = asyncio.run(_fetchval(db_url, "SELECT MIN(tools_vocab_epoch) FROM workflows"))
    assert min_epoch >= _EPOCH_HORIZON

    # Add a stale row → it appears in the partial index residue.
    asyncio.run(
        _execute(
            db_url,
            "INSERT INTO workflows (id, account_id, name, version, script, tools, tools_vocab_epoch) "
            "VALUES ('wf_stale', 'acc_root', 'stale', 1, 'S', '[]'::jsonb, 0);",
        )
    )
    stale_count = asyncio.run(
        _fetchval(
            db_url,
            "SELECT count(*) FROM workflows WHERE tools_vocab_epoch < $1",
            _EPOCH_HORIZON,
        )
    )
    assert stale_count == 1
