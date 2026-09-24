"""Migration 0182 against a live Postgres (#2446 d): ``wf_runs.trigger_id`` +
the partial active-runs index, the upgrade leaving pre-existing rows NULL
(= not trigger-launched, so never counted by any per-trigger cap), and a clean
downgrade.
"""

from __future__ import annotations

import asyncio

import asyncpg
import pytest

from tests.conftest import needs_docker
from tests.helpers.alembic import run_alembic

_SEED_SQL = """
INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
VALUES ('acc_root', NULL, TRUE, 'root');
INSERT INTO environments (id, name, config, account_id)
VALUES ('env_root', 'env', '{}'::jsonb, 'acc_root');
INSERT INTO wf_runs
    (id, workflow_id, account_id, environment_id, script, script_sha,
     host_semantics_epoch, status)
VALUES ('run_legacy', NULL, 'acc_root', 'env_root', 'x', 'sha', 0, 'running');
"""


async def _fetch(db_url: str, sql: str) -> list[asyncpg.Record]:
    conn = await asyncpg.connect(db_url)
    try:
        return list(await conn.fetch(sql))
    finally:
        await conn.close()


async def _execute(db_url: str, sql: str) -> None:
    conn = await asyncpg.connect(db_url)
    try:
        await conn.execute(sql)
    finally:
        await conn.close()


def _column(db_url: str) -> list[asyncpg.Record]:
    return asyncio.run(
        _fetch(
            db_url,
            "SELECT is_nullable, column_default FROM information_schema.columns "
            "WHERE table_name = 'wf_runs' AND column_name = 'trigger_id'",
        )
    )


def _index(db_url: str) -> list[asyncpg.Record]:
    return asyncio.run(
        _fetch(
            db_url,
            "SELECT i.indisvalid, pg_get_indexdef(i.indexrelid) AS def "
            "FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
            "WHERE c.relname = 'wf_runs_trigger_active_idx'",
        )
    )


@needs_docker
@pytest.mark.integration
def test_upgrade_adds_nullable_column_and_valid_partial_index(migration_db_url: str) -> None:
    db_url = migration_db_url
    up = run_alembic(["upgrade", "0181"], db_url)
    assert up.returncode == 0, up.stderr
    asyncio.run(_execute(db_url, _SEED_SQL))

    up = run_alembic(["upgrade", "0182"], db_url)
    assert up.returncode == 0, up.stderr

    (col,) = _column(db_url)
    assert (col["is_nullable"], col["column_default"]) == ("YES", None)
    (idx,) = _index(db_url)
    assert idx["indisvalid"] is True
    assert "(trigger_id)" in idx["def"]
    assert "suspended" in idx["def"]
    legacy = asyncio.run(_fetch(db_url, "SELECT trigger_id FROM wf_runs WHERE id = 'run_legacy'"))
    assert legacy[0]["trigger_id"] is None


@needs_docker
@pytest.mark.integration
def test_downgrade_drops_index_and_column(migration_db_url: str) -> None:
    db_url = migration_db_url
    assert run_alembic(["upgrade", "0182"], db_url).returncode == 0
    down = run_alembic(["downgrade", "0181"], db_url)
    assert down.returncode == 0, down.stderr
    assert _column(db_url) == []
    assert _index(db_url) == []
    # And re-upgrade is clean (the migration is re-runnable).
    assert run_alembic(["upgrade", "0182"], db_url).returncode == 0
    assert len(_index(db_url)) == 1
