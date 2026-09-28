"""Migration 0185 against a live Postgres (#2468): ``wf_runs.visibility``.

Covers the backfill (model-dispatch runs and every descendant become ``session``),
the insert trigger that stamps later rows, and a clean downgrade.
"""

from __future__ import annotations

import asyncio

import asyncpg
import pytest

from tests.conftest import needs_docker
from tests.helpers.alembic import run_alembic

_RUN_COLUMNS = (
    "(id, account_id, environment_id, script, script_sha, host_semantics_epoch, status, "
    "parent_run_id, caller)"
)
_DISPATCH = '\'{"kind": "session", "id": "ses_gone", "purpose": "model_dispatch"}\'::jsonb'

_SEED_SQL = f"""
INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
VALUES ('acc_v', NULL, TRUE, 'root');
INSERT INTO environments (id, name, config, account_id)
VALUES ('env_v', 'env', '{{}}'::jsonb, 'acc_v');
INSERT INTO wf_runs {_RUN_COLUMNS}
VALUES
    ('run_plain', 'acc_v', 'env_v', 'x', 'sha', 0, 'completed', NULL, NULL),
    ('run_plain_child', 'acc_v', 'env_v', 'x', 'sha', 0, 'completed', 'run_plain', NULL),
    ('run_dispatch', 'acc_v', 'env_v', 'x', 'sha', 0, 'completed', NULL, {_DISPATCH}),
    ('run_dispatch_child', 'acc_v', 'env_v', 'x', 'sha', 0, 'completed', 'run_dispatch', NULL),
    ('run_dispatch_grandchild', 'acc_v', 'env_v', 'x', 'sha', 0, 'completed',
     'run_dispatch_child', NULL);
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


def _visibility(db_url: str) -> dict[str, str]:
    rows = asyncio.run(_fetch(db_url, "SELECT id, visibility FROM wf_runs"))
    return {r["id"]: r["visibility"] for r in rows}


@needs_docker
@pytest.mark.integration
def test_backfill_then_trigger_mark_model_dispatch_subtrees_session(
    migration_db_url: str,
) -> None:
    db_url = migration_db_url
    assert run_alembic(["upgrade", "0184"], db_url).returncode == 0
    asyncio.run(_execute(db_url, _SEED_SQL))
    up = run_alembic(["upgrade", "0185"], db_url)
    assert up.returncode == 0, up.stderr

    assert _visibility(db_url) == {
        "run_plain": "account",
        "run_plain_child": "account",
        "run_dispatch": "session",
        "run_dispatch_child": "session",
        "run_dispatch_grandchild": "session",
    }
    (col,) = asyncio.run(
        _fetch(
            db_url,
            "SELECT is_nullable, column_default FROM information_schema.columns "
            "WHERE table_name = 'wf_runs' AND column_name = 'visibility'",
        )
    )
    assert (col["is_nullable"], col["column_default"]) == ("NO", None)

    # Rows inserted without the column, as an older application image does.
    asyncio.run(
        _execute(
            db_url,
            f"""
            INSERT INTO wf_runs {_RUN_COLUMNS}
            VALUES
                ('new_plain', 'acc_v', 'env_v', 'x', 'sha', 0, 'running', 'run_plain', NULL),
                ('new_dispatch', 'acc_v', 'env_v', 'x', 'sha', 0, 'running', 'run_plain',
                 {_DISPATCH}),
                ('new_dispatch_child', 'acc_v', 'env_v', 'x', 'sha', 0, 'running',
                 'run_dispatch', NULL);
            """,
        )
    )
    visibility = _visibility(db_url)
    assert visibility["new_plain"] == "account"
    assert visibility["new_dispatch"] == "session"
    assert visibility["new_dispatch_child"] == "session"


@needs_docker
@pytest.mark.integration
def test_downgrade_drops_trigger_function_and_column(migration_db_url: str) -> None:
    db_url = migration_db_url
    assert run_alembic(["upgrade", "0185"], db_url).returncode == 0
    down = run_alembic(["downgrade", "0184"], db_url)
    assert down.returncode == 0, down.stderr

    leftovers = asyncio.run(
        _fetch(
            db_url,
            "SELECT 'column' AS kind FROM information_schema.columns "
            "WHERE table_name = 'wf_runs' AND column_name = 'visibility' "
            "UNION ALL SELECT 'trigger' FROM pg_trigger "
            "WHERE tgname = 'wf_runs_stamp_visibility_trg' "
            "UNION ALL SELECT 'function' FROM pg_proc "
            "WHERE proname = '_aios_stamp_wf_run_visibility'",
        )
    )
    assert leftovers == []
    assert run_alembic(["upgrade", "0185"], db_url).returncode == 0
