"""Migration 0184 against a live Postgres (#2467): ``wf_runs.principal``.

Covers the backfill, which must classify a run as ``session`` from any evidence that
survives session deletion (not just ``launcher_session_id``), the insert trigger
that stamps every later row, and a clean downgrade.
"""

from __future__ import annotations

import asyncio

import asyncpg
import pytest

from tests.conftest import needs_docker
from tests.helpers.alembic import run_alembic

_RUN_COLUMNS = (
    "(id, account_id, environment_id, script, script_sha, host_semantics_epoch, status, "
    "launcher_session_id, parent_run_id, caller, trigger_id)"
)

_SEED_SQL = f"""
INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
VALUES ('acc_p', NULL, TRUE, 'root');
INSERT INTO environments (id, name, config, account_id)
VALUES ('env_p', 'env', '{{}}'::jsonb, 'acc_p');
INSERT INTO agents (id, name, model, account_id)
VALUES ('agent_p', 'agent-p', 'test/model', 'acc_p');
INSERT INTO sessions (
    id, agent_id, environment_id, workspace_volume_path, account_id,
    archive_when_idle, last_event_seq, created_by_type, created_by_ref
)
VALUES ('ses_live', 'agent_p', 'env_p', '/tmp/ses-live', 'acc_p',
        FALSE, 0, 'api_actor', 'key_p');
INSERT INTO wf_runs {_RUN_COLUMNS}
VALUES
    -- operator root and its sub-run
    ('run_op', 'acc_p', 'env_p', 'x', 'sha', 0, 'running', NULL, NULL, NULL, NULL),
    ('run_op_child', 'acc_p', 'env_p', 'x', 'sha', 0, 'running', NULL, 'run_op',
     '{{"kind": "run", "id": "run_op"}}'::jsonb, NULL),
    -- launched by a live session
    ('run_live', 'acc_p', 'env_p', 'x', 'sha', 0, 'running', 'ses_live', NULL, NULL, NULL),
    -- launched by a session since deleted: only the caller survives
    ('run_orphan', 'acc_p', 'env_p', 'x', 'sha', 0, 'running', NULL, NULL,
     '{{"kind": "session", "id": "ses_gone"}}'::jsonb, NULL),
    ('run_orphan_child', 'acc_p', 'env_p', 'x', 'sha', 0, 'running', NULL, 'run_orphan',
     '{{"kind": "run", "id": "run_orphan"}}'::jsonb, NULL),
    ('run_orphan_grandchild', 'acc_p', 'env_p', 'x', 'sha', 0, 'running', NULL,
     'run_orphan_child', '{{"kind": "run", "id": "run_orphan_child"}}'::jsonb, NULL),
    -- fired by a trigger, whose owner session is gone
    ('run_trigger', 'acc_p', 'env_p', 'x', 'sha', 0, 'running', NULL, NULL, NULL, 'trg_gone'),
    -- fired by a trigger before 0182 stamped trigger_id: only the audit row survives
    ('run_trigger_pre_0182', 'acc_p', 'env_p', 'x', 'sha', 0, 'running',
     NULL, NULL, NULL, NULL);
INSERT INTO trigger_runs (
    id, trigger_id, account_id, owner_session_id, trigger_name, trigger_context,
    status, result_id
)
VALUES ('trun_old', 'trg_old', 'acc_p', 'ses_gone', 'nightly', 'cron', 'ok',
        'run_trigger_pre_0182');
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


def _principals(db_url: str) -> dict[str, str]:
    rows = asyncio.run(_fetch(db_url, "SELECT id, principal FROM wf_runs"))
    return {r["id"]: r["principal"] for r in rows}


def _upgrade_seeded(db_url: str) -> None:
    up = run_alembic(["upgrade", "0183"], db_url)
    assert up.returncode == 0, up.stderr
    asyncio.run(_execute(db_url, _SEED_SQL))
    up = run_alembic(["upgrade", "0184"], db_url)
    assert up.returncode == 0, up.stderr


@needs_docker
@pytest.mark.integration
def test_backfill_classifies_from_evidence_that_survives_session_deletion(
    migration_db_url: str,
) -> None:
    db_url = migration_db_url
    _upgrade_seeded(db_url)

    assert _principals(db_url) == {
        "run_op": "operator",
        "run_op_child": "operator",
        "run_live": "session",
        "run_orphan": "session",
        "run_orphan_child": "session",
        "run_orphan_grandchild": "session",
        "run_trigger": "session",
        "run_trigger_pre_0182": "session",
    }
    # NOT NULL, and the backfill's default is gone, so the trigger is the only writer.
    (col,) = asyncio.run(
        _fetch(
            db_url,
            "SELECT is_nullable, column_default FROM information_schema.columns "
            "WHERE table_name = 'wf_runs' AND column_name = 'principal'",
        )
    )
    assert (col["is_nullable"], col["column_default"]) == ("NO", None)


@needs_docker
@pytest.mark.integration
def test_trigger_stamps_every_insert_and_deletion_does_not_change_it(
    migration_db_url: str,
) -> None:
    db_url = migration_db_url
    _upgrade_seeded(db_url)

    # Rows inserted without the column (as an older application image does), plus one
    # that passes a principal contradicting its launcher: the trigger decides both.
    # Deleting the session afterwards nulls the launchers but moves no principal.
    asyncio.run(
        _execute(
            db_url,
            f"""
            INSERT INTO wf_runs {_RUN_COLUMNS}
            VALUES
                ('new_op', 'acc_p', 'env_p', 'x', 'sha', 0, 'running',
                 NULL, NULL, NULL, NULL),
                ('new_op_child', 'acc_p', 'env_p', 'x', 'sha', 0, 'running',
                 NULL, 'run_op', NULL, NULL),
                ('new_session', 'acc_p', 'env_p', 'x', 'sha', 0, 'running',
                 'ses_live', NULL, NULL, NULL),
                ('new_session_caller', 'acc_p', 'env_p', 'x', 'sha', 0, 'running',
                 NULL, NULL, '{{"kind": "session", "id": "ses_live"}}'::jsonb, NULL),
                ('new_orphan_child', 'acc_p', 'env_p', 'x', 'sha', 0, 'running',
                 NULL, 'run_orphan', NULL, NULL);
            INSERT INTO wf_runs
                (id, account_id, environment_id, script, script_sha,
                 host_semantics_epoch, status, launcher_session_id, principal)
            VALUES ('new_claims_operator', 'acc_p', 'env_p', 'x', 'sha', 0, 'running',
                    'ses_live', 'operator');
            DELETE FROM sessions WHERE id = 'ses_live';
            """,
        )
    )

    rows = asyncio.run(
        _fetch(
            db_url,
            "SELECT id, principal, launcher_session_id FROM wf_runs "
            "WHERE id LIKE 'new_%' OR id = 'run_live'",
        )
    )
    assert {r["id"]: r["principal"] for r in rows} == {
        "new_op": "operator",
        "new_op_child": "operator",
        "new_session": "session",
        "new_session_caller": "session",
        "new_orphan_child": "session",
        "new_claims_operator": "session",
        "run_live": "session",
    }
    assert {r["launcher_session_id"] for r in rows} == {None}


@needs_docker
@pytest.mark.integration
def test_downgrade_drops_trigger_function_and_column(migration_db_url: str) -> None:
    db_url = migration_db_url
    assert run_alembic(["upgrade", "0184"], db_url).returncode == 0
    down = run_alembic(["downgrade", "0183"], db_url)
    assert down.returncode == 0, down.stderr

    leftovers = asyncio.run(
        _fetch(
            db_url,
            "SELECT 'column' AS kind FROM information_schema.columns "
            "WHERE table_name = 'wf_runs' AND column_name = 'principal' "
            "UNION ALL SELECT 'trigger' FROM pg_trigger "
            "WHERE tgname = 'wf_runs_stamp_principal_trg' "
            "UNION ALL SELECT 'function' FROM pg_proc "
            "WHERE proname = '_aios_stamp_wf_run_principal'",
        )
    )
    assert leftovers == []
    assert run_alembic(["upgrade", "0184"], db_url).returncode == 0
