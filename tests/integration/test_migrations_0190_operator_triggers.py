"""Migration 0190 against a live Postgres (#2473): operator-owned triggers.

Covers the backfill of existing (session) rows, the ``triggers_owner_kind_shape``
CHECK in both directions, the operator name index, the nullable
``trigger_runs.owner_session_id``, the account FK, and a clean downgrade.
"""

from __future__ import annotations

import asyncio
from typing import Any

import asyncpg
import pytest

from tests.conftest import needs_docker
from tests.helpers.alembic import run_alembic

_CHAIN_SQL = """
INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
VALUES ('acc_mig', NULL, TRUE, 'mig-test');
INSERT INTO environments (id, name, account_id) VALUES ('env_mig', 'mig-env', 'acc_mig');
INSERT INTO agents (id, name, model, account_id)
VALUES ('agn_mig', 'mig-agent', 'fake/test', 'acc_mig');
INSERT INTO sessions (id, agent_id, environment_id, workspace_volume_path, account_id)
VALUES ('ses_mig', 'agn_mig', 'env_mig', '/tmp/ws-mig', 'acc_mig');
"""

# What an application image from before 0190 inserts: no owner_kind column.
_OLD_SHAPE_ROW_SQL = """
INSERT INTO triggers
    (id, owner_session_id, account_id, name, source, source_spec, action, enabled, next_fire)
VALUES
    ('trig_old', 'ses_mig', 'acc_mig', 'old', 'cron', '{"schedule": "*/5 * * * *"}'::jsonb,
     '{"kind": "wake_owner", "content": "go"}'::jsonb, TRUE, now());
"""

_WORKFLOW_ACTION = (
    '{"kind": "workflow", "workflow_id": "wf_x", "workflow_version": null, "version": null, '
    '"input_template": null, "vault_ids": [], "max_outstanding_runs": null, '
    '"budget_usd": 1.5}'
)


def _operator_row(
    trigger_id: str,
    name: str,
    *,
    owner: str = "NULL",
    source: str = "'cron'",
    source_spec: str = """'{"schedule": "*/5 * * * *"}'::jsonb""",
    action: str = _WORKFLOW_ACTION,
    environment: str = "'env_mig'",
    ingest_token_hash: str = "NULL",
    next_fire: str = "now()",
) -> str:
    return f"""
        INSERT INTO triggers
            (id, owner_kind, owner_session_id, account_id, name, source, source_spec,
             action, enabled, next_fire, environment_id, ingest_token_hash)
        VALUES
            ('{trigger_id}', 'operator', {owner}, 'acc_mig', '{name}', {source}, {source_spec},
             '{action}'::jsonb, TRUE, {next_fire}, {environment}, {ingest_token_hash})
    """


async def _run(db_url: str, sql: str) -> list[Any]:
    conn = await asyncpg.connect(db_url)
    try:
        if sql.lstrip().upper().startswith("SELECT"):
            return list(await conn.fetch(sql))
        await conn.execute(sql)
        return []
    finally:
        await conn.close()


def _sql(db_url: str, sql: str) -> list[Any]:
    return asyncio.run(_run(db_url, sql))


def _rejects(db_url: str, sql: str, error: type[Exception]) -> None:
    with pytest.raises(error):
        _sql(db_url, sql)


@needs_docker
@pytest.mark.integration
def test_existing_rows_are_session_triggers_and_operator_rows_are_shaped(
    migration_db_url: str,
) -> None:
    db_url = migration_db_url
    assert run_alembic(["upgrade", "0189"], db_url).returncode == 0
    _sql(db_url, _CHAIN_SQL)
    _sql(db_url, _OLD_SHAPE_ROW_SQL)
    up = run_alembic(["upgrade", "0190"], db_url)
    assert up.returncode == 0, up.stderr

    [row] = _sql(db_url, "SELECT owner_kind FROM triggers WHERE id = 'trig_old'")
    assert row["owner_kind"] == "session"

    # A pre-0190 image keeps inserting session triggers.
    _sql(db_url, _OLD_SHAPE_ROW_SQL.replace("trig_old", "trig_old2").replace("'old'", "'old2'"))
    # A well-formed operator trigger.
    _sql(db_url, _operator_row("trig_op", "weekly"))
    # A session trigger and an operator trigger may share a name.
    _sql(db_url, _operator_row("trig_op_old", "old"))

    check = asyncpg.CheckViolationError
    _rejects(db_url, _operator_row("t1", "a", owner="'ses_mig'"), check)
    _rejects(
        db_url,
        """INSERT INTO triggers
            (id, owner_kind, owner_session_id, account_id, name, source, source_spec,
             action, enabled, next_fire)
        VALUES ('t2', 'session', NULL, 'acc_mig', 'b', 'cron',
                '{"schedule": "*/5 * * * *"}'::jsonb,
                '{"kind": "wake_owner", "content": "go"}'::jsonb, TRUE, now())""",
        check,
    )
    _rejects(
        db_url,
        _operator_row(
            "t3",
            "c",
            source="'external_event'",
            source_spec="'{}'::jsonb",
            ingest_token_hash="'abc'",
            next_fire="NULL",
        ),
        check,
    )
    _rejects(
        db_url,
        _operator_row("t4", "d", action=_WORKFLOW_ACTION.replace('"budget_usd": 1.5', '"x": 1')),
        check,
    )
    _rejects(db_url, _operator_row("t5", "e", environment="NULL"), check)
    _rejects(db_url, _operator_row("t6", "weekly"), asyncpg.UniqueViolationError)

    # An operator trigger's fire is recorded without an owner session.
    _sql(
        db_url,
        """INSERT INTO trigger_runs
            (id, trigger_id, account_id, owner_session_id, trigger_name, trigger_context, status)
        VALUES ('trun_op', 'trig_op', 'acc_mig', NULL, 'weekly', 'cron', 'ok')""",
    )
    [fk] = _sql(
        db_url,
        "SELECT confdeltype::text AS confdeltype, convalidated FROM pg_constraint "
        "WHERE conname = 'triggers_account_id_fkey'",
    )
    assert fk["confdeltype"] == "c" and fk["convalidated"]

    down = run_alembic(["downgrade", "0189"], db_url)
    assert down.returncode == 0, down.stderr
    remaining = _sql(db_url, "SELECT id FROM triggers ORDER BY id")
    assert [r["id"] for r in remaining] == ["trig_old", "trig_old2"]
    assert _sql(db_url, "SELECT id FROM trigger_runs") == []
