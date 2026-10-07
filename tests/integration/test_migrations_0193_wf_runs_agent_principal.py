"""Migration 0193 against a live Postgres (#2476): principal ``agent``.

Covers the backfill (an ``as_agent`` run and its operator sub-runs become ``agent``),
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
    "parent_run_id, as_agent_id, as_agent_version, caller)"
)
_SESSION_CALLER = '\'{"kind": "session", "id": "ses_gone"}\'::jsonb'

_SEED_SQL = f"""
INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
VALUES ('acc_p', NULL, TRUE, 'root');
INSERT INTO environments (id, name, config, account_id)
VALUES ('env_p', 'env', '{{}}'::jsonb, 'acc_p');
INSERT INTO wf_runs {_RUN_COLUMNS}
VALUES
    ('run_root', 'acc_p', 'env_p', 'x', 'sha', 0, 'completed', NULL, NULL, NULL, NULL),
    ('run_item', 'acc_p', 'env_p', 'x', 'sha', 0, 'completed', 'run_root', NULL, NULL, NULL),
    ('run_arm', 'acc_p', 'env_p', 'x', 'sha', 0, 'completed', 'run_item', 'agt_c', 1, NULL),
    ('run_arm_child', 'acc_p', 'env_p', 'x', 'sha', 0, 'completed', 'run_arm', NULL, NULL,
     NULL),
    ('run_arm_session_launched', 'acc_p', 'env_p', 'x', 'sha', 0, 'completed', 'run_arm',
     NULL, NULL, {_SESSION_CALLER});
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


def _principal(db_url: str) -> dict[str, str]:
    rows = asyncio.run(_fetch(db_url, "SELECT id, principal FROM wf_runs"))
    return {r["id"]: r["principal"] for r in rows}


@needs_docker
@pytest.mark.integration
def test_backfill_then_trigger_mark_as_agent_subtrees_agent(migration_db_url: str) -> None:
    db_url = migration_db_url
    assert run_alembic(["upgrade", "0192"], db_url).returncode == 0
    asyncio.run(_execute(db_url, _SEED_SQL))
    assert _principal(db_url)["run_arm"] == "operator"  # 0184 inherited the parent's
    up = run_alembic(["upgrade", "0193"], db_url)
    assert up.returncode == 0, up.stderr

    assert _principal(db_url) == {
        "run_root": "operator",
        "run_item": "operator",
        "run_arm": "agent",
        "run_arm_child": "agent",
        "run_arm_session_launched": "session",
    }

    # Rows inserted without the column, as an application image does.
    asyncio.run(
        _execute(
            db_url,
            f"""
            INSERT INTO wf_runs {_RUN_COLUMNS}
            VALUES
                ('new_arm', 'acc_p', 'env_p', 'x', 'sha', 0, 'running', 'run_item', 'agt_c',
                 1, NULL),
                ('new_arm_child', 'acc_p', 'env_p', 'x', 'sha', 0, 'running', 'new_arm', NULL,
                 NULL, NULL),
                ('new_item', 'acc_p', 'env_p', 'x', 'sha', 0, 'running', 'run_root', NULL,
                 NULL, NULL);
            """,
        )
    )
    principal = _principal(db_url)
    assert principal["new_arm"] == "agent"
    assert principal["new_arm_child"] == "agent"
    assert principal["new_item"] == "operator"


@needs_docker
@pytest.mark.integration
def test_downgrade_restores_operator_principal(migration_db_url: str) -> None:
    db_url = migration_db_url
    assert run_alembic(["upgrade", "0192"], db_url).returncode == 0
    asyncio.run(_execute(db_url, _SEED_SQL))
    assert run_alembic(["upgrade", "0193"], db_url).returncode == 0
    down = run_alembic(["downgrade", "0192"], db_url)
    assert down.returncode == 0, down.stderr

    assert set(_principal(db_url).values()) == {"operator", "session"}
    asyncio.run(
        _execute(
            db_url,
            f"""
            INSERT INTO wf_runs {_RUN_COLUMNS}
            VALUES ('down_arm', 'acc_p', 'env_p', 'x', 'sha', 0, 'running', 'run_item',
                    'agt_c', 1, NULL);
            """,
        )
    )
    assert _principal(db_url)["down_arm"] == "operator"
    assert run_alembic(["upgrade", "0193"], db_url).returncode == 0
