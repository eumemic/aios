"""Integration tests for migration 0176 (ssh_servers arm + ssh_key kind).

Pins the round-trip against a real Postgres:
  - a clean upgrade→downgrade→upgrade cycle (no ssh_key rows) is reversible,
    and the six ``ssh_servers`` columns appear/disappear;
  - the downgrade is fail-loud when an ``ssh_key`` row exists — the binary
    ``vault_credentials_shape_check`` is restored BEFORE the columns are dropped,
    so an ``ssh_key`` row (which the binary shape rejects) aborts the migration
    rather than being silently destroyed.

Each test mutates ``alembic_version``, so each test gets its own database.
"""

from __future__ import annotations

import asyncio

import asyncpg
import pytest

from tests.conftest import needs_docker
from tests.helpers.alembic import run_alembic

_SURFACE_TABLES = ("agents", "agent_versions", "workflows", "workflow_versions", "wf_runs")

_ACCOUNT_SQL = """
INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
VALUES ('acc_root', NULL, TRUE, 'root')
ON CONFLICT DO NOTHING
"""

_VAULT_SQL = """
INSERT INTO vaults (id, account_id, display_name, metadata)
VALUES ('vlt_test', 'acc_root', 'test', '{}'::jsonb)
"""

# A well-formed ssh_key row under the three-way shape CHECK: no target_url, no
# allowed_hosts, secret_name populated.
_SSH_KEY_CRED_SQL = r"""
INSERT INTO vault_credentials (
    id, vault_id, account_id, display_name, target_url, secret_name,
    allowed_hosts, auth_type, ciphertext, nonce, metadata
)
VALUES (
    'vcr_ssh', 'vlt_test', 'acc_root', NULL, NULL, 'PROD_KEY',
    NULL, 'ssh_key', '\x00'::bytea, '\x00'::bytea, '{}'::jsonb
)
"""


async def _has_ssh_servers_column(db_url: str, table: str) -> bool:
    conn = await asyncpg.connect(db_url)
    try:
        return bool(
            await conn.fetchval(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_schema='public' AND table_name=$1 AND column_name='ssh_servers'",
                table,
            )
        )
    finally:
        await conn.close()


async def _execute(db_url: str, sql: str) -> None:
    conn = await asyncpg.connect(db_url)
    try:
        await conn.execute(sql)
    finally:
        await conn.close()


async def _row_exists(db_url: str, cred_id: str) -> bool:
    conn = await asyncpg.connect(db_url)
    try:
        return bool(await conn.fetchval("SELECT 1 FROM vault_credentials WHERE id = $1", cred_id))
    finally:
        await conn.close()


@needs_docker
@pytest.mark.integration
def test_clean_round_trip(migration_db_url: str) -> None:
    """Upgrade adds ssh_servers to all six surfaces; downgrade removes them; re-upgrade restores."""
    db_url = migration_db_url

    up = run_alembic(["upgrade", "head"], db_url)
    assert up.returncode == 0, f"upgrade failed:\n{up.stderr}\n{up.stdout}"
    for table in (*_SURFACE_TABLES, "sessions"):
        assert asyncio.run(_has_ssh_servers_column(db_url, table)), f"{table} missing ssh_servers"

    down = run_alembic(["downgrade", "0175"], db_url)
    assert down.returncode == 0, f"downgrade failed:\n{down.stderr}\n{down.stdout}"
    for table in (*_SURFACE_TABLES, "sessions"):
        assert not asyncio.run(_has_ssh_servers_column(db_url, table)), f"{table} kept ssh_servers"

    reup = run_alembic(["upgrade", "head"], db_url)
    assert reup.returncode == 0, f"re-upgrade failed:\n{reup.stderr}\n{reup.stdout}"


@needs_docker
@pytest.mark.integration
def test_ssh_key_row_accepted_and_blocks_downgrade(migration_db_url: str) -> None:
    """An ssh_key row passes the three-way shape CHECK; the downgrade then fails
    loud (the binary shape CHECK it restores rejects the row), preserving it."""
    db_url = migration_db_url

    up = run_alembic(["upgrade", "head"], db_url)
    assert up.returncode == 0, f"upgrade failed:\n{up.stderr}\n{up.stdout}"

    asyncio.run(_execute(db_url, _ACCOUNT_SQL))
    asyncio.run(_execute(db_url, _VAULT_SQL))
    asyncio.run(_execute(db_url, _SSH_KEY_CRED_SQL))

    down = run_alembic(["downgrade", "0175"], db_url)
    assert down.returncode != 0, f"downgrade should have failed loud:\n{down.stdout}"

    assert asyncio.run(_row_exists(db_url, "vcr_ssh"))
    assert asyncio.run(_has_ssh_servers_column(db_url, "agents"))
