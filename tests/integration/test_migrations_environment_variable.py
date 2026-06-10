"""Integration tests for migration 0081 (environment_variable credentials).

Pins two behaviors of the round-trip against a real Postgres:
  - a clean upgrade→downgrade→upgrade cycle (no env-var rows) is reversible;
  - the downgrade is fail-loud when an ``environment_variable`` row exists —
    the narrowed ``auth_type`` CHECK is re-added BEFORE the new columns are
    dropped, so the migration aborts (preserving ``secret_name``/
    ``allowed_hosts``) rather than silently destroying the data.

Each test mutates ``alembic_version``, so the container is function-scoped.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import asyncpg
import pytest

from tests.conftest import _docker_available, needs_docker
from tests.integration.test_migrations import _alembic_url, _run_alembic

_ACCOUNT_SQL = """
INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
VALUES ('acc_root', NULL, TRUE, 'root')
ON CONFLICT DO NOTHING
"""

_VAULT_SQL = """
INSERT INTO vaults (id, account_id, display_name, metadata)
VALUES ('vlt_test', 'acc_root', 'test', '{}'::jsonb)
"""

_ENV_VAR_CRED_SQL = r"""
INSERT INTO vault_credentials (
    id, vault_id, account_id, display_name, target_url, secret_name,
    allowed_hosts, auth_type, ciphertext, nonce, metadata
)
VALUES (
    'vcr_test', 'vlt_test', 'acc_root', NULL, NULL, 'GITHUB_TOKEN',
    ARRAY['api.github.com/repos/eumemic'], 'environment_variable',
    '\x00'::bytea, '\x00'::bytea, '{}'::jsonb
)
"""


@pytest.fixture
def postgres() -> Iterator[object]:
    """Fresh function-scoped Postgres — each test mutates ``alembic_version``."""
    if not _docker_available():
        pytest.skip("Docker not available")
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("postgres:16-alpine") as pg:
        yield pg


async def _vault_credentials_columns(db_url: str) -> set[str]:
    conn = await asyncpg.connect(db_url)
    try:
        rows = await conn.fetch(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name='vault_credentials'"
        )
        return {row["column_name"] for row in rows}
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
def test_clean_round_trip(postgres: object) -> None:
    """Upgrade adds the columns; downgrade (no env-var rows) removes them; re-upgrade restores."""
    db_url = _alembic_url(postgres)

    up = _run_alembic(["upgrade", "head"], db_url)
    assert up.returncode == 0, f"upgrade failed:\n{up.stderr}\n{up.stdout}"
    cols = asyncio.run(_vault_credentials_columns(db_url))
    assert {"secret_name", "allowed_hosts"} <= cols

    down = _run_alembic(["downgrade", "0080"], db_url)
    assert down.returncode == 0, f"downgrade failed:\n{down.stderr}\n{down.stdout}"
    cols = asyncio.run(_vault_credentials_columns(db_url))
    assert "secret_name" not in cols
    assert "allowed_hosts" not in cols

    reup = _run_alembic(["upgrade", "head"], db_url)
    assert reup.returncode == 0, f"re-upgrade failed:\n{reup.stderr}\n{reup.stdout}"


@needs_docker
@pytest.mark.integration
def test_env_var_row_accepted_and_blocks_downgrade(postgres: object) -> None:
    """An env-var row passes the widened CHECKs; the downgrade then fails loud,
    preserving the row rather than dropping its columns."""
    db_url = _alembic_url(postgres)

    up = _run_alembic(["upgrade", "head"], db_url)
    assert up.returncode == 0, f"upgrade failed:\n{up.stderr}\n{up.stdout}"

    # The widened auth_type CHECK + shape CHECK + secret_name uniq index all
    # accept a well-formed environment_variable row.
    asyncio.run(_execute(db_url, _ACCOUNT_SQL))
    asyncio.run(_execute(db_url, _VAULT_SQL))
    asyncio.run(_execute(db_url, _ENV_VAR_CRED_SQL))

    # Downgrade must abort: the narrowed auth_type CHECK is re-added before the
    # columns are dropped, so the env-var row blocks it fail-loud.
    down = _run_alembic(["downgrade", "0080"], db_url)
    assert down.returncode != 0, f"downgrade should have failed loud:\n{down.stdout}"

    # The row (and its secret_name/allowed_hosts columns) survived intact.
    assert asyncio.run(_row_exists(db_url, "vcr_test"))
    cols = asyncio.run(_vault_credentials_columns(db_url))
    assert {"secret_name", "allowed_hosts"} <= cols
