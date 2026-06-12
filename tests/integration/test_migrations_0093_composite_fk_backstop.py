"""Migration 0093 adds composite tenant FKs for secret-bearing chains."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import asyncpg
import pytest

from tests.conftest import _docker_available, needs_docker
from tests.integration.test_migrations import _alembic_url, _run_alembic

_CHAIN_SQL = """
INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
VALUES ('acc_root', NULL, TRUE, 'root'),
       ('acc_a', 'acc_root', FALSE, 'tenant-a'),
       ('acc_b', 'acc_root', FALSE, 'tenant-b');
INSERT INTO environments (id, name, account_id)
VALUES ('env_a', 'env-a', 'acc_a');
INSERT INTO agents (id, name, model, account_id)
VALUES ('agent_a', 'agent-a', 'test/model', 'acc_a');
INSERT INTO sessions (id, agent_id, environment_id, workspace_volume_path, account_id)
VALUES ('sess_a', 'agent_a', 'env_a', '/tmp/ws-a', 'acc_a');
INSERT INTO vaults (id, display_name, account_id)
VALUES ('vault_b', 'foreign vault', 'acc_b');
"""

_LEGACY_CROSS_TENANT_SESSION_VAULT_SQL = """
INSERT INTO session_vaults (session_id, vault_id, rank, account_id)
VALUES ('sess_a', 'vault_b', 0, 'acc_a');
"""


@pytest.fixture
def postgres() -> Iterator[object]:
    """Fresh function-scoped Postgres; each test mutates alembic_version."""
    if not _docker_available():
        pytest.skip("Docker not available")
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("postgres:16-alpine") as pg:
        yield pg


async def _execute(db_url: str, sql: str) -> None:
    conn = await asyncpg.connect(db_url)
    try:
        await conn.execute(sql)
    finally:
        await conn.close()


@needs_docker
@pytest.mark.integration
def test_clean_database_upgrades_to_composite_fk_backstop(postgres: object) -> None:
    db_url = _alembic_url(postgres)

    up = _run_alembic(["upgrade", "head"], db_url)

    assert up.returncode == 0, f"upgrade to head failed:\n{up.stderr}\n{up.stdout}"


@needs_docker
@pytest.mark.integration
def test_legacy_cross_tenant_session_vault_fails_validation(postgres: object) -> None:
    db_url = _alembic_url(postgres)

    up = _run_alembic(["upgrade", "0092"], db_url)
    assert up.returncode == 0, f"upgrade to 0092 failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(_execute(db_url, _CHAIN_SQL + _LEGACY_CROSS_TENANT_SESSION_VAULT_SQL))

    up = _run_alembic(["upgrade", "head"], db_url)

    assert up.returncode != 0, f"upgrade should have failed loud:\n{up.stdout}"
    output = up.stderr + up.stdout
    assert "session_vaults_vault_account_id_fkey" in output
