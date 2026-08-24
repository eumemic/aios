"""Integration tests for migration 0060's index swap.

Migration 0060 relaxes the active-row partial unique on ``connections``
from globally exclusive to per-account:

* upgrade: drops ``connections_active_external_account_uniq``, creates
  ``connections_active_account_external_uniq`` on
  ``(account_id, connector, external_account_id) WHERE archived_at IS NULL``.
* downgrade: drops the per-account index, restores the global one.

These tests run the real alembic chain against a fresh database so
each scenario gets an isolated ``alembic_version`` and ``connections``
catalog state.
"""

from __future__ import annotations

import asyncio

import asyncpg
import pytest

from tests.conftest import needs_docker
from tests.helpers.alembic import run_alembic


async def _connections_indexes(db_url: str) -> set[str]:
    """Return the set of index names on ``public.connections``."""
    conn = await asyncpg.connect(db_url)
    try:
        rows = await conn.fetch(
            "SELECT indexname FROM pg_indexes WHERE schemaname='public' AND tablename='connections'"
        )
        return {row["indexname"] for row in rows}
    finally:
        await conn.close()


async def _version_num(db_url: str) -> str:
    conn = await asyncpg.connect(db_url)
    try:
        result: str = await conn.fetchval("SELECT version_num FROM alembic_version")
        return result
    finally:
        await conn.close()


@needs_docker
@pytest.mark.integration
def test_upgrade_to_0060_swaps_index_to_per_account(migration_db_url: str) -> None:
    """After upgrade to 0060: per-account index present, global index gone."""
    db_url = migration_db_url

    result = run_alembic(["upgrade", "0060"], db_url)
    assert result.returncode == 0, f"alembic upgrade failed:\n{result.stderr}\n{result.stdout}"

    indexes = asyncio.run(_connections_indexes(db_url))
    assert "connections_active_account_external_uniq" in indexes
    assert "connections_active_external_account_uniq" not in indexes
    assert asyncio.run(_version_num(db_url)) == "0060"


@needs_docker
@pytest.mark.integration
def test_downgrade_to_0059_restores_global_index(migration_db_url: str) -> None:
    """Downgrade 0060 -> 0059 restores the pre-migration global index."""
    db_url = migration_db_url

    result = run_alembic(["upgrade", "0060"], db_url)
    assert result.returncode == 0, f"alembic upgrade failed:\n{result.stderr}\n{result.stdout}"

    result = run_alembic(["downgrade", "0059"], db_url)
    assert result.returncode == 0, f"alembic downgrade failed:\n{result.stderr}\n{result.stdout}"

    indexes = asyncio.run(_connections_indexes(db_url))
    assert "connections_active_external_account_uniq" in indexes
    assert "connections_active_account_external_uniq" not in indexes
    assert asyncio.run(_version_num(db_url)) == "0059"


@needs_docker
@pytest.mark.integration
def test_up_down_up_is_idempotent(migration_db_url: str) -> None:
    """``upgrade -> downgrade -> upgrade`` lands on the same per-account index.

    Idempotency here means "the migration is reversible and re-applicable" —
    the index swap on the second upgrade converges on the same name + key
    columns as the first run. ``DROP INDEX IF EXISTS`` on both ends keeps
    re-application safe even when the previous step leaves the catalog in
    an unexpected state.
    """
    db_url = migration_db_url

    result = run_alembic(["upgrade", "0060"], db_url)
    assert result.returncode == 0, f"alembic upgrade failed:\n{result.stderr}\n{result.stdout}"
    result = run_alembic(["downgrade", "0059"], db_url)
    assert result.returncode == 0, f"alembic downgrade failed:\n{result.stderr}\n{result.stdout}"
    result = run_alembic(["upgrade", "0060"], db_url)
    assert result.returncode == 0, f"alembic upgrade failed:\n{result.stderr}\n{result.stdout}"

    indexes = asyncio.run(_connections_indexes(db_url))
    assert "connections_active_account_external_uniq" in indexes
    assert "connections_active_external_account_uniq" not in indexes
    assert asyncio.run(_version_num(db_url)) == "0060"
