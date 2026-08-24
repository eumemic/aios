"""Integration tests for the 0055/0056 stop-hook migration repair.

Production aios DBs are stamped ``alembic_version = 0055`` with a physical
``sessions.stop_hook`` column (added by #603, code-reverted by #613/#615).
Migration ``0055`` was re-added verbatim so the stamp resolves, and ``0056``
drops the orphaned column. These tests run the real alembic chain against
a fresh database for each case — each stamps/mutates ``alembic_version``, so
shared schema state would leak between tests.
"""

from __future__ import annotations

import asyncio
from typing import cast

import asyncpg
import pytest

from tests.conftest import needs_docker
from tests.helpers.alembic import run_alembic


async def _sessions_columns(db_url: str) -> set[str]:
    """Return the set of column names on the ``sessions`` table."""
    conn = await asyncpg.connect(db_url)
    try:
        rows = await conn.fetch(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name='sessions'"
        )
        return {row["column_name"] for row in rows}
    finally:
        await conn.close()


async def _version_num(db_url: str) -> str:
    conn = await asyncpg.connect(db_url)
    try:
        return cast(str, await conn.fetchval("SELECT version_num FROM alembic_version"))
    finally:
        await conn.close()


@needs_docker
@pytest.mark.integration
def test_full_chain_from_scratch_has_no_stop_hook(migration_db_url: str) -> None:
    """A fresh upgrade-to-head lands past 0056 with no ``stop_hook`` column."""
    db_url = migration_db_url

    result = run_alembic(["upgrade", "head"], db_url)
    assert result.returncode == 0, f"alembic upgrade failed:\n{result.stderr}\n{result.stdout}"

    columns = asyncio.run(_sessions_columns(db_url))
    assert "stop_hook" not in columns
    # ``stop_hook`` is dropped by 0056; later migrations don't re-add it.
    # Pin to "drop occurred" rather than a specific head so future migrations
    # don't churn this test.
    assert asyncio.run(_version_num(db_url)) >= "0056"


@needs_docker
@pytest.mark.integration
def test_forward_from_stamped_0055_drops_stop_hook(migration_db_url: str) -> None:
    """Reproduce prod (stamped 0055 with the column), then upgrade clears it."""
    db_url = migration_db_url

    result = run_alembic(["upgrade", "0055"], db_url)
    assert result.returncode == 0, f"alembic upgrade failed:\n{result.stderr}\n{result.stdout}"
    assert "stop_hook" in asyncio.run(_sessions_columns(db_url))
    assert asyncio.run(_version_num(db_url)) == "0055"

    result = run_alembic(["upgrade", "head"], db_url)
    assert result.returncode == 0, f"alembic upgrade failed:\n{result.stderr}\n{result.stdout}"
    # Past 0056 — stop_hook is dropped by that revision and stays gone.
    assert asyncio.run(_version_num(db_url)) >= "0056"
    assert "stop_hook" not in asyncio.run(_sessions_columns(db_url))


@needs_docker
@pytest.mark.integration
def test_downgrade_head_to_0054_runs_clean(migration_db_url: str) -> None:
    """Downgrading head -> 0055 -> 0054 restores and then drops ``stop_hook``."""
    db_url = migration_db_url

    result = run_alembic(["upgrade", "head"], db_url)
    assert result.returncode == 0, f"alembic upgrade failed:\n{result.stderr}\n{result.stdout}"

    result = run_alembic(["downgrade", "0055"], db_url)
    assert result.returncode == 0, f"alembic downgrade failed:\n{result.stderr}\n{result.stdout}"
    assert asyncio.run(_version_num(db_url)) == "0055"
    assert "stop_hook" in asyncio.run(_sessions_columns(db_url))

    result = run_alembic(["downgrade", "0054"], db_url)
    assert result.returncode == 0, f"alembic downgrade failed:\n{result.stderr}\n{result.stdout}"
    assert asyncio.run(_version_num(db_url)) == "0054"
    assert "stop_hook" not in asyncio.run(_sessions_columns(db_url))
