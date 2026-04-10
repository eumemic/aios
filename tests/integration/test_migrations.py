"""Verify the initial migration applies cleanly against a real Postgres."""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path

import asyncpg
import pytest

from tests.conftest import _docker_available, needs_docker

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def postgres() -> Iterator[object]:
    if not _docker_available():
        pytest.skip("Docker not available")
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("postgres:16-alpine") as pg:
        yield pg


def _alembic_url(pg: object) -> str:
    """Return the connection URL alembic env.py expects."""
    host = pg.get_container_host_ip()
    port = pg.get_exposed_port(5432)
    user = pg.username
    password = pg.password
    db = pg.dbname
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def _run_alembic(args: list[str], db_url: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", "alembic", *args],
        cwd=PROJECT_ROOT,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "AIOS_DB_URL": db_url,
            "HOME": str(Path.home()),
        },
        capture_output=True,
        text=True,
        check=False,
    )


@needs_docker
@pytest.mark.integration
def test_migration_creates_all_tables(postgres: object) -> None:
    db_url = _alembic_url(postgres)
    result = _run_alembic(["upgrade", "head"], db_url)
    assert result.returncode == 0, f"alembic upgrade failed:\n{result.stderr}\n{result.stdout}"

    # Now connect with asyncpg and verify all 5 tables + key indexes exist.
    import asyncio

    async def check() -> None:
        conn = await asyncpg.connect(db_url)
        try:
            tables = await conn.fetch(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename;"
            )
            names = {row["tablename"] for row in tables}
            assert {
                "credentials",
                "environments",
                "agents",
                "sessions",
                "events",
                "alembic_version",
            } <= names, f"missing tables: {names}"

            # Spot-check a few critical indexes
            indexes = await conn.fetch(
                "SELECT indexname FROM pg_indexes WHERE schemaname = 'public';"
            )
            index_names = {row["indexname"] for row in indexes}
            for required in (
                "credentials_name_uniq",
                "agents_name_uniq",
                "events_session_seq_idx",
                "events_session_message_seq_idx",
                "sessions_lease_idx",
            ):
                assert required in index_names, f"missing index {required}"
        finally:
            await conn.close()

    asyncio.run(check())


@needs_docker
@pytest.mark.integration
def test_migration_downgrade_drops_tables(postgres: object) -> None:
    db_url = _alembic_url(postgres)
    # Make sure we're at head
    _run_alembic(["upgrade", "head"], db_url)
    # Then go back to base
    result = _run_alembic(["downgrade", "base"], db_url)
    assert result.returncode == 0, f"alembic downgrade failed:\n{result.stderr}\n{result.stdout}"

    import asyncio

    async def check() -> None:
        conn = await asyncpg.connect(db_url)
        try:
            tables = await conn.fetch(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public';"
            )
            names = {row["tablename"] for row in tables}
            for table in ("credentials", "environments", "agents", "sessions", "events"):
                assert table not in names, f"{table} should be dropped"
        finally:
            await conn.close()

    asyncio.run(check())
