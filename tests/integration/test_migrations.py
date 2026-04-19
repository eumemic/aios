"""Verify the initial migration applies cleanly against a real Postgres."""

from __future__ import annotations

import os
import shutil
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
    uv = shutil.which("uv")
    if uv is None:
        raise FileNotFoundError("uv not found on PATH")
    return subprocess.run(
        [uv, "run", "alembic", *args],
        cwd=PROJECT_ROOT,
        env={
            "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/local/bin"),
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


# Columns added by migration 0017 (focal-channel attention model). Listed
# as (table, column) pairs used by the cycle test below.
_MIGRATION_0017_COLUMNS: tuple[tuple[str, str], ...] = (
    ("sessions", "focal_channel"),
    ("events", "orig_channel"),
    ("events", "focal_channel_at_arrival"),
    ("channel_bindings", "notification_mode"),
)


async def _column_exists(conn: asyncpg.Connection, table: str, column: str) -> bool:
    row = await conn.fetchrow(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = $1 AND column_name = $2",
        table,
        column,
    )
    return row is not None


@needs_docker
@pytest.mark.integration
def test_migration_0017_focal_channel_cycle(postgres: object) -> None:
    """Exercise migration 0017's up/down/up cycle.

    Verifies that the focal-channel columns appear at head, are removed
    on ``downgrade -1`` (back to 0016), and reappear on ``upgrade head``.
    """
    db_url = _alembic_url(postgres)

    # Start at head (idempotent regardless of prior test state).
    upgraded = _run_alembic(["upgrade", "head"], db_url)
    assert upgraded.returncode == 0, f"initial upgrade failed:\n{upgraded.stderr}"

    import asyncio

    async def assert_columns(expected: bool) -> None:
        conn = await asyncpg.connect(db_url)
        try:
            for table, column in _MIGRATION_0017_COLUMNS:
                exists = await _column_exists(conn, table, column)
                if expected:
                    assert exists, f"{table}.{column} missing after upgrade"
                else:
                    assert not exists, f"{table}.{column} still present after downgrade"
        finally:
            await conn.close()

    # 1. Columns exist at head.
    asyncio.run(assert_columns(True))

    # 2. Downgrade one step → back to 0016. Columns gone.
    downgraded = _run_alembic(["downgrade", "-1"], db_url)
    assert downgraded.returncode == 0, f"downgrade -1 failed:\n{downgraded.stderr}"
    asyncio.run(assert_columns(False))

    # 3. Upgrade back to head. Columns reappear.
    re_upgraded = _run_alembic(["upgrade", "head"], db_url)
    assert re_upgraded.returncode == 0, f"re-upgrade failed:\n{re_upgraded.stderr}"
    asyncio.run(assert_columns(True))


_MIGRATION_0018_COLUMNS: tuple[tuple[str, str], ...] = (
    ("agents", "triage"),
    ("agent_versions", "triage"),
)


@needs_docker
@pytest.mark.integration
def test_migration_0018_agent_triage_cycle(postgres: object) -> None:
    """Exercise migration 0018's up/down/up cycle for the triage column.

    The triage gate is a nullable JSONB field on both the current agent
    row and each historical version, so a downgrade has to drop both.
    A broken downgrade would leave the ``agents`` table diverged from
    ``agent_versions`` — subtle and easy to miss without an explicit cycle.
    """
    db_url = _alembic_url(postgres)

    upgraded = _run_alembic(["upgrade", "head"], db_url)
    assert upgraded.returncode == 0, f"initial upgrade failed:\n{upgraded.stderr}"

    import asyncio

    async def assert_columns(expected: bool) -> None:
        conn = await asyncpg.connect(db_url)
        try:
            for table, column in _MIGRATION_0018_COLUMNS:
                exists = await _column_exists(conn, table, column)
                if expected:
                    assert exists, f"{table}.{column} missing after upgrade"
                else:
                    assert not exists, f"{table}.{column} still present after downgrade"
        finally:
            await conn.close()

    asyncio.run(assert_columns(True))

    downgraded = _run_alembic(["downgrade", "-1"], db_url)
    assert downgraded.returncode == 0, f"downgrade -1 failed:\n{downgraded.stderr}"
    asyncio.run(assert_columns(False))

    re_upgraded = _run_alembic(["upgrade", "head"], db_url)
    assert re_upgraded.returncode == 0, f"re-upgrade failed:\n{re_upgraded.stderr}"
    asyncio.run(assert_columns(True))
