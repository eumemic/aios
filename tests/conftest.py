"""Shared pytest fixtures for aios tests.

* ``postgres_container`` — session-scoped testcontainer running Postgres 16
* ``migrated_db_url`` — runs alembic upgrade head + applies the procrastinate
  schema and aios lock-release trigger against the testcontainer
* ``_reset_db_state`` — function-scoped: TRUNCATEs all public-schema tables
  before each test so the session-scoped DB stays isolated between tests
* ``aios_env_minimal`` — env vars only, no DB seeding. For tests that
  exercise pre-bootstrap state (bootstrap endpoint tests, etc.)
* ``aios_env`` — ``aios_env_minimal`` plus a bootstrapped root account
  whose key is ``AIOS_API_KEY``. The default for tests that need an
  authenticated route to work without manual setup

Tests that need Docker are marked ``integration``; pytest -m "not integration"
runs only the unit tests, which is what most local dev iterations use.
"""

from __future__ import annotations

import base64
import os
import secrets
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Set dummy env vars at conftest IMPORT time (i.e. before pytest collection
# imports any test modules).  Some test modules import production code at
# module level — e.g. ``test_worker_heartbeat.py`` imports
# ``aios.harness.worker`` which transitively imports ``procrastinate_app``
# which calls ``get_settings()`` eagerly.  A session-scoped autouse fixture
# fires too late; collection has already crashed.
#
# ``setdefault`` lets e2e tests override with testcontainer-backed values.
os.environ.setdefault("AIOS_API_KEY", "test-key-for-unit-tests")
os.environ.setdefault(
    "AIOS_VAULT_KEY",
    base64.b64encode(secrets.token_bytes(32)).decode("ascii"),
)
os.environ.setdefault("AIOS_DB_URL", "postgresql://x:x@localhost:5432/x")


def _docker_available() -> bool:
    """Check if Docker is available, ensuring ``DOCKER_HOST`` is set.

    The Docker CLI auto-discovers Docker Desktop's socket, but the
    Python ``docker`` library and ``testcontainers`` require
    ``DOCKER_HOST`` in the environment. This function sets it
    whenever Docker is available but ``DOCKER_HOST`` is missing.
    """
    # If DOCKER_HOST is already set, just verify Docker is reachable.
    if "DOCKER_HOST" in os.environ:
        try:
            result = subprocess.run(["docker", "info"], capture_output=True, check=False, timeout=5)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    # DOCKER_HOST is not set. Try known socket paths.
    for sock in [
        Path("/var/run/docker.sock"),
        Path.home() / ".docker" / "run" / "docker.sock",
    ]:
        if sock.exists():
            os.environ["DOCKER_HOST"] = f"unix://{sock}"
            try:
                result = subprocess.run(
                    ["docker", "info"], capture_output=True, check=False, timeout=5
                )
                if result.returncode == 0:
                    return True
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
            del os.environ["DOCKER_HOST"]

    return False


needs_docker = pytest.mark.skipif(
    not _docker_available(),
    reason="Docker is not running; integration tests need it for the postgres testcontainer",
)


@pytest.fixture(scope="session")
def postgres_container() -> Iterator[Any]:
    if not _docker_available():
        pytest.skip("Docker not available")
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("postgres:16-alpine") as pg:
        yield pg


@pytest.fixture(scope="session")
def db_url(postgres_container: Any) -> str:
    host = postgres_container.get_container_host_ip()
    port = postgres_container.get_exposed_port(5432)
    user = postgres_container.username
    password = postgres_container.password
    db = postgres_container.dbname
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


@pytest.fixture(scope="session")
def migrated_db_url(db_url: str) -> str:
    """Run alembic upgrade head against the testcontainer, then apply the
    procrastinate schema and the aios lock-release trigger. Returns the URL."""
    import asyncio

    from aios.db.migrations import apply_procrastinate_schema, upgrade_to_head

    upgrade_to_head(db_url)
    asyncio.run(apply_procrastinate_schema(db_url))
    return db_url


@pytest.fixture
async def _reset_db_state(migrated_db_url: str) -> None:
    """TRUNCATE every public-schema table before each test.

    Restores the cross-test isolation that module-scoped Postgres used to
    provide.  ``TRUNCATE`` is metadata-only in Postgres, so this is O(tables)
    regardless of row count.
    """
    import asyncpg

    conn = await asyncpg.connect(migrated_db_url)
    try:
        rows = await conn.fetch("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        if rows:
            quoted = ", ".join(f'"{r["tablename"]}"' for r in rows)
            await conn.execute(f"TRUNCATE {quoted} RESTART IDENTITY CASCADE")
    finally:
        await conn.close()


@pytest.fixture
def aios_env_minimal(
    migrated_db_url: str, _reset_db_state: None, tmp_path: Path
) -> Iterator[dict[str, str]]:
    """Set the env vars the FastAPI app needs, without seeding any data.

    Use this when the test specifically needs a fresh-install state —
    e.g. the bootstrap-endpoint tests, which expect no root account
    to exist yet. Most tests want :func:`aios_env`, which layers a
    bootstrapped root on top so the auth dep accepts ``AIOS_API_KEY``
    as a bearer token without further setup.
    """
    env_vars = {
        "AIOS_API_KEY": "aios_" + secrets.token_urlsafe(32),
        "AIOS_VAULT_KEY": base64.b64encode(secrets.token_bytes(32)).decode("ascii"),
        "AIOS_DB_URL": migrated_db_url,
        "AIOS_WORKSPACE_ROOT": str(tmp_path / "workspaces"),
    }
    with mock.patch.dict(os.environ, env_vars):
        from aios.config import get_settings
        from aios.db import pool

        get_settings.cache_clear()
        pool._pool = None
        yield env_vars
        get_settings.cache_clear()
        pool._pool = None


@pytest.fixture
async def aios_env(aios_env_minimal: dict[str, str]) -> dict[str, str]:
    """Env vars + a bootstrapped root whose key is ``AIOS_API_KEY``.

    Auth looks the bearer token up against ``account_keys``, so any
    test that hits an authenticated route needs a matching DB row.
    This fixture seeds that row using ``AIOS_API_KEY``'s sha256 hash.

    Async so the seed step runs on the same event loop pytest-asyncio
    drives the test on — avoids the spurious ``asyncio.run`` event-loop
    isolation that caused ASGI-callable teardown flakiness in CI.
    """
    account_id = "acc_test_stub"  # PR 3 scaffolding
    import asyncpg

    from aios.db import queries
    from aios.services.accounts import hash_key

    plaintext = aios_env_minimal["AIOS_API_KEY"]
    db_url = aios_env_minimal["AIOS_DB_URL"]

    conn = await asyncpg.connect(db_url)
    try:
        root, _key_id = await queries.bootstrap_root_account(
            conn,
            display_name="root",
            key_hash=hash_key(plaintext),
            key_label="test-root",
            account_id=account_id,
        )
        # PR 4: many tests still pass the PR 3 stub literal ``"acc_test_stub"``
        # as the account_id arg. Now that migration 0043 enforces NOT NULL +
        # FK on every resource table, the stub must correspond to an actual
        # row. Insert it as a child of root so existing test bodies keep
        # working without sweeping rewrites.
        await conn.execute(
            """
            INSERT INTO accounts
                (id, parent_account_id, can_mint_children, display_name)
            VALUES ('acc_test_stub', $1, FALSE, 'test-stub')
            """,
            root.id,
        )
    finally:
        await conn.close()
    return aios_env_minimal
