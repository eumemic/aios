"""Shared pytest fixtures for aios tests.

* ``postgres_container`` — module-scoped testcontainer running Postgres 16
* ``migrated_db_url`` — runs alembic upgrade head against the testcontainer
* ``app_env`` — sets the env vars the FastAPI app needs (api key, vault key, db url)
* ``test_client`` — a FastAPI test client wired against a fresh app + the migrated DB

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


@pytest.fixture(autouse=True, scope="session")
def _unit_test_env() -> Iterator[None]:
    """Provide dummy env vars so ``get_settings()`` doesn't crash in unit
    tests that happen to import tool handlers or other code that reads
    settings at module level. Without this, fresh checkouts (worktrees,
    CI runners) fail because there's no ``.env`` file.

    The ``aios_env`` fixture used by e2e/integration tests overrides these
    with real (testcontainer-backed) values at function scope.
    """
    dummy = {
        "AIOS_API_KEY": "test-key-for-unit-tests",
        "AIOS_VAULT_KEY": base64.b64encode(secrets.token_bytes(32)).decode("ascii"),
        "AIOS_DB_URL": "postgresql://x:x@localhost:5432/x",
    }
    with mock.patch.dict(os.environ, dummy):
        from aios.config import get_settings

        get_settings.cache_clear()
        yield
        get_settings.cache_clear()


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


@pytest.fixture(scope="module")
def postgres_container() -> Iterator[Any]:
    if not _docker_available():
        pytest.skip("Docker not available")
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("postgres:16-alpine") as pg:
        yield pg


@pytest.fixture(scope="module")
def db_url(postgres_container: Any) -> str:
    host = postgres_container.get_container_host_ip()
    port = postgres_container.get_exposed_port(5432)
    user = postgres_container.username
    password = postgres_container.password
    db = postgres_container.dbname
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def run_alembic_upgrade(db_url: str, target: str = "head") -> None:
    """Run ``alembic upgrade <target>`` against ``db_url`` via subprocess."""
    env = {**os.environ, "AIOS_DB_URL": db_url}
    result = subprocess.run(
        ["uv", "run", "alembic", "upgrade", target],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"alembic upgrade {target} failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )


@pytest.fixture(scope="module")
def migrated_db_url(db_url: str) -> str:
    """Run alembic upgrade head against the testcontainer DB and return its URL."""
    run_alembic_upgrade(db_url, "head")
    return db_url


@pytest.fixture
def aios_env(migrated_db_url: str, tmp_path: Path) -> Iterator[dict[str, str]]:
    """Set the env vars the FastAPI app needs."""
    env_vars = {
        "AIOS_API_KEY": secrets.token_urlsafe(32),
        "AIOS_VAULT_KEY": base64.b64encode(secrets.token_bytes(32)).decode("ascii"),
        "AIOS_DB_URL": migrated_db_url,
        "AIOS_WORKSPACE_ROOT": str(tmp_path / "workspaces"),
    }
    with mock.patch.dict(os.environ, env_vars):
        # Reset the cached settings singleton
        from aios.config import get_settings
        from aios.db import pool

        get_settings.cache_clear()
        pool._pool = None
        yield env_vars
        get_settings.cache_clear()
        pool._pool = None
