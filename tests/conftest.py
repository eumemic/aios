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


def _docker_available() -> bool:
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
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


@pytest.fixture(scope="module")
def migrated_db_url(db_url: str) -> str:
    """Run alembic upgrade head against the testcontainer DB and return its URL."""
    env = {
        **os.environ,
        "AIOS_DB_URL": db_url,
    }
    result = subprocess.run(
        ["uv", "run", "alembic", "upgrade", "head"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"alembic upgrade failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
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
