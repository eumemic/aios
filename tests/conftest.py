"""Shared pytest fixtures for aios tests.

* ``postgres_container`` — session-scoped testcontainer running Postgres 16
* ``migrated_db_url`` — runs alembic upgrade head + applies the procrastinate
  schema and aios lock-release trigger against the testcontainer
* ``_truncate_sql`` — session-scoped: the ``TRUNCATE`` statement covering
  every public-schema table, computed once after migrations
* ``_reset_db_state`` — function-scoped: TRUNCATEs all public-schema tables
  before each test so the session-scoped DB stays isolated between tests.
  Yields the asyncpg connection so downstream fixtures (``aios_env``) can
  reuse it without paying a second connect round-trip
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
import logging
import os
import secrets
import subprocess
import threading
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import structlog

_tmpl_log = logging.getLogger("aios.test.template_cache")

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
os.environ.setdefault(
    "AIOS_EGRESS_CA_KEY",
    base64.b64encode(secrets.token_bytes(32)).decode("ascii"),
)
os.environ.setdefault("AIOS_DB_URL", "postgresql://x:x@localhost:5432/x")

# Scope ``AIOS_INSTANCE_ID`` per pytest-xdist worker so that
# ``SandboxRegistry.reap_orphans`` (which lists containers by the
# ``aios.instance_id`` label) only ever sees this worker's containers.
# ``Settings.instance_id`` defaults to the literal ``"default"`` —
# without this override, two xdist workers in the same CI job would
# share an instance_id and a hypothetical future test that triggers
# the orphan-reaper path would ``docker rm -f`` the sibling worker's
# live sandbox.  Today no test exercises that path, but pre-empting
# the footgun is cheap and keeps ``-n 2`` safe for future test growth.
# ``PYTEST_XDIST_WORKER`` is set by xdist to ``gw0`` / ``gw1`` / ...
# per worker and is absent in single-process runs.
_xdist_worker = os.environ.get("PYTEST_XDIST_WORKER")
if _xdist_worker:
    os.environ.setdefault("AIOS_INSTANCE_ID", f"test_{_xdist_worker}")


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

    # Migration tests exercise every schema transition, not crash durability.
    # Disabling synchronous disk persistence removes fsync latency from the
    # 100+ Alembic runs while preserving PostgreSQL's SQL/transaction behavior.
    container = (
        PostgresContainer("postgres:16-alpine")
        .with_command("postgres -c fsync=off -c synchronous_commit=off -c full_page_writes=off")
        # Keep ephemeral database files out of overlayfs.  The suite validates
        # SQL and every migration transition, not persistence across restarts.
        .with_kwargs(tmpfs={"/var/lib/postgresql/data": "rw"})
    )
    with container as pg:
        yield pg
