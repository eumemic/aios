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
import os
import secrets
import subprocess
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import structlog

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


@pytest.fixture(scope="session")
def _truncate_sql(migrated_db_url: str) -> str:
    """Pre-built ``TRUNCATE`` statement covering every public-schema table.

    The schema is fixed once :func:`migrated_db_url` has run, so the
    ``pg_tables`` lookup belongs at session scope, not in every
    :func:`_reset_db_state` call.  Eliminating the per-test catalog scan
    shaves ~3 ms per test off the e2e suite.
    """
    import asyncio

    import asyncpg

    async def fetch() -> str:
        conn = await asyncpg.connect(migrated_db_url)
        try:
            rows = await conn.fetch(
                # ``alembic_version`` is schema-version *metadata*, not test
                # data: it is stamped once by :func:`migrated_db_url` and must
                # survive the per-test reset.  Truncating it leaves the DB
                # reading as "unmigrated" (``version_num`` empty), which the
                # boot-admission gate (tests/integration/test_boot_admission_gate_db.py)
                # correctly treats as behind every ``contract_rev`` — so the
                # clean-DB case wrongly raises ``DatabaseBehindContract``.
                # Exclude it from the truncate so reset clears rows, not schema.
                "SELECT tablename FROM pg_tables "
                "WHERE schemaname = 'public' AND tablename <> 'alembic_version'"
            )
        finally:
            await conn.close()
        if not rows:
            return ""
        quoted = ", ".join(f'"{r["tablename"]}"' for r in rows)
        return f"TRUNCATE {quoted} RESTART IDENTITY CASCADE"

    return asyncio.run(fetch())


@pytest.fixture
async def _reset_db_state(migrated_db_url: str, _truncate_sql: str) -> AsyncIterator[Any]:
    """TRUNCATE every public-schema table before each test, yielding the
    connection so downstream fixtures (``aios_env``) can reuse it instead
    of opening their own.

    Restores the cross-test isolation that module-scoped Postgres used to
    provide.  ``TRUNCATE`` is metadata-only in Postgres, so this is
    O(tables) regardless of row count.
    """
    import asyncpg

    conn = await asyncpg.connect(migrated_db_url)
    try:
        if _truncate_sql:
            await conn.execute(_truncate_sql)
        yield conn
    finally:
        await conn.close()


@pytest.fixture(autouse=True)
def _reset_sse_starlette_shutdown_state() -> Iterator[None]:
    """Neutralize sse-starlette's cross-test ``AppStatus.should_exit`` contamination.

    ``sse-starlette.sse.AppStatus`` is a module-level class with a class-attribute
    ``should_exit`` flag plus a per-thread background watcher that polls a captured
    ``uvicorn.Server`` instance.  When an in-process uvicorn fixture's teardown sets
    ``server.should_exit = True``, the watcher (if it has had time to poll —
    interval is 0.5 s) latches the global ``AppStatus.should_exit = True`` and is
    never reset.  Every later test's first SSE request then sees
    ``AppStatus.should_exit`` is True at the top of
    :func:`sse_starlette.sse._listen_for_exit_signal`, which returns immediately,
    triggering :func:`cancel_on_finish` in the SSE :class:`EventSourceResponse`'s
    task group and killing the SSE generator before its body iterator yields
    anything.  The client sees ``peer closed connection without sending complete
    message body``; the server logs ``ASGI callable returned without completing
    response.``

    The watcher firing is timing-dependent — fast hosts (developer laptops) often
    finish a test before the next 0.5 s poll, so contamination doesn't latch.  CI
    hosts (slower Ubuntu runners) routinely lose the race, which is why the three
    connector e2e tests (signal register/captcha, telegram multi-connection) flaked
    in CI but never locally.  Tearing the flag back to False before each test
    breaks the contamination chain without disabling the watcher entirely (kept
    on so production deployments still observe SIGTERM-triggered graceful drain).
    See aios#365.
    """
    from sse_starlette.sse import AppStatus

    AppStatus.should_exit = False
    yield
    AppStatus.should_exit = False


@pytest.fixture
def aios_env_minimal(
    migrated_db_url: str, _reset_db_state: Any, tmp_path: Path
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
        "AIOS_EGRESS_CA_KEY": base64.b64encode(secrets.token_bytes(32)).decode("ascii"),
        "AIOS_DB_URL": migrated_db_url,
        "AIOS_WORKSPACE_ROOT": str(tmp_path / "workspaces"),
        # Tests exercise the production default. Legacy env-key behavior is opt-in.
        "AIOS_INFERENCE_CREDENTIAL_POLICY": "account_only",
        # Issue #807: point the docker_harness-driven e2e provisions at the
        # repo's authored seccomp profile. The config default resolves to the
        # baked /app/docker path, which doesn't exist on the host running the
        # tests; this overrides it to the in-tree file so seccomp is actually
        # enforced on e2e sandboxes (e.g. the Limited-provision regression).
        "AIOS_SANDBOX_SECCOMP_PROFILE": str(
            Path(__file__).parents[1] / "docker" / "seccomp-sandbox.json"
        ),
    }
    with mock.patch.dict(os.environ, env_vars):
        from aios.config import get_settings

        get_settings.cache_clear()
        yield env_vars
        get_settings.cache_clear()


@pytest.fixture
async def aios_env(aios_env_minimal: dict[str, str], _reset_db_state: Any) -> dict[str, str]:
    """Env vars + a bootstrapped root whose key is ``AIOS_API_KEY``.

    Auth looks the bearer token up against ``account_keys``, so any
    test that hits an authenticated route needs a matching DB row.
    This fixture seeds that row using ``AIOS_API_KEY``'s sha256 hash.

    Reuses the connection yielded by :func:`_reset_db_state` to dodge
    a second ``asyncpg.connect`` round-trip per test (~27 ms).

    Async so the seed step runs on the same event loop pytest-asyncio
    drives the test on — avoids the spurious ``asyncio.run`` event-loop
    isolation that caused ASGI-callable teardown flakiness in CI.
    """
    from aios.services.accounts import hash_key

    plaintext = aios_env_minimal["AIOS_API_KEY"]
    conn = _reset_db_state

    # PR 6: insert ``acc_test_stub`` as the root account itself and bind
    # the bootstrap API key directly to it, so auth resolves the bearer
    # to the same account_id every test body uses as its scoping arg.
    # Bypasses ``bootstrap_root_account`` (which would generate a fresh
    # ULID we'd have to thread back through 44 test files).
    await conn.execute(
        """
        INSERT INTO accounts
            (id, parent_account_id, can_mint_children, display_name)
        VALUES ('acc_test_stub', NULL, TRUE, 'test-root')
        """
    )
    await conn.execute(
        """
        INSERT INTO account_keys (key_id, account_id, hash, label)
        VALUES ('akey_test', 'acc_test_stub', $1, 'test-root')
        """,
        hash_key(plaintext),
    )
    return aios_env_minimal


@pytest.fixture(autouse=True)
def _configure_structlog_for_tests() -> Iterator[None]:
    """Configure structlog before every test so caplog captures all messages.

    Two changes vs. structlog's bare default (which is what tests see when
    nothing calls aios.logging.configure_logging at process start):

    1. logger_factory=structlog.stdlib.LoggerFactory() — routes log records
       through Python's stdlib logging, which is what caplog hooks. The bare
       default uses PrintLoggerFactory, which writes to stdout and is invisible
       to caplog.

    2. cache_logger_on_first_use=False — modules with module-level
       log = structlog.get_logger() bind their config on first call. Caching
       freezes that binding, so tests reconfiguring structlog (e.g., here)
       cannot affect already-bound loggers. Disabling caching costs ~1us
       per call to rebind — negligible at test-suite scale.

    Function scope (not session) is deliberate: mid-suite code reconfigures
    structlog and would otherwise leak across tests. ``aios.api.app.create_app``
    calls ``aios.logging.configure_logging`` (which sets
    cache_logger_on_first_use=True), and the ``capture_logs`` fixtures in
    test_db_pool.py / test_worker_exit_diagnostics.py call
    ``structlog.reset_defaults()`` in teardown. A once-per-session fixture
    cannot defend against either — re-applying before each test does.
    """
    current = structlog.get_config()
    structlog.configure(
        **{
            **current,
            "logger_factory": structlog.stdlib.LoggerFactory(),
            "cache_logger_on_first_use": False,
        }
    )
    yield
    structlog.configure(**current)
