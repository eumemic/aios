"""Verify the initial migration applies cleanly against a real Postgres.

Exports
-------
``_alembic_url``  — build a connection URL from an ``IsolatedPostgres``.
``_run_alembic``  — run an Alembic command in-process.  When a
    ``MigrationTemplateCache`` has been installed via ``install_cache``,
    the first upgrade on a virgin DB is satisfied by cloning from a
    session-scoped template instead of replaying the full migration chain.
``PROJECT_ROOT``  — repo root ``Path``.
"""

from __future__ import annotations

import os
import re
import subprocess
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

import asyncpg
import psycopg
import pytest

from tests.conftest import needs_docker

if TYPE_CHECKING:
    from tests.conftest import MigrationTemplateCache

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Session-scoped template cache, installed by the ``_install_migration_cache``
# autouse fixture below.  When set, ``_run_alembic`` transparently clones from
# pre-migrated template DBs instead of replaying migrations from scratch.
_cache: MigrationTemplateCache | None = None
# Resolved head revision, cached so ScriptDirectory is parsed at most once.
_head_rev: str | None = None


def _resolve_head() -> str:
    global _head_rev
    if _head_rev is None:
        from alembic.config import Config
        from alembic.script import ScriptDirectory

        cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
        cfg.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
        _head_rev = ScriptDirectory.from_config(cfg).get_current_head() or "head"
    return _head_rev


def install_cache(c: MigrationTemplateCache | None) -> None:
    """Install (or clear) the module-level template cache."""
    global _cache
    _cache = c


def _alembic_url(pg: object) -> str:
    """Return the connection URL alembic env.py expects."""
    host = pg.get_container_host_ip()  # type: ignore[attr-defined]
    port = pg.get_exposed_port(5432)  # type: ignore[attr-defined]
    user = pg.username  # type: ignore[attr-defined]
    password = pg.password  # type: ignore[attr-defined]
    db = pg.dbname  # type: ignore[attr-defined]
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def _run_alembic(
    args: list[str],
    db_url: str,
    *,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run Alembic in-process, with transparent template-clone acceleration.

    When a ``MigrationTemplateCache`` is installed and the target DB is
    virgin (no ``alembic_version`` table), the first upgrade is satisfied
    by dropping the empty DB and cloning from a pre-migrated template.
    Subsequent upgrades/downgrades on already-migrated DBs always run the
    real Alembic path so upgrade→insert→upgrade and downgrade chains work.
    """
    if (
        _cache is not None
        and args[0] == "upgrade"
        and _is_virgin(db_url)
    ):
        revision = args[1]
        cache_key = _resolve_head() if revision == "head" else revision

        m = re.match(r"(postgresql://[^/]+/)(.+)", db_url)
        if m:
            target_dbname = m.group(2)
            admin_url = re.sub(r"/[^/]+$", f"/{_cache._container.dbname}", db_url)

            # Drop the empty target DB.
            with psycopg.connect(admin_url, autocommit=True) as adm:
                adm.execute(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname = %s AND pid <> pg_backend_pid()",
                    (target_dbname,),
                )
                adm.execute(f'DROP DATABASE IF EXISTS "{target_dbname}"')

            # Ensure template exists, then clone.
            tmpl_name = _cache._ensure_template(cache_key, extra_env=extra_env)
            with psycopg.connect(admin_url, autocommit=True) as adm:
                adm.execute(
                    f'CREATE DATABASE "{target_dbname}" TEMPLATE "{tmpl_name}"'
                )
            return subprocess.CompletedProcess(args, 0, "", "")

    return _run_alembic_raw(args, db_url, extra_env=extra_env)


def _is_virgin(db_url: str) -> bool:
    """True when the database has no ``alembic_version`` table."""
    with psycopg.connect(db_url) as conn:
        row = conn.execute(
            "SELECT EXISTS ("
            "  SELECT 1 FROM pg_tables"
            "  WHERE schemaname = 'public' AND tablename = 'alembic_version'"
            ")"
        ).fetchone()
        return not (row and row[0])


def _run_alembic_raw(
    args: list[str],
    db_url: str,
    *,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run Alembic in-process without cache lookup."""
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    stdout = StringIO()
    stderr = StringIO()
    env_patch = {"AIOS_DB_URL": db_url}
    if extra_env:
        env_patch.update(extra_env)
    try:
        with (
            mock.patch.dict(os.environ, env_patch),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            if args[0] == "upgrade":
                command.upgrade(cfg, args[1])
            elif args[0] == "downgrade":
                command.downgrade(cfg, args[1])
            else:
                raise ValueError(f"unsupported alembic command: {args}")
    except Exception as exc:
        stderr.write(str(exc))
        return subprocess.CompletedProcess(args, 1, stdout.getvalue(), stderr.getvalue())
    return subprocess.CompletedProcess(args, 0, stdout.getvalue(), stderr.getvalue())


@pytest.fixture(autouse=True, scope="session")
def _install_migration_cache(
    migration_template_cache: "MigrationTemplateCache",
) -> None:
    """Wire the session-scoped template cache into ``_run_alembic``."""
    install_cache(migration_template_cache)


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
                # Subsystem tables (0033, #328 PR 2/8) live alongside today's
                # connector tables until the code switch in PR 4.
                "connectors",
                "bindings",
                "chat_sessions",
                "routing_rules",
                "runtimes",
                "runtime_tokens",
                "inbound_acks",
            } <= names, f"missing tables: {names}"

            # Spot-check a few critical indexes
            indexes = await conn.fetch(
                "SELECT indexname FROM pg_indexes WHERE schemaname = 'public';"
            )
            index_names = {row["indexname"] for row in indexes}
            for required in (
                "credentials_name_uniq",
                "agents_name_uniq",
                "events_session_message_seq_idx",
                "events_model_request_end_calibration_idx",
                "bindings_connection_active_uniq",
                "runtime_tokens_connector_idx",
            ):
                assert required in index_names, f"missing index {required}"
            assert "events_session_seq_idx" not in index_names, (
                "events_session_seq_idx should be dropped by migration 0080"
            )
        finally:
            await conn.close()

    asyncio.run(check())


# NOTE: down/up cycle tests for migrations 0017 (focal-channel) and 0018
# (events.channel) used to live here but were removed when migration 0026
# (connector redesign #200) dropped ``channel_bindings`` / ``routing_rules``
# / old ``connections``.  The 0019 downgrade adds a column to
# ``channel_bindings``, which 0026 has already dropped — so any downgrade
# chain through 0019 fails.  Per the connector-redesign plan, 0026's
# downgrade is data-lossy and exists only so ``alembic downgrade`` doesn't
# error, not as a rollback path.  Cycle tests for those earlier columns no
# longer make sense; the upgrade-to-head test below is what verifies the
# migration ladder actually applies cleanly.
