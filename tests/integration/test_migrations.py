"""Verify the initial migration applies cleanly against a real Postgres."""

from __future__ import annotations

import os
import subprocess
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

import asyncpg
import pytest

from tests.conftest import needs_docker

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _alembic_url(pg: object) -> str:
    """Return the connection URL alembic env.py expects."""
    host = pg.get_container_host_ip()  # type: ignore[attr-defined]
    port = pg.get_exposed_port(5432)  # type: ignore[attr-defined]
    user = pg.username  # type: ignore[attr-defined]
    password = pg.password  # type: ignore[attr-defined]
    db = pg.dbname  # type: ignore[attr-defined]
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def _run_alembic(args: list[str], db_url: str) -> subprocess.CompletedProcess[str]:
    """Run Alembic in-process, avoiding 132 repeated ``uv`` interpreter boots."""
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    stdout = StringIO()
    stderr = StringIO()
    try:
        with (
            mock.patch.dict(os.environ, {"AIOS_DB_URL": db_url}),
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
