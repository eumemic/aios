"""Integration tests for migration 0057's workspace_volume_path backfill.

Before #626 ``AIOS_WORKSPACE_ROOT`` could be configured as a relative path
(e.g. ``./workspaces`` in a dev ``.env``); ``insert_session`` then stored
``workspaces/<account>/<session>`` on the row.  ``Path.resolve()`` resolves
those relative strings against the worker's current working directory at
provision time, lands outside the workspace jail, and surfaces
``ForbiddenError`` on every tool call.  Migration 0057 backfills the legacy
rows to absolute by prepending the operator-supplied ``AIOS_WORKSPACE_ROOT``.

These tests exercise the real alembic CLI against a fresh Postgres for each
case so each migration run gets an isolated ``alembic_version`` and
``sessions`` table.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import asyncpg
import pytest

from tests.conftest import _docker_available, needs_docker
from tests.integration.test_migrations import PROJECT_ROOT, _alembic_url


@pytest.fixture
def postgres() -> Iterator[object]:
    """Fresh function-scoped Postgres — each test mutates ``alembic_version``."""
    if not _docker_available():
        pytest.skip("Docker not available")
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("postgres:16-alpine") as pg:
        yield pg


def _run_alembic_with_env(
    args: list[str], db_url: str, *, workspace_root: str
) -> subprocess.CompletedProcess[str]:
    """Like ``_run_alembic`` but also threads ``AIOS_WORKSPACE_ROOT`` through.

    The 0057 migration reads ``AIOS_WORKSPACE_ROOT`` to know the prefix it
    should prepend to legacy relative paths.  ``test_migrations._run_alembic``
    starts from a clean env (only ``PATH`` / ``AIOS_DB_URL`` / ``HOME``) so we
    can't reuse it directly.
    """
    uv = shutil.which("uv")
    if uv is None:
        raise FileNotFoundError("uv not found on PATH")
    return subprocess.run(
        [uv, "run", "alembic", *args],
        cwd=PROJECT_ROOT,
        env={
            "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/local/bin"),
            "AIOS_DB_URL": db_url,
            "AIOS_WORKSPACE_ROOT": workspace_root,
            "HOME": str(Path.home()),
        },
        capture_output=True,
        text=True,
        check=False,
    )


async def _list_workspace_paths(db_url: str) -> list[tuple[str, str]]:
    """Return ``(id, workspace_volume_path)`` for every session, ordered by id."""
    conn = await asyncpg.connect(db_url)
    try:
        rows = await conn.fetch("SELECT id, workspace_volume_path FROM sessions ORDER BY id")
        return [(row["id"], row["workspace_volume_path"]) for row in rows]
    finally:
        await conn.close()


async def _seed_session(db_url: str, *, session_id: str, workspace_volume_path: str) -> None:
    """Insert a session row directly via SQL, bypassing the service layer.

    We can't go through ``insert_session`` here — its ``Path.is_absolute()``
    guard (added by #626) would reject the relative path we're trying to
    seed.  Raw SQL is what lets us reproduce the pre-#626 legacy state.
    """
    conn = await asyncpg.connect(db_url)
    try:
        # Seed dependencies first.  Accounts / agents / environments have
        # FK constraints from ``sessions`` so they must exist before the
        # session row inserts.
        await conn.execute(
            """
            INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
            VALUES ('acc_root', NULL, TRUE, 'root')
            ON CONFLICT DO NOTHING
            """
        )
        await conn.execute(
            """
            INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
            VALUES ('acc_a', 'acc_root', FALSE, 'a')
            ON CONFLICT DO NOTHING
            """
        )
        await conn.execute(
            """
            INSERT INTO agents (
                id, account_id, name, model, system, tools, description, metadata,
                window_min, window_max, version, created_at, updated_at
            )
            VALUES (
                'agent_test', 'acc_a', 'test', 'openrouter/test', '', '[]'::jsonb,
                NULL, '{}'::jsonb, 50000, 150000, 1, now(), now()
            )
            ON CONFLICT DO NOTHING
            """
        )
        await conn.execute(
            """
            INSERT INTO environments (id, account_id, name, config, created_at, updated_at)
            VALUES ('env_test', 'acc_a', 'test', '{}'::jsonb, now(), now())
            ON CONFLICT DO NOTHING
            """
        )
        await conn.execute(
            """
            INSERT INTO sessions (
                id, account_id, agent_id, environment_id, agent_version,
                title, metadata, status, workspace_volume_path, env,
                focal_channel, focal_locked
            )
            VALUES (
                $1, 'acc_a', 'agent_test', 'env_test', 1,
                NULL, '{}'::jsonb, 'idle', $2, '{}'::jsonb,
                NULL, FALSE
            )
            """,
            session_id,
            workspace_volume_path,
        )
    finally:
        await conn.close()


@needs_docker
@pytest.mark.integration
def test_relative_path_rewritten_to_absolute(postgres: object) -> None:
    """A row with ``workspaces/acc/sess`` is rewritten by 0057 to
    ``<AIOS_WORKSPACE_ROOT>/acc/sess`` with no other changes."""
    db_url = _alembic_url(postgres)
    workspace_root = "/var/lib/aios/workspaces"

    # Upgrade only as far as 0056 so we can seed the legacy state.
    result = _run_alembic_with_env(["upgrade", "0056"], db_url, workspace_root=workspace_root)
    assert result.returncode == 0, f"alembic upgrade failed:\n{result.stderr}\n{result.stdout}"

    asyncio.run(
        _seed_session(
            db_url,
            session_id="sess_legacy",
            workspace_volume_path="workspaces/acc_a/sess_legacy",
        )
    )

    # Run the 0057 backfill.
    result = _run_alembic_with_env(["upgrade", "head"], db_url, workspace_root=workspace_root)
    assert result.returncode == 0, f"alembic upgrade failed:\n{result.stderr}\n{result.stdout}"

    paths = asyncio.run(_list_workspace_paths(db_url))
    assert paths == [("sess_legacy", "/var/lib/aios/workspaces/acc_a/sess_legacy")]


@needs_docker
@pytest.mark.integration
def test_absolute_paths_untouched(postgres: object) -> None:
    """Rows whose ``workspace_volume_path`` is already absolute must be
    left alone.  The migration's ``WHERE`` clause filters on
    ``NOT LIKE '/%'`` so an absolute path never matches."""
    db_url = _alembic_url(postgres)
    workspace_root = "/var/lib/aios/workspaces"

    result = _run_alembic_with_env(["upgrade", "0056"], db_url, workspace_root=workspace_root)
    assert result.returncode == 0, f"alembic upgrade failed:\n{result.stderr}\n{result.stdout}"

    absolute = "/var/lib/aios/workspaces/acc_a/sess_new"
    asyncio.run(_seed_session(db_url, session_id="sess_new", workspace_volume_path=absolute))

    result = _run_alembic_with_env(["upgrade", "head"], db_url, workspace_root=workspace_root)
    assert result.returncode == 0, f"alembic upgrade failed:\n{result.stderr}\n{result.stdout}"

    paths = asyncio.run(_list_workspace_paths(db_url))
    assert paths == [("sess_new", absolute)]


@needs_docker
@pytest.mark.integration
def test_migration_is_idempotent(postgres: object) -> None:
    """Running 0057 a second time (downgrade then upgrade) doesn't double the
    prefix.  ``WHERE workspace_volume_path NOT LIKE '/%'`` ensures the
    already-rewritten absolute rows don't re-match on a re-upgrade."""
    db_url = _alembic_url(postgres)
    workspace_root = "/var/lib/aios/workspaces"

    result = _run_alembic_with_env(["upgrade", "0056"], db_url, workspace_root=workspace_root)
    assert result.returncode == 0, f"alembic upgrade failed:\n{result.stderr}\n{result.stdout}"

    asyncio.run(
        _seed_session(
            db_url,
            session_id="sess_legacy",
            workspace_volume_path="workspaces/acc_a/sess_legacy",
        )
    )

    result = _run_alembic_with_env(["upgrade", "head"], db_url, workspace_root=workspace_root)
    assert result.returncode == 0, f"alembic upgrade failed:\n{result.stderr}\n{result.stdout}"

    # Down then up — downgrade restores the legacy relative form so the
    # second ``upgrade`` matches it again and rewrites to the same absolute
    # value as the first upgrade.  Idempotency here means "two upgrade runs
    # converge on the same row", not "the WHERE clause skips on re-entry".
    result = _run_alembic_with_env(["downgrade", "0056"], db_url, workspace_root=workspace_root)
    assert result.returncode == 0, f"alembic downgrade failed:\n{result.stderr}\n{result.stdout}"
    result = _run_alembic_with_env(["upgrade", "head"], db_url, workspace_root=workspace_root)
    assert result.returncode == 0, f"alembic upgrade failed:\n{result.stderr}\n{result.stdout}"

    paths = asyncio.run(_list_workspace_paths(db_url))
    assert paths == [("sess_legacy", "/var/lib/aios/workspaces/acc_a/sess_legacy")]


@needs_docker
@pytest.mark.integration
def test_downgrade_reverses_rewrite(postgres: object) -> None:
    """``downgrade 0056`` strips the prepended ``AIOS_WORKSPACE_ROOT`` prefix
    so the row returns to its original relative form."""
    db_url = _alembic_url(postgres)
    workspace_root = "/var/lib/aios/workspaces"

    result = _run_alembic_with_env(["upgrade", "0056"], db_url, workspace_root=workspace_root)
    assert result.returncode == 0, f"alembic upgrade failed:\n{result.stderr}\n{result.stdout}"

    asyncio.run(
        _seed_session(
            db_url,
            session_id="sess_legacy",
            workspace_volume_path="workspaces/acc_a/sess_legacy",
        )
    )

    result = _run_alembic_with_env(["upgrade", "head"], db_url, workspace_root=workspace_root)
    assert result.returncode == 0, f"alembic upgrade failed:\n{result.stderr}\n{result.stdout}"

    result = _run_alembic_with_env(["downgrade", "0056"], db_url, workspace_root=workspace_root)
    assert result.returncode == 0, f"alembic downgrade failed:\n{result.stderr}\n{result.stdout}"

    paths = asyncio.run(_list_workspace_paths(db_url))
    assert paths == [("sess_legacy", "workspaces/acc_a/sess_legacy")]
