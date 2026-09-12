"""Integration: the real ``delete_session`` service call site purges the
per-session ``/root/.cache`` (and ``/tmp``) bind-mount sources synchronously.

The bug under fix: commit ``db1ebfa4`` (#2347/#2348) added the per-session
``_cache/<id>`` bind-mount source and the asynchronous ``_reap_session_cache``
reaper, but omitted the matching ``(session_cache_dir(session_id), ...)`` entry
from ``purge_session_directories``. So a deleted session's ``_tmp/<id>`` was
removed by the synchronous delete-time purge (``af52d8ac``) while its
``_cache/<id>`` survived on disk until the periodic ``host_dir_reaper`` later
observed it — and when ``host_dir_reaper_enabled=False`` (the documented P1
disk-fill override) ``_cache`` had no cleanup at all.

These tests drive the full ``aios.services.sessions.delete_session`` path
(``src/aios/services/sessions.py:2630``), which reads the session's
``workspace_volume_path`` row, commits the row deletion, and then calls
``purge_session_directories`` inside the same transaction block — against a
real migrated Postgres (testcontainer). They assert the on-disk
``_cache/<id>`` directory is gone the moment ``delete_session`` returns,
independent of the reaper kill-switch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import asyncpg
import pytest

from aios.config import get_settings
from aios.db import queries
from aios.db.pool import create_pool
from aios.sandbox.volumes import (
    session_attachments_dir,
    session_cache_dir,
    session_repos_root,
    session_tmp_dir,
    session_uploads_dir,
)
from aios.services import agents as agents_service
from aios.services import environments as environments_service
from aios.services import sessions as sessions_service

pytestmark = pytest.mark.integration

ACCOUNT = "acc_purge_cache"
PER_SESSION_DIRS = ("_uploads", "_attachments", "_session_repos", "_tmp", "_cache")


async def _seed(pool: asyncpg.Pool[Any], *, workspace_root: Path) -> str:
    """Seed an account/agent/env/session whose ``workspace_volume_path`` lives
    under ``workspace_root``; create every per-session host dir on disk.
    Returns the session id."""
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
            "VALUES ($1, NULL, TRUE, 'purge-cache-root')",
            ACCOUNT,
        )
    agent = await agents_service.create_agent(
        pool,
        account_id=ACCOUNT,
        name="purge-cache-agent",
        model="openrouter/test",
        system="",
        tools=[],
        description=None,
        metadata={},
        window_min=50_000,
        window_max=150_000,
    )
    env = await environments_service.create_environment(
        pool, account_id=ACCOUNT, name="purge-cache-env"
    )
    async with pool.acquire() as conn:
        session = await queries.insert_session(
            conn,
            account_id=ACCOUNT,
            agent_id=agent.id,
            environment_id=env.id,
            agent_version=agent.version,
            title=None,
            metadata={},
        )
    session_id = session.id
    # Materialize every per-session host dir on disk (the bind-mount sources +
    # the workspace). ``insert_session`` already persisted the canonical
    # ``workspace_volume_path = workspace_root / ACCOUNT / session_id``.
    workspace = workspace_root / ACCOUNT / session_id
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "marker.txt").write_text("payload")
    for child in PER_SESSION_DIRS:
        d = workspace_root / child / session_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "marker.txt").write_text("payload")
    return session_id


async def test_delete_session_purges_cache_dir_synchronously(
    migrated_db_url: str, _reset_db_state: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real ``delete_session`` service removes ``_cache/<id>`` (and
    ``_tmp/<id>``) the moment it returns — no reaper tick required.

    Regression for the db1ebfa4 omission: before the fix ``_cache/<id>`` survived
    ``delete_session`` while ``_tmp/<id>`` was purged.
    """
    settings = get_settings()
    monkeypatch.setattr(settings, "workspace_root", tmp_path)

    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        session_id = await _seed(pool, workspace_root=tmp_path)

        # All per-session dirs present at delete time.
        assert session_cache_dir(session_id).exists()
        assert session_tmp_dir(session_id).exists()

        await sessions_service.delete_session(pool, session_id, account_id=ACCOUNT)

        # Synchronous delete-time purge: every per-session host dir is gone.
        assert not (tmp_path / ACCOUNT / session_id).exists(), "workspace dir survived delete"
        assert not session_uploads_dir(session_id).exists()
        assert not session_attachments_dir(session_id).exists()
        assert not session_repos_root(session_id).exists()
        assert not session_tmp_dir(session_id).exists(), "tmp MUST be purged (#2280)"
        assert not session_cache_dir(session_id).exists(), (
            "cache MUST be purged by delete_session, not linger until host_dir_reaper"
        )
        # The session row is gone.
        async with pool.acquire() as conn:
            row = await conn.fetchval(
                "SELECT id FROM sessions WHERE id=$1 AND account_id=$2",
                session_id,
                ACCOUNT,
            )
        assert row is None, "delete_session left the session row behind"
    finally:
        await pool.close()


async def test_delete_session_purges_cache_dir_with_reaper_kill_switch_off(
    migrated_db_url: str, _reset_db_state: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With ``host_dir_reaper_enabled=False`` the reaper is a no-op, so the
    synchronous ``purge_session_directories`` (called by ``delete_session``) is
    the sole kill-switch-independent reclamation path. ``_cache/<id>`` must
    still vanish at delete time — closing the unbounded-accumulation gap under
    the documented P1 disk-fill override."""
    settings = get_settings()
    monkeypatch.setattr(settings, "workspace_root", tmp_path)
    monkeypatch.setattr(settings, "host_dir_reaper_enabled", False)

    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        session_id = await _seed(pool, workspace_root=tmp_path)
        cache_dir = session_cache_dir(session_id)
        tmp_dir = session_tmp_dir(session_id)
        assert cache_dir.exists() and tmp_dir.exists()

        await sessions_service.delete_session(pool, session_id, account_id=ACCOUNT)

        assert not tmp_dir.exists(), "tmp must be purged even with the reaper disabled"
        assert not cache_dir.exists(), (
            "cache must be purged synchronously even with the reaper disabled"
        )
    finally:
        await pool.close()
