"""DB-backed shared-run keep-set and activation-vs-reap mutex tests (#1970)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool

pytestmark = [pytest.mark.integration, pytest.mark.docker]


async def _seed(conn: asyncpg.Connection[Any]) -> None:
    await conn.execute(
        """
        INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
        VALUES ('acc_reaper', NULL, TRUE, 'reaper')
        """
    )
    await conn.execute(
        "INSERT INTO environments (id, name, config, account_id) VALUES "
        "('env_reaper', 'reaper', '{}'::jsonb, 'acc_reaper')"
    )


async def _insert_run(
    conn: asyncpg.Connection[Any],
    run_id: str,
    status: str,
    mode: str,
    path: str,
    *,
    archived: bool = False,
) -> None:
    await conn.execute(
        """
        INSERT INTO wf_runs
          (id, account_id, environment_id, script, script_sha, status,
           workspace_mode, workspace_path, archived_at)
        VALUES ($1, 'acc_reaper', 'env_reaper', 'async def main(input): return input',
                'sha', $2, $3, $4, CASE WHEN $5 THEN now() ELSE NULL END)
        """,
        run_id,
        status,
        mode,
        path,
        archived,
    )


async def test_shared_run_keep_set_sql_semantics(
    migrated_db_url: str, _reset_db_state: None
) -> None:
    """Real rows prove pending/running/suspended live shared pointers only are kept."""
    conn = await asyncpg.connect(migrated_db_url)
    try:
        await _seed(conn)
        rows = [
            ("run_pending", "pending", "shared", "/ws/pending", False),
            ("run_running", "running", "shared", "/ws/running", False),
            ("run_suspended", "suspended", "shared", "/ws/suspended", False),
            ("run_terminal", "completed", "shared", "/ws/terminal", False),
            ("run_fresh", "running", "fresh", "/ws/fresh", False),
            ("run_archived", "running", "shared", "/ws/archived", True),
        ]
        for row in rows:
            await _insert_run(conn, *row[:4], archived=row[4])
        assert set(await queries.unscoped_live_workspace_volume_paths(conn)) == {
            "/ws/pending",
            "/ws/running",
            "/ws/suspended",
        }
    finally:
        await conn.close()


async def test_shared_activation_serializes_with_reap(
    migrated_db_url: str, _reset_db_state: None, tmp_path: Path
) -> None:
    """Activation persists under the mutex; a racing reaper then rechecks and keeps it."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=2)
    path = str((tmp_path / "workspace").resolve())
    activation_locked = asyncio.Event()
    release = asyncio.Event()
    reaper_locked = asyncio.Event()
    try:
        async with pool.acquire() as conn:
            await _seed(conn)

        async def activate() -> None:
            async with pool.acquire() as conn, conn.transaction():
                await queries.acquire_workspace_advisory_xact_lock(conn, path)
                activation_locked.set()
                await release.wait()
                await _insert_run(conn, "run_racing", "pending", "shared", path)

        async def reaper() -> None:
            await activation_locked.wait()
            async with pool.acquire() as conn, conn.transaction():
                await queries.acquire_workspace_advisory_xact_lock(conn, path)
                reaper_locked.set()
                assert await queries.unscoped_workspace_path_is_live(conn, path)

        activate_task = asyncio.create_task(activate())
        reap_task = asyncio.create_task(reaper())
        await activation_locked.wait()
        await asyncio.sleep(0.05)
        assert not reaper_locked.is_set(), "reaper must not enter activation's persist window"
        release.set()
        await asyncio.gather(activate_task, reap_task)
    finally:
        await pool.close()
