"""Integration test: ``clone_session`` must refuse an archived parent.

Pre-fix the cloneability check only consulted ``status``:

    SELECT status FROM sessions WHERE id = $1 AND account_id = $2 FOR UPDATE
    if status in _CLONEABLE_STATUSES: ...

so an archived (idle / terminated) session passed the gate, and
``clone_session`` proceeded to resurrect the parent's full event log
into a live new session — defeating the archive intent (sessions
archived for compliance, retention windows, or simply "this work is
done; freeze it" become live again on clone).

Same defect class as PR #573 (``update_session``), PR #580 / #587
(``clone`` resource-table + ``update_memory_store``) — the archive
intent must hold across every mutation/copy surface, not just the
update surface.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.errors import ConflictError
from aios.services import agents as agents_service
from aios.services import environments as environments_service
from aios.services import sessions as sessions_service

pytestmark = pytest.mark.integration


@pytest.fixture
async def archived_parent_session(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str]]:
    """Yield ``(pool, account_id, parent_session_id)`` for an idle
    session that has been archived."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_clone_arch', NULL, TRUE, 'clone-archived-parent-test')
                """
            )
        agent = await agents_service.create_agent(
            pool,
            account_id="acc_clone_arch",
            name="clone-arch-test",
            model="openrouter/test",
            system="",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        env = await environments_service.create_environment(
            pool, account_id="acc_clone_arch", name="clone-arch-env"
        )
        parent = await sessions_service.create_session(
            pool,
            account_id="acc_clone_arch",
            agent_id=agent.id,
            environment_id=env.id,
            agent_version=agent.version,
            title="parent-to-archive",
            metadata={},
        )
        async with pool.acquire() as conn:
            archived = await queries.archive_session(conn, parent.id, account_id="acc_clone_arch")
        assert archived.archived_at is not None
        yield pool, "acc_clone_arch", parent.id
    finally:
        await pool.close()


async def test_clone_refuses_archived_parent(
    archived_parent_session: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """An archived session must NOT be cloneable: cloning resurrects
    the parent's event log into a live new session, defeating the
    archive intent."""
    pool, account_id, parent_id = archived_parent_session

    with pytest.raises(ConflictError) as excinfo:
        await sessions_service.clone_session(pool, parent_id, account_id=account_id)

    detail = excinfo.value.detail
    assert detail is not None
    assert detail.get("id") == parent_id
