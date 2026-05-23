"""Integration test: ``delete_session`` must succeed when the session
was ever attached to a connection (or otherwise had a ``bindings`` row).

``bindings.session_id REFERENCES sessions(id)`` carries NO ``ON DELETE
CASCADE`` (migration 0033). Every other session-children FK (in
``session_vaults``, ``session_memory_stores``, ``session_github_repositories``,
``files``, ``events``, ``chat_sessions``) has the cascade — ``bindings``
is the lone outlier. ``delete_session`` pre-deletes from ``session_vaults``
and ``events`` (redundant given the cascades, but explicit-as-intent
style), then runs ``DELETE FROM sessions`` — which trips the FK
constraint and raises ``asyncpg.ForeignKeyViolationError`` whenever a
binding row references the session.

The operator-curated single_session attach flow leaves a row even
after detach (archived bindings stay in the table), so any session
that's been bound via the connector tools is undeletable. Surfaces as
500 from ``DELETE /v1/sessions/{id}``.

Fix: pre-delete from ``bindings`` in ``delete_session``, mirroring
the explicit-deletes pattern for the other session children.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.services import agents as agents_service
from aios.services import environments as environments_service

pytestmark = pytest.mark.integration


@pytest.fixture
async def pool_session_with_binding(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str]]:
    """Yield ``(pool, session_id)`` for an acc_a session that has an
    active ``bindings`` row tying it to an acc_a connection."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_root', NULL,       TRUE,  'tenant-root'),
                       ('acc_a',    'acc_root', FALSE, 'tenant-a')
                """
            )

        agent_a = await agents_service.create_agent(
            pool,
            account_id="acc_a",
            name="a-agent",
            model="openrouter/test",
            system="",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        env_a = await environments_service.create_environment(
            pool,
            account_id="acc_a",
            name="a-env",
        )
        async with pool.acquire() as conn:
            session = await queries.insert_session(
                conn,
                account_id="acc_a",
                agent_id=agent_a.id,
                environment_id=env_a.id,
                agent_version=agent_a.version,
                title=None,
                metadata={},
            )
            connection = await queries.insert_connection(
                conn,
                account_id="acc_a",
                connector="signal",
                external_account_id="+15550001",
                metadata={},
            )
            await queries.insert_binding(
                conn,
                account_id="acc_a",
                connection_id=connection.id,
                mode="single_session",
                session_id=session.id,
            )
        yield pool, session.id
    finally:
        await pool.close()


class TestDeleteSessionWithBinding:
    async def test_delete_session_removes_bindings(
        self,
        pool_session_with_binding: tuple[asyncpg.Pool[Any], str],
    ) -> None:
        """``delete_session`` must succeed and clean up the binding row
        even when the session has been attached to a connection.

        Pre-fix: ``asyncpg.ForeignKeyViolationError`` propagates from
        ``DELETE FROM sessions`` because ``bindings.session_id`` has no
        ``ON DELETE CASCADE`` and ``delete_session`` doesn't pre-delete
        from ``bindings`` (it pre-deletes from ``session_vaults`` and
        ``events``, but missed ``bindings``).
        """
        pool, session_id = pool_session_with_binding
        async with pool.acquire() as conn:
            # Today: raises asyncpg.ForeignKeyViolationError.
            await queries.delete_session(conn, session_id, account_id="acc_a")
            # Bindings for this session must be gone.
            remaining = await conn.fetchval(
                "SELECT COUNT(*) FROM bindings WHERE session_id = $1",
                session_id,
            )
        assert remaining == 0, (
            f"delete_session left {remaining} binding row(s) referencing "
            f"the deleted session — bindings.session_id has no ON DELETE "
            f"CASCADE and delete_session must pre-delete from bindings."
        )
