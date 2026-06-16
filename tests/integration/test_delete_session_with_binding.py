"""Integration test: ``delete_session`` must succeed when the session
was ever attached to a connection (or otherwise had a ``bindings`` row),
and its ``bindings`` rows must be cleaned up — by the DB cascade, with no
help from any application-side hand-DELETE.

``bindings.session_id`` now carries ``ON DELETE CASCADE`` (migration 0105,
composite tenant FK ``(session_id, account_id) REFERENCES
sessions(id, account_id)`` per the 0093 precedent), uniform with every
other session-children FK (``session_vaults``, ``session_memory_stores``,
``session_github_repositories``, ``files``, ``events``, ``chat_sessions``).
The 0033 connector redesign had recreated ``bindings`` with a bare
``REFERENCES sessions(id)`` and regressed the cascade the original 0015
table carried; ``delete_session`` papered over it with an explicit
``DELETE FROM bindings``.

Before the cascade was restored, ``DELETE FROM sessions`` tripped the FK
and raised ``asyncpg.ForeignKeyViolationError`` whenever a binding row
referenced the session (the operator-curated single_session attach flow
leaves an archived row even after detach, so any session ever bound via
the connector tools was undeletable — surfacing as a 500 from
``DELETE /v1/sessions/{id}``). The cascade makes that correct-by-
construction: this test guards that ``delete_session`` succeeds and the
binding is gone, so it keeps passing without the removed hand-DELETE.
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

        Before the cascade was restored, ``asyncpg.ForeignKeyViolationError``
        propagated from ``DELETE FROM sessions`` because
        ``bindings.session_id`` had no ``ON DELETE CASCADE`` and
        ``delete_session`` no longer carries a compensating
        ``DELETE FROM bindings``. The 0105 cascade makes the row deletion
        cascade in Postgres, so this passes purely on the constraint.
        """
        pool, session_id = pool_session_with_binding
        async with pool.acquire() as conn:
            # Without the 0105 cascade this raises ForeignKeyViolationError,
            # since delete_session no longer pre-deletes from bindings.
            await queries.delete_session(conn, session_id, account_id="acc_a")
            # Bindings for this session must be gone — cascaded by Postgres.
            remaining = await conn.fetchval(
                "SELECT COUNT(*) FROM bindings WHERE session_id = $1",
                session_id,
            )
        assert remaining == 0, (
            f"delete_session left {remaining} binding row(s) referencing "
            f"the deleted session — bindings.session_id must carry ON DELETE "
            f"CASCADE so the row deletion cascades."
        )

    async def test_raw_delete_from_sessions_cascades_bindings(
        self,
        pool_session_with_binding: tuple[asyncpg.Pool[Any], str],
    ) -> None:
        """A session-deletion path that knows *nothing* about bindings —
        a raw ``DELETE FROM sessions`` — must still succeed and leave no
        stranded binding row.

        This is the foreclosure guarantee of #1095: the invariant is held
        by the DB constraint, not by application-code vigilance, so any new
        delete path (bulk-purge, admin DELETE, a ``delete_session`` variant)
        cannot strand a binding or trip the FK. Before the 0105 cascade this
        raw ``DELETE`` tripped ``bindings_session_id_fkey`` and raised
        ``asyncpg.ForeignKeyViolationError``.
        """
        pool, session_id = pool_session_with_binding
        async with pool.acquire() as conn:
            # Without the cascade this raises ForeignKeyViolationError.
            await conn.execute(
                "DELETE FROM sessions WHERE id = $1 AND account_id = $2",
                session_id,
                "acc_a",
            )
            remaining = await conn.fetchval(
                "SELECT COUNT(*) FROM bindings WHERE session_id = $1",
                session_id,
            )
        assert remaining == 0, (
            f"raw DELETE FROM sessions left {remaining} binding row(s) — "
            f"bindings.session_id ON DELETE CASCADE must clear them at "
            f"delete time regardless of application code."
        )
