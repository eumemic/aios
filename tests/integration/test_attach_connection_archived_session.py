"""Integration test: ``attach_connection`` must refuse to bind a connection
to an archived session.

The defect is the bind-time twin of PR #526 / PR #541 (resolver-side
checks for archived bound sessions).  Even with the resolver-side
checks in place, ``service.attach_connection`` happily inserts an
active ``bindings`` row pointing at an archived ``session_id``: the FK
constraint references the ``sessions`` table without a partial
``WHERE archived_at IS NULL`` predicate, and ``insert_binding`` only
maps FK violations to 4xx errors — an archived row still satisfies the
FK so no violation fires.

Operator-visible symptom: ``POST /v1/connections/:id/attach`` returns
200 with a ``Connection`` carrying the archived ``session_id``.
Operator believes the connection is live; subsequent inbounds are
DETACHED (resolver fix #526) so messages never reach a session.
Synchronous bind-time failure is the right surface — operators learn
about the misconfiguration *at the action that caused it*, not silently
on the next inbound from a real user.
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
from aios.services import connections as connections_service
from aios.services import environments as environments_service

pytestmark = pytest.mark.integration


@pytest.fixture
async def archived_session_and_detached_connection(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str, str]]:
    """Yield ``(pool, account_id, connection_id, archived_session_id)``
    where the connection is detached (no active binding) and the
    session has been archived."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_attach_arch', NULL, TRUE, 'attach-archived-test')
                """
            )
        agent = await agents_service.create_agent(
            pool,
            account_id="acc_attach_arch",
            name="attach-arch-test",
            model="openrouter/test",
            system="",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        env = await environments_service.create_environment(
            pool, account_id="acc_attach_arch", name="attach-arch-env"
        )
        async with pool.acquire() as conn:
            session = await queries.insert_session(
                conn,
                account_id="acc_attach_arch",
                agent_id=agent.id,
                environment_id=env.id,
                agent_version=agent.version,
                title=None,
                metadata={},
            )
            connection = await queries.insert_connection(
                conn,
                connector="echo",
                external_account_id="test-account",
                metadata={},
                account_id="acc_attach_arch",
            )
            await queries.archive_session(conn, session.id, account_id="acc_attach_arch")
        yield pool, "acc_attach_arch", connection.id, session.id
    finally:
        await pool.close()


async def test_attach_connection_refuses_archived_session(
    archived_session_and_detached_connection: tuple[asyncpg.Pool[Any], str, str, str],
) -> None:
    """Binding an archived session must surface synchronously as a 4xx
    so the operator learns at the action that caused the misconfiguration,
    not silently on the next inbound (which the resolver fix #526 would
    DETACH).  Pre-fix ``attach_connection`` returns the connection with
    the archived session_id; post-fix it raises :class:`ConflictError`."""
    pool, account_id, connection_id, archived_session_id = archived_session_and_detached_connection

    with pytest.raises(ConflictError) as excinfo:
        await connections_service.attach_connection(
            pool, connection_id, session_id=archived_session_id, account_id=account_id
        )

    detail = excinfo.value.detail
    assert detail is not None
    assert detail.get("session_id") == archived_session_id, (
        f"ConflictError must carry the archived session_id in detail for the "
        f"operator to act on. Got detail={detail!r}."
    )
