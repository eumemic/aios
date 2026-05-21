"""Integration test: ``bind_chat_to_session`` must refuse to bind a
chat to an archived session.

Sibling write path to PR #549 (``attach_connection`` archived-session
check). ``bind_chat_to_session`` validates ``session_id`` via
``queries.get_session`` (which does NOT filter ``archived_at IS NULL``)
and then stamps the ``chat_sessions`` ledger; the operator receives a
201 ``BoundChat`` carrying the archived ``session_id``. Subsequent
inbounds DETACH at the resolver's tier-1 check (post-#541), so the
messages never land — but that's a delayed signal from a different
actor. The operator who just bound the chat believes it's live and
discovers otherwise only when a real user pings.

Bind-time synchronous 4xx is the right surface, mirroring #549's
``attach_connection`` fix: the operator learns at the action that
caused the misconfiguration, not silently on the next inbound.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.errors import ConflictError
from aios.services import connections as connections_service
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration


@pytest.fixture
async def archived_session_and_connection(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str, str]]:
    """Yield ``(pool, account_id, connection_id, archived_session_id)``
    where the session has been archived after creation."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_bind_arch', NULL, TRUE, 'bind-archived-test')
                """
            )
        _agent, _env, session = await seed_agent_env_session(
            pool, account_id="acc_bind_arch", prefix="bind-arch"
        )
        async with pool.acquire() as conn:
            connection = await queries.insert_connection(
                conn,
                connector="echo",
                external_account_id="test-account",
                metadata={},
                account_id="acc_bind_arch",
            )
            await queries.archive_session(conn, session.id, account_id="acc_bind_arch")
        yield pool, "acc_bind_arch", connection.id, session.id
    finally:
        await pool.close()


async def test_bind_chat_to_session_refuses_archived_session(
    archived_session_and_connection: tuple[asyncpg.Pool[Any], str, str, str],
) -> None:
    """Pre-fix: ``bind_chat_to_session`` returns 201 with a ``BoundChat``
    carrying the archived ``session_id`` and stamps the ``chat_sessions``
    ledger. Post-fix: raises ``ConflictError`` carrying ``session_id``
    in detail, and the ledger is left unstamped."""
    pool, account_id, connection_id, archived_session_id = archived_session_and_connection
    chat_id = "operator-bound-chat"

    with pytest.raises(ConflictError) as excinfo:
        await connections_service.bind_chat_to_session(
            pool,
            connection_id,
            chat_id=chat_id,
            session_id=archived_session_id,
            account_id=account_id,
        )

    detail = excinfo.value.detail
    assert detail is not None
    assert detail.get("session_id") == archived_session_id, (
        f"ConflictError must carry the archived session_id in detail for the "
        f"operator to act on. Got detail={detail!r}."
    )

    # Ledger must remain unstamped — confirms the refusal is atomic
    # (no half-completed write that the loser of a race could discover).
    async with pool.acquire() as conn:
        ledger_session_id = await queries.lookup_chat_session(
            conn, connection_id, chat_id, account_id=account_id
        )
    assert ledger_session_id is None, (
        f"chat_sessions ledger must not be stamped when the bind is refused; "
        f"found stamped session_id={ledger_session_id!r}."
    )
