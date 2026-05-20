"""Integration test: the connector resolver must not return an
archived session_id from any of its three tiers.

After PR #523 (``append_event`` refuses archived sessions), an inbound
that flowed through the resolver pointing at an archived target would
fail downstream with ``InboundDrop.SESSION_MISSING`` — the inbound
router translates that to a 500-class status because semantically
"session vanished mid-flight" is a transient error. Well-behaved
connectors treat 5xx as retry-with-backoff, so every queued post-archive
inbound becomes a permanent retry loop hammering the API.

The CORRECT terminal signal is ``InboundDrop.DETACHED`` (router maps to
422 "operator config issue"), which tells the connector "stop retrying,
the operator must reconfigure." This is the same shape as
``InboundDrop.ARCHIVED_TEMPLATE`` for the per_chat path — the resolver
already returns ``DETACHED``/``ARCHIVED_TEMPLATE`` for permanent-config
failures upstream of the append, and archived single_session targets
deserve the same treatment.

PR #526 covered tier-3 ``single_session``.  Tier-1 (stale ledger entry
pointing at an archived session) is the routine variant: any per_chat
session spawned by tier-3 and stamped into the ledger becomes a stale
entry the moment the operator archives the session.  Subsequent
inbounds short-circuit tier-1's ``lookup_chat_session`` and land in the
archived id, where ``append_event`` (post-#523) raises NotFoundError →
``SESSION_MISSING`` → 500 → connector retry-forever.  Stamp-then-
archive is exactly the lifecycle every per_chat session goes through,
so the ledger is the most-reachable leak surface, not the least.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.services import agents as agents_service
from aios.services import connections as connections_service
from aios.services import environments as environments_service

pytestmark = pytest.mark.integration


@pytest.fixture
async def archived_session_bound_to_connection(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str, str]]:
    """Yield ``(pool, account_id, connection_id, session_id)`` for a
    single_session binding whose target session has been archived."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_resolver_arch', NULL, TRUE, 'resolver-archived-test')
                """
            )
        agent = await agents_service.create_agent(
            pool,
            account_id="acc_resolver_arch",
            name="resolver-arch-test",
            model="openrouter/test",
            system="",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        env = await environments_service.create_environment(
            pool, account_id="acc_resolver_arch", name="resolver-arch-env"
        )
        async with pool.acquire() as conn:
            session = await queries.insert_session(
                conn,
                account_id="acc_resolver_arch",
                agent_id=agent.id,
                environment_id=env.id,
                agent_version=agent.version,
                title=None,
                metadata={},
            )
            # Insert a connection directly via the DB (the public
            # ``create_connection`` API requires a richer fixture set).
            connection = await queries.insert_connection(
                conn,
                connector="echo",
                external_account_id="test-account",
                metadata={},
                account_id="acc_resolver_arch",
            )
        # Bind single_session, then archive the target session.
        await connections_service.attach_connection(
            pool,
            connection.id,
            account_id="acc_resolver_arch",
            session_id=session.id,
        )
        async with pool.acquire() as conn:
            await queries.archive_session(conn, session.id, account_id="acc_resolver_arch")
        yield pool, "acc_resolver_arch", connection.id, session.id
    finally:
        await pool.close()


async def test_resolver_returns_detached_for_archived_single_session_target(
    archived_session_bound_to_connection: tuple[asyncpg.Pool[Any], str, str, str],
) -> None:
    """Tier-3 ``single_session`` resolution must treat an archived
    target as DETACHED so connectors get a 422 terminal signal instead
    of a 500-class retry-forever ``SESSION_MISSING`` from the
    downstream ``append_event`` failure."""
    from aios_connectors.resolver import ResolveDrop, resolve_target_session

    pool, account_id, connection_id, _archived_session_id = archived_session_bound_to_connection
    async with pool.acquire() as conn:
        connection = await queries.get_connection(conn, connection_id, account_id=account_id)

    result = await resolve_target_session(
        pool, connection=connection, chat_id="any-chat", account_id=account_id
    )

    assert result.drop == ResolveDrop.DETACHED, (
        f"resolver must return DETACHED (→ router 422) for archived "
        f"single_session targets, not pass through the archived id and let "
        f"the inbound retry-loop on append_event NotFoundError. Got "
        f"{result!r}. Pre-fix the resolver returns the archived session_id "
        f"with drop=None; post-fix it returns drop=DETACHED with session_id "
        f"None."
    )
    assert result.session_id is None


@pytest.fixture
async def chat_sessions_ledger_with_archived_session(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str, str]]:
    """Yield ``(pool, account_id, connection_id, chat_id)`` for a
    ``chat_sessions`` ledger row whose target session has been archived.

    Models the routine per_chat lifecycle: tier-3 spawned a session and
    stamped the ledger, then the operator archived that session.  The
    ledger now points at an archived id; tier-1 lookup must not return
    it (or future inbounds 500-retry-forever — see module docstring)."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    chat_id = "stale-ledger-chat"
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_ledger_arch', NULL, TRUE, 'ledger-archived-test')
                """
            )
        agent = await agents_service.create_agent(
            pool,
            account_id="acc_ledger_arch",
            name="ledger-arch-test",
            model="openrouter/test",
            system="",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        env = await environments_service.create_environment(
            pool, account_id="acc_ledger_arch", name="ledger-arch-env"
        )
        async with pool.acquire() as conn:
            session = await queries.insert_session(
                conn,
                account_id="acc_ledger_arch",
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
                account_id="acc_ledger_arch",
            )
            # Stamp the ledger BEFORE archiving — mirrors the order
            # produced by a real per_chat spawn followed by an operator
            # archive of the spawned session.
            await queries.insert_chat_session(
                conn,
                connection_id=connection.id,
                chat_id=chat_id,
                session_id=session.id,
                account_id="acc_ledger_arch",
            )
            await queries.archive_session(conn, session.id, account_id="acc_ledger_arch")
        yield pool, "acc_ledger_arch", connection.id, chat_id
    finally:
        await pool.close()


async def test_resolver_returns_detached_for_stale_ledger_archived_session(
    chat_sessions_ledger_with_archived_session: tuple[asyncpg.Pool[Any], str, str, str],
) -> None:
    """Tier-1 ``lookup_chat_session`` short-circuits on the stamped
    ledger row without checking the bound session's ``archived_at``.

    Routine per_chat lifecycle: tier-3 stamps the ledger at spawn time
    (session live), operator later archives the session.  The ledger
    row is now stale; the next inbound's tier-1 lookup returns the
    archived id with ``drop=None``; ``handle_inbound`` proceeds to
    ``append_event`` which (post-#523) raises NotFoundError; the
    inbound surfaces as ``SESSION_MISSING`` → HTTP 500; connectors
    retry-forever on 5xx.  The resolver must return ``DETACHED``
    instead so the connector sees a 422 terminal signal.

    Same defect class PR #526 closed for tier-3 single_session.  Tier-1
    is the more reachable surface because every per_chat-spawned
    session that the operator later archives produces a stale ledger
    entry — and per_chat is the common case."""
    from aios_connectors.resolver import ResolveDrop, resolve_target_session

    pool, account_id, connection_id, chat_id = chat_sessions_ledger_with_archived_session
    async with pool.acquire() as conn:
        connection = await queries.get_connection(conn, connection_id, account_id=account_id)

    result = await resolve_target_session(
        pool, connection=connection, chat_id=chat_id, account_id=account_id
    )

    assert result.drop == ResolveDrop.DETACHED, (
        f"tier-1 lookup must filter archived sessions out of the ledger so "
        f"connectors get a 422 terminal signal instead of 500-retry-forever "
        f"from the downstream append_event NotFoundError. Got {result!r}. "
        f"Pre-fix the resolver returns the archived session_id with "
        f"drop=None; post-fix it returns drop=DETACHED with session_id None."
    )
    assert result.session_id is None
