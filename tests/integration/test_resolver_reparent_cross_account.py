"""Integration test: the resolver must DETACH (not route) a cross-account
``session_id`` — the post-``reparent_connection`` state.

Background
----------
``reparent_connection`` (PR #696) rewrites the ``account_id`` of a
connection's child rows (``bindings``, ``chat_sessions``,
``routing_rules``) to the destination, but does **not** rewrite their
``session_id`` / ``target_id`` — those keep pointing at sessions owned
by the SOURCE account (sessions are account-scoped and are not
reparented).

Pre-fix the resolver's ``_session_is_archived`` conflated a 0-row
(cross-account) session lookup with "live" (returned ``False``), so:

* tier-1 (``chat_sessions`` ledger) returned the stale cross-account
  ``session_id`` with ``drop=None``; or
* tier-3 (``single_session`` binding) returned the stale cross-account
  ``binding.session_id`` with ``drop=None``;

``handle_inbound`` then proceeded to ``append_event`` whose
account-scoped seq allocation matched zero rows → ``NotFoundError`` →
``InboundDrop.SESSION_MISSING``. The per-message loss was mislabeled as a
generic ``SESSION_MISSING`` that does not point at reparent as the
cause. The fix makes ``_session_is_archived`` treat a 0-row session the
same as an archived one → ``ResolveDrop.DETACHED`` (→ 422), the terminal
signal the reparent docstring already says is the expected outcome when
a child points at the source account.

These tests exercise the resolver/inbound path post-reparent against a
real Postgres — the path the sibling carry-over tests in
``test_reparent_unique_index.py`` deliberately do not cover (they only
assert ``account_id``-scoped read visibility of child rows, never
calling ``resolve_target_session`` / ``handle_inbound`` against the
post-reparent state — exactly the path that previously 404-dropped).
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, patch

import asyncpg
import pytest

from aios.crypto.vault import CryptoBox
from aios.db import queries
from aios.db.pool import create_pool
from aios.models.inbound_policy import AllowAll
from aios.services import connections as connections_service
from aios.services.inbound import InboundDrop, handle_inbound
from aios_connectors.resolver import ResolveDrop, resolve_target_session
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration


@pytest.fixture
async def pool_two_accounts(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], CryptoBox]]:
    """Root + two-child tenant layout (``acc_root`` / ``acc_a`` / ``acc_b``)
    with a fresh ``CryptoBox``, shared by the reparent-then-resolve cases
    below. Mirrors the fixture in ``test_reparent_unique_index.py``: a pool
    (not a single conn) because ``reparent_connection`` and
    ``handle_inbound`` both take a pool."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    crypto_box = CryptoBox(os.urandom(32))
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_root', NULL,      TRUE,  'tenant-root'),
                       ('acc_a',    'acc_root', FALSE, 'tenant-a'),
                       ('acc_b',    'acc_root', FALSE, 'tenant-b')
                """
            )
        yield pool, crypto_box
    finally:
        await pool.close()


async def _reparent_to_b(
    pool: asyncpg.Pool[Any], connection_id: str, crypto_box: CryptoBox
) -> None:
    """Move ``connection_id`` from acc_a to acc_b as the root operator."""
    await connections_service.reparent_connection(
        pool,
        connection_id,
        destination_account_id="acc_b",
        requester_account_id="acc_root",
        crypto_box=crypto_box,
    )


class TestResolverDetachesCrossAccountSessionAfterReparent:
    """After reparent, every carried child row's ``session_id`` /
    ``binding.session_id`` points at a SOURCE-account session while the
    resolver runs under the DESTINATION scope. The resolver must surface
    ``DETACHED`` for each tier instead of handing the stale id to
    ``append_event`` (which would find zero rows → ``SESSION_MISSING``)."""

    async def test_tier1_ledger_cross_account_session_is_detached(
        self, pool_two_accounts: tuple[asyncpg.Pool[Any], CryptoBox]
    ) -> None:
        """Tier-1: ``lookup_chat_session`` under acc_b returns the stale
        acc_a ``session_id`` (the ledger row was carried to acc_b).
        Pre-fix ``_session_is_archived`` 0-rows under acc_b → ``False`` →
        resolver returned the stale id with ``drop=None``; ``handle_inbound``
        then 404-dropped at ``append_event``. Post-fix the 0-row is treated
        as not-live → ``DETACHED``."""
        pool, crypto_box = pool_two_accounts
        _agent, _env, session = await seed_agent_env_session(
            pool, account_id="acc_a", prefix="xacc-ledger"
        )
        connection = await connections_service.create_connection(
            pool,
            account_id="acc_a",
            connector="signal",
            external_account_id="+15551001",
            metadata={},
            crypto_box=crypto_box,
        )
        chat_id = "chat-pre-reparent"
        async with pool.acquire() as conn:
            await queries.insert_chat_session(
                conn,
                connection_id=connection.id,
                chat_id=chat_id,
                session_id=session.id,
                account_id="acc_a",
            )
        await _reparent_to_b(pool, connection.id, crypto_box)

        async with pool.acquire() as conn:
            on_dest = await queries.lookup_chat_session(
                conn, connection.id, chat_id, account_id="acc_b"
            )
            moved_conn = await queries.get_connection(conn, connection.id, account_id="acc_b")
        # The carry-over CTE DOES move the ledger row's account_id …
        assert on_dest == session.id, "ledger row must be visible under destination scope"
        # … but the session it points at is still owned by acc_a (the
        # cross-account state that is the bug's root cause).
        async with pool.acquire() as conn:
            owner = await conn.fetchval("SELECT account_id FROM sessions WHERE id = $1", session.id)
        assert owner == "acc_a", "session must still be owned by the source account"

        result = await resolve_target_session(
            pool, connection=moved_conn, chat_id=chat_id, account_id="acc_b"
        )
        assert result.drop is ResolveDrop.DETACHED, (
            f"tier-1 must DETACH a cross-account session_id, not hand it to "
            f"append_event. Got {result!r}."
        )
        assert result.session_id is None

    async def test_tier3_single_session_cross_account_session_is_detached(
        self, pool_two_accounts: tuple[asyncpg.Pool[Any], CryptoBox]
    ) -> None:
        """Tier-3 ``single_session``: the carried binding's ``session_id``
        points at the acc_a session; under the acc_b scope
        ``_session_is_archived`` 0-rows → must ``DETACHED`` (not return the
        stale id with ``drop=None``). A chat_id with no ledger row and no
        routing rules reaches tier-3 directly."""
        pool, crypto_box = pool_two_accounts
        _agent, _env, session = await seed_agent_env_session(
            pool, account_id="acc_a", prefix="xacc-single"
        )
        connection = await connections_service.create_connection(
            pool,
            account_id="acc_a",
            connector="signal",
            external_account_id="+15551002",
            metadata={},
            crypto_box=crypto_box,
        )
        await connections_service.attach_connection(
            pool, connection.id, account_id="acc_a", session_id=session.id
        )
        await _reparent_to_b(pool, connection.id, crypto_box)

        async with pool.acquire() as conn:
            binding = await queries.get_active_binding(conn, connection.id, account_id="acc_b")
            moved_conn = await queries.get_connection(conn, connection.id, account_id="acc_b")
        assert binding is not None and binding.session_id == session.id, (
            "binding must be visible under acc_b, carrying the stale acc_a session_id"
        )

        result = await resolve_target_session(
            pool,
            connection=moved_conn,
            chat_id="chat-never-seen-before",
            account_id="acc_b",
        )
        assert result.drop is ResolveDrop.DETACHED, (
            f"tier-3 single_session must DETACH a cross-account session_id. Got {result!r}."
        )
        assert result.session_id is None

    async def test_handle_inbound_returns_detached_not_session_missing(
        self, pool_two_accounts: tuple[asyncpg.Pool[Any], CryptoBox]
    ) -> None:
        """End-to-end: an inbound for a previously-routed chat under the
        destination scope must drop as ``DETACHED``, not
        ``SESSION_MISSING``. Pre-fix the cross-account ledger id reached
        ``append_event`` → ``NotFoundError`` → ``SESSION_MISSING``; the
        connector-http runner treats both 404 and 422 as non-fatal
        (drops/acks), but ``session_missing`` does not point at reparent
        as the cause while ``detached`` does."""
        pool, crypto_box = pool_two_accounts
        _agent, _env, session = await seed_agent_env_session(
            pool, account_id="acc_a", prefix="xacc-inbound"
        )
        connection = await connections_service.create_connection(
            pool,
            account_id="acc_a",
            connector="signal",
            external_account_id="+15551003",
            metadata={},
            crypto_box=crypto_box,
        )
        chat_id = "chat-inbound"
        async with pool.acquire() as conn:
            await queries.insert_chat_session(
                conn,
                connection_id=connection.id,
                chat_id=chat_id,
                session_id=session.id,
                account_id="acc_a",
            )
        # Open the admission gate (fail-closed DenyAll by default) so the
        # inbound reaches the resolver. The policy jsonb rides the
        # connection row, so it carries across the reparent to acc_b.
        await connections_service.set_inbound_policy(
            pool, connection.id, account_id="acc_a", policy=AllowAll()
        )
        await _reparent_to_b(pool, connection.id, crypto_box)

        with patch("aios.services.inbound.defer_wake", AsyncMock()):
            result = await handle_inbound(
                pool,
                account_id="acc_b",
                connection_id=connection.id,
                event_id="evt-cross-account-1",
                chat_id=chat_id,
                sender={"id": "sender-1", "display_name": "Sender One"},
                content="hello post-reparent",
            )
        assert result.drop_reason is InboundDrop.DETACHED, (
            f"post-reparent inbound for a previously-routed chat must drop as "
            f"DETACHED (→ 422), not SESSION_MISSING (which mislabels the cause; "
            f"pre-fix required append_event to find a zero-row session first). "
            f"Got {result!r}."
        )
        assert result.session_id is None
        assert not result.deduped

    async def test_live_same_account_session_still_routes(
        self, pool_two_accounts: tuple[asyncpg.Pool[Any], CryptoBox]
    ) -> None:
        """Control / regression guard: a LIVE session owned by the SAME
        account as the resolver scope must still route normally
        (``drop=None``, the session_id returned). Pins that the
        ``_session_is_archived`` change (0-row → ``True``) did not flip
        the existing-row-non-archived case — the ``or`` short-circuits to
        ``row["archived_at"] is not None`` which is ``False`` for a live
        session. This is the happy path that must not regress, and is the
        inverse of the cross-account (0-row) and archived (row, archived_at
        set) cases the fix targets."""
        pool, crypto_box = pool_two_accounts
        _agent, _env, session = await seed_agent_env_session(
            pool, account_id="acc_b", prefix="xacc-live"
        )
        connection = await connections_service.create_connection(
            pool,
            account_id="acc_b",
            connector="signal",
            external_account_id="+15551004",
            metadata={},
            crypto_box=crypto_box,
        )
        await connections_service.attach_connection(
            pool, connection.id, account_id="acc_b", session_id=session.id
        )
        async with pool.acquire() as conn:
            moved_conn = await queries.get_connection(conn, connection.id, account_id="acc_b")
        result = await resolve_target_session(
            pool, connection=moved_conn, chat_id="chat-live", account_id="acc_b"
        )
        assert result.drop is None, (
            f"a live same-account session must route (drop=None), not DETACH. Got {result!r}."
        )
        assert result.session_id == session.id
