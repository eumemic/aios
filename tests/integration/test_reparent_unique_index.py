"""Integration test: the per-account partial-unique index on
``connections`` enforces the reparent semantics end-to-end.

Migration 0060 relaxed the active-row uniqueness on connections from
globally exclusive ``(connector, external_account_id) WHERE archived_at
IS NULL`` to per-account ``(account_id, connector, external_account_id)
WHERE archived_at IS NULL``. The relaxation is what makes the reparent
primitive (:func:`aios.services.connections.reparent_connection`)
expressible — under the old global UNIQUE every reparent transit would
briefly violate it (source and hypothetical destination both holding
the active identity).

This file pins the post-migration contract against a real Postgres:

* Two accounts may hold the same active ``(connector, external_account_id)``
  triple — what was a cross-tenant collision pre-0060 is now legal.
* A single account still cannot — the new index keeps the
  same-account invariant intact.
* The reparent query moves ``account_id`` in place: ``connection.id``
  is preserved, ``updated_at`` advances.
* A reparent into a destination that already has an active connection
  for the same ``(connector, external_account_id)`` raises
  :class:`ConflictError`, mapped from the new index's
  :class:`asyncpg.UniqueViolationError`.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.crypto.vault import CryptoBox
from aios.db import queries
from aios.db.pool import create_pool
from aios.errors import ConflictError, NotFoundError
from aios.services import connections as connections_service

pytestmark = pytest.mark.integration


@pytest.fixture
async def pool_two_accounts(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], CryptoBox]]:
    """Asyncpg pool with the same root + two-child layout as
    ``conn_two_accounts``, plus a fresh ``CryptoBox`` for the secrets
    re-encryption path. Pool (not a single conn) because the secrets
    roundtrip exercises ``service.reparent_connection`` →
    ``service.get_connection_secrets``, both of which take a pool.
    """
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


class TestReparentUniqueIndex:
    async def test_two_accounts_may_hold_same_triple(
        self, conn_two_accounts: asyncpg.Connection[Any]
    ) -> None:
        """Post-0060 the cross-tenant collision is no longer a collision."""
        a_row = await queries.insert_connection(
            conn_two_accounts,
            account_id="acc_a",
            connector="signal",
            external_account_id="+15550001",
            metadata={},
        )
        b_row = await queries.insert_connection(
            conn_two_accounts,
            account_id="acc_b",
            connector="signal",
            external_account_id="+15550001",
            metadata={},
        )
        assert a_row.id != b_row.id
        # Both rows live side-by-side, both unarchived.
        assert a_row.archived_at is None
        assert b_row.archived_at is None

    async def test_same_account_repeat_returns_existing(
        self, conn_two_accounts: asyncpg.Connection[Any]
    ) -> None:
        """A second insert in the same account on the same triple is
        idempotent — the partial-unique still binds per-account."""
        first = await queries.insert_connection(
            conn_two_accounts,
            account_id="acc_a",
            connector="signal",
            external_account_id="+15550001",
            metadata={},
        )
        second = await queries.insert_connection(
            conn_two_accounts,
            account_id="acc_a",
            connector="signal",
            external_account_id="+15550001",
            metadata={},
        )
        assert second.id == first.id

    async def test_reparent_moves_account_id_preserves_id_bumps_updated_at(
        self, conn_two_accounts: asyncpg.Connection[Any]
    ) -> None:
        """Happy path: source row's ``account_id`` flips to the destination,
        ``id`` is preserved, ``updated_at`` advances."""
        source = await queries.insert_connection(
            conn_two_accounts,
            account_id="acc_a",
            connector="signal",
            external_account_id="+15550001",
            metadata={},
        )
        # ``updated_at`` is set by ``now()``. Sleep a moment so the
        # post-reparent timestamp is observably later than the
        # pre-reparent one — both are taken from the DB clock, but
        # ``now()`` within a single transaction is constant, so we just
        # need the two transactions to be separated by enough that
        # truncation to microseconds doesn't collapse them.
        await asyncio.sleep(0.01)
        moved = await queries.reparent_connection(
            conn_two_accounts,
            source.id,
            destination_account_id="acc_b",
        )
        assert moved.id == source.id
        assert moved.updated_at > source.updated_at

        # Confirm via a fresh read that ``account_id`` actually moved.
        row = await conn_two_accounts.fetchrow(
            "SELECT account_id FROM connections WHERE id = $1",
            source.id,
        )
        assert row is not None
        assert row["account_id"] == "acc_b"

    async def test_reparent_collision_at_destination_raises_conflict(
        self, conn_two_accounts: asyncpg.Connection[Any]
    ) -> None:
        """When the destination already holds an active connection on
        the same triple, the per-account UNIQUE rejects the reparent;
        the query layer surfaces it as :class:`ConflictError`."""
        source = await queries.insert_connection(
            conn_two_accounts,
            account_id="acc_a",
            connector="signal",
            external_account_id="+15550001",
            metadata={},
        )
        # Destination already owns the same identity.
        await queries.insert_connection(
            conn_two_accounts,
            account_id="acc_b",
            connector="signal",
            external_account_id="+15550001",
            metadata={},
        )
        with pytest.raises(ConflictError) as excinfo:
            await queries.reparent_connection(
                conn_two_accounts,
                source.id,
                destination_account_id="acc_b",
            )
        assert excinfo.value.detail == {"destination_account_id": "acc_b"}

        # Source row's ``account_id`` must stay on acc_a — the UNIQUE
        # violation aborts the UPDATE, so no partial state.
        row = await conn_two_accounts.fetchrow(
            "SELECT account_id FROM connections WHERE id = $1",
            source.id,
        )
        assert row is not None
        assert row["account_id"] == "acc_a"


class TestReparentSecretsRoundtrip:
    """End-to-end through the service layer: secrets configured on the
    source must decrypt correctly under the destination after reparent.

    Secrets are keyed to the owning ``account_id`` via
    :meth:`CryptoBox.derive_account_subkey`. Without the service-layer
    re-encryption inside the reparent transaction, the post-reparent
    ``get_connection_secrets`` call would attempt to decrypt the
    source-keyed ciphertext under the destination subkey and raise
    :class:`CryptoDecryptError`, silently bricking the connection.
    """

    async def test_secrets_decrypt_under_destination_after_reparent(
        self, pool_two_accounts: tuple[asyncpg.Pool[Any], CryptoBox]
    ) -> None:
        pool, crypto_box = pool_two_accounts
        # Create on acc_a, set secrets, reparent to acc_b, read back
        # from acc_b — the plaintext must round-trip intact.
        connection = await connections_service.create_connection(
            pool,
            account_id="acc_a",
            connector="telegram",
            external_account_id="bot:1234",
            metadata={},
            crypto_box=crypto_box,
        )
        original_secrets = {"bot_token": "tg-secret-XYZ987"}
        await connections_service.set_connection_secrets(
            pool,
            connection.id,
            account_id="acc_a",
            secrets=original_secrets,
            crypto_box=crypto_box,
        )
        await connections_service.reparent_connection(
            pool,
            connection.id,
            destination_account_id="acc_b",
            requester_account_id="acc_root",
            crypto_box=crypto_box,
        )
        recovered = await connections_service.get_connection_secrets(
            pool,
            connection.id,
            account_id="acc_b",
            crypto_box=crypto_box,
        )
        assert recovered == original_secrets

    async def test_no_secrets_reparent_is_a_noop_on_blob(
        self, pool_two_accounts: tuple[asyncpg.Pool[Any], CryptoBox]
    ) -> None:
        """A connection with no secrets reparents cleanly and still reads
        back as empty under the destination — guards against a regression
        that always tried to decrypt even when the columns are NULL."""
        pool, crypto_box = pool_two_accounts
        connection = await connections_service.create_connection(
            pool,
            account_id="acc_a",
            connector="telegram",
            external_account_id="bot:5678",
            metadata={},
            crypto_box=crypto_box,
        )
        await connections_service.reparent_connection(
            pool,
            connection.id,
            destination_account_id="acc_b",
            requester_account_id="acc_root",
            crypto_box=crypto_box,
        )
        recovered = await connections_service.get_connection_secrets(
            pool,
            connection.id,
            account_id="acc_b",
            crypto_box=crypto_box,
        )
        assert recovered == {}


class TestReparentArchivedRows:
    """Service-layer archived-row gates against a real Postgres.

    The unit tests cover these paths with mocked ``fetchrow`` /
    ``get_account``. These integration tests exercise the same paths
    against a real connection row (archived via
    :func:`queries.archive_connection`) and a real archived account
    row (via :func:`queries.archive_account`) so the gates can't drift
    silently from the schema.
    """

    async def test_archived_source_connection_reparent_raises_conflict(
        self, pool_two_accounts: tuple[asyncpg.Pool[Any], CryptoBox]
    ) -> None:
        """An archived connection cannot be reparented.

        The service's SELECT FOR UPDATE reads ``archived_at`` directly
        from the row inside the reparent transaction; the unit test
        pins this with a mocked fetchrow, but only an integration test
        exercises the real row written by ``archive_connection``
        (which also scrubs the secrets columns).
        """
        pool, crypto_box = pool_two_accounts
        connection = await connections_service.create_connection(
            pool,
            account_id="acc_a",
            connector="telegram",
            external_account_id="bot:9999",
            metadata={},
            crypto_box=crypto_box,
        )
        async with pool.acquire() as conn:
            await queries.archive_connection(conn, connection.id, account_id="acc_a")
        with pytest.raises(ConflictError) as excinfo:
            await connections_service.reparent_connection(
                pool,
                connection.id,
                destination_account_id="acc_b",
                requester_account_id="acc_root",
                crypto_box=crypto_box,
            )
        assert excinfo.value.detail == {"id": connection.id}

    async def test_archived_destination_account_reparent_raises_not_found(
        self, pool_two_accounts: tuple[asyncpg.Pool[Any], CryptoBox]
    ) -> None:
        """An archived destination account is not a valid reparent target.

        ``queries.get_account`` returns archived rows, so without the
        service's ``archived_at`` check an operator could DELETE an
        account (soft-archive) and then reparent a connection into it,
        permanently bricking the connection (no bearer can auth as an
        archived account).
        """
        pool, crypto_box = pool_two_accounts
        connection = await connections_service.create_connection(
            pool,
            account_id="acc_a",
            connector="telegram",
            external_account_id="bot:archived-dest",
            metadata={},
            crypto_box=crypto_box,
        )
        async with pool.acquire() as conn:
            archived = await queries.archive_account(conn, "acc_b")
            assert archived is not None and archived.archived_at is not None
        with pytest.raises(NotFoundError) as excinfo:
            await connections_service.reparent_connection(
                pool,
                connection.id,
                destination_account_id="acc_b",
                requester_account_id="acc_root",
                crypto_box=crypto_box,
            )
        assert excinfo.value.detail == {"destination_account_id": "acc_b"}
