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
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.errors import ConflictError

pytestmark = pytest.mark.integration


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
