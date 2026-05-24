"""Integration test: ``insert_connection`` semantics under the per-account unique index.

Pre-#694 (migration 0060) the active-row partial unique on connections
was ``(connector, external_account_id) WHERE archived_at IS NULL`` —
globally exclusive across tenants. The jarbot v2 transfer primitive
(:func:`aios.services.connections.reparent_connection`) cannot move a
connection between accounts under that constraint without a brief
violation in transit, so 0060 relaxes it to
``(account_id, connector, external_account_id) WHERE archived_at IS
NULL``.

This file pins the post-migration contract: two accounts may
simultaneously hold the same external identity, but a single account
still can't double-bind. The cross-tenant uniqueness that the global
index used to enforce is now the job of the connector daemon (one
process per identity) and the operator (don't claim someone else's
phone number) — the database no longer encodes universal exclusivity.
"""

from __future__ import annotations

from typing import Any

import asyncpg
import pytest

from aios.db import queries

pytestmark = pytest.mark.integration


class TestInsertConnectionCrossTenant:
    async def test_cross_tenant_same_identity_is_allowed(
        self, conn_two_accounts: asyncpg.Connection[Any]
    ) -> None:
        """Tenant B inserting on tenant A's active identity now succeeds.

        The per-account unique index lets two accounts hold the same
        external identity simultaneously — the global-exclusivity
        invariant was given up by migration 0060 (#694) to make the
        reparent primitive expressible.
        """
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

    async def test_same_tenant_repeat_returns_existing(
        self, conn_two_accounts: asyncpg.Connection[Any]
    ) -> None:
        """Idempotency on ``(connector, external_account_id)`` within one tenant."""
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

    async def test_archived_then_reinsert_same_tenant(
        self, conn_two_accounts: asyncpg.Connection[Any]
    ) -> None:
        """The partial unique index permits a fresh insert after archive."""
        first = await queries.insert_connection(
            conn_two_accounts,
            account_id="acc_a",
            connector="signal",
            external_account_id="+15550001",
            metadata={},
        )
        await queries.archive_connection(conn_two_accounts, first.id, account_id="acc_a")
        second = await queries.insert_connection(
            conn_two_accounts,
            account_id="acc_a",
            connector="signal",
            external_account_id="+15550001",
            metadata={},
        )
        assert second.id != first.id
        assert second.archived_at is None
