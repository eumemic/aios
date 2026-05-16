"""Integration test: ``insert_connection`` surfaces cross-tenant collisions as ``ConflictError``.

The ``connections`` table carries a globally-unique partial index on
``(connector, external_account_id) WHERE archived_at IS NULL``. Real
messaging identities (Signal phone numbers, Telegram bot tokens, Matrix
accounts, etc.) are universally exclusive, so the constraint mirrors
physical reality — and a cross-tenant collision on the active row is a
caller-visible error, not contention to swallow.

Pins the contract: a tenant inserting on another tenant's active
identity raises :class:`aios.errors.ConflictError` carrying
``connector`` + ``external_account_id`` in ``detail``. No silent
retry, no existence-leak via failure shape.
"""

from __future__ import annotations

from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.errors import ConflictError

pytestmark = pytest.mark.integration


class TestInsertConnectionCrossTenant:
    async def test_cross_tenant_collision_raises_conflict_error(
        self, conn_two_accounts: asyncpg.Connection[Any]
    ) -> None:
        """Tenant B inserting on tenant A's active identity → ``ConflictError``."""
        await queries.insert_connection(
            conn_two_accounts,
            account_id="acc_a",
            connector="signal",
            external_account_id="+15550001",
            metadata={},
        )
        with pytest.raises(ConflictError) as excinfo:
            await queries.insert_connection(
                conn_two_accounts,
                account_id="acc_b",
                connector="signal",
                external_account_id="+15550001",
                metadata={},
            )
        assert excinfo.value.detail == {
            "connector": "signal",
            "external_account_id": "+15550001",
        }

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

    async def test_archived_other_tenant_lets_new_tenant_take_identity(
        self, conn_two_accounts: asyncpg.Connection[Any]
    ) -> None:
        """When A's row is archived, B may claim the same external identity.

        The partial index excludes archived rows, so the constraint
        doesn't fire — and B is the sole tenant on the active row.
        """
        a_first = await queries.insert_connection(
            conn_two_accounts,
            account_id="acc_a",
            connector="signal",
            external_account_id="+15550001",
            metadata={},
        )
        await queries.archive_connection(conn_two_accounts, a_first.id, account_id="acc_a")
        b_takes = await queries.insert_connection(
            conn_two_accounts,
            account_id="acc_b",
            connector="signal",
            external_account_id="+15550001",
            metadata={},
        )
        assert b_takes.id != a_first.id
