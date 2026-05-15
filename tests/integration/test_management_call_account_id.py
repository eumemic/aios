"""Integration tests: ``pending_management_calls`` queries enforce ``account_id``.

Migration 0041 created the table without an ``account_id`` column; the
multi-tenancy migrations (0043 rename, 0044 backfill+NOT NULL) added
``account_id`` to every other reserved resource table but missed this
one. ``get_management_call`` and ``mark_management_call_resolved``
were retrofitted with ``WHERE account_id = $N`` predicates regardless,
so every Signal connector RPC raises ``asyncpg.UndefinedColumnError``
at runtime — invisible to the unit suite (fully mocked) and to the
existing e2e signal test (doesn't exercise result intake).

These tests pin the round trip end-to-end against the real schema:
insert with one account id, read with the same one, read with a
different one returns ``None``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
import pytest

from aios.db import queries

pytestmark = pytest.mark.integration


@pytest.fixture
async def conn_two_accounts(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[asyncpg.Connection[Any]]:
    """Asyncpg conn with two seeded accounts (``acc_a``, ``acc_b``)."""
    conn = await asyncpg.connect(migrated_db_url)
    try:
        # One root + two children — partial unique index
        # ``accounts_one_active_root`` permits only a single non-archived
        # ``parent_account_id IS NULL`` row at a time.
        await conn.execute(
            """
            INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
            VALUES ('acc_root', NULL,      TRUE,  'tenant-root'),
                   ('acc_a',    'acc_root', FALSE, 'tenant-a'),
                   ('acc_b',    'acc_root', FALSE, 'tenant-b')
            """
        )
        yield conn
    finally:
        await conn.close()


class TestGetManagementCallTenancy:
    async def test_owner_can_fetch_their_pending_call(
        self, conn_two_accounts: asyncpg.Connection[Any]
    ) -> None:
        await queries.insert_management_call(
            conn_two_accounts,
            account_id="acc_a",
            call_id="mgmt_a_pending",
            connector="signal",
            method="register",
            params={"account": "+15551234567"},
            expires_at=datetime.now(UTC) + timedelta(seconds=60),
        )

        row = await queries.get_management_call(
            conn_two_accounts, "mgmt_a_pending", account_id="acc_a"
        )

        assert row is not None
        assert row["id"] == "mgmt_a_pending"
        assert row["connector"] == "signal"
        assert row["method"] == "register"
        assert row["status"] == "pending"

    async def test_other_tenant_cannot_fetch_call(
        self, conn_two_accounts: asyncpg.Connection[Any]
    ) -> None:
        await queries.insert_management_call(
            conn_two_accounts,
            account_id="acc_a",
            call_id="mgmt_a_pending",
            connector="signal",
            method="register",
            params={"account": "+15551234567"},
            expires_at=datetime.now(UTC) + timedelta(seconds=60),
        )

        row = await queries.get_management_call(
            conn_two_accounts, "mgmt_a_pending", account_id="acc_b"
        )

        assert row is None


class TestMarkManagementCallResolvedTenancy:
    async def test_owner_can_resolve_their_call(
        self, conn_two_accounts: asyncpg.Connection[Any]
    ) -> None:
        await queries.insert_management_call(
            conn_two_accounts,
            account_id="acc_a",
            call_id="mgmt_a_pending",
            connector="signal",
            method="register",
            params={"account": "+15551234567"},
            expires_at=datetime.now(UTC) + timedelta(seconds=60),
        )

        moved = await queries.mark_management_call_resolved(
            conn_two_accounts,
            account_id="acc_a",
            call_id="mgmt_a_pending",
            result={"status": "sms_sent"},
            is_error=False,
        )

        assert moved is True

    async def test_other_tenant_cannot_resolve_call(
        self, conn_two_accounts: asyncpg.Connection[Any]
    ) -> None:
        await queries.insert_management_call(
            conn_two_accounts,
            account_id="acc_a",
            call_id="mgmt_a_pending",
            connector="signal",
            method="register",
            params={"account": "+15551234567"},
            expires_at=datetime.now(UTC) + timedelta(seconds=60),
        )

        moved = await queries.mark_management_call_resolved(
            conn_two_accounts,
            account_id="acc_b",
            call_id="mgmt_a_pending",
            result={"status": "stolen"},
            is_error=False,
        )

        assert moved is False
