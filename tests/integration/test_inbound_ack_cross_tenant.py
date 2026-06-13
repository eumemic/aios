"""Integration test: the ``connector_inbound_acks`` dedup ledger must be
tenant-scoped.

The ledger's primary key predates multi-tenancy. While migration 0050's
global active-unique made ``(connector, external_account_id)`` universally
exclusive, the 3-tuple PK ``(connector, external_account_id, event_id)``
was tenant-unique *by construction* — one external identity belonged to at
most one account. Migration 0060 relaxed connections uniqueness to
``(account_id, connector, external_account_id)`` to support the #694
reparent primitive, so two accounts can now independently hold the same
external identity. And ``event_id`` is a deterministic function of the chat
namespace (the Telegram connector emits ``telegram-{chat_id}-{message_id}``),
not a random global ULID.

With ``account_id`` absent from the ledger key, account A's ack for an
event_id silently swallows account B's *first-ever* delivery of the same
event_id: ``try_record_inbound_ack`` returns ``False``, ``handle_inbound``
reports ``deduped=True``, and B's genuinely-new user message is never
appended. This file pins the post-fix contract — independent dedup
keyspaces per tenant, with the within-tenant dedup still intact.
"""

from __future__ import annotations

from typing import Any

import asyncpg
import pytest

from aios.db import queries

pytestmark = pytest.mark.integration


class TestInboundAckCrossTenant:
    async def test_same_identity_event_id_records_per_tenant(
        self, conn_two_accounts: asyncpg.Connection[Any]
    ) -> None:
        """Two tenants sharing an external identity keep independent dedup
        keyspaces: each account's first delivery of a given event_id records,
        regardless of the other account's ledger."""
        first = await queries.try_record_inbound_ack(
            conn_two_accounts,
            account_id="acc_a",
            connector="telegram",
            external_account_id="bot1",
            event_id="telegram-chat1-100",
            appended_seq=1,
        )
        assert first is True

        second = await queries.try_record_inbound_ack(
            conn_two_accounts,
            account_id="acc_b",
            connector="telegram",
            external_account_id="bot1",
            event_id="telegram-chat1-100",
            appended_seq=1,
        )
        assert second is True, (
            "a different tenant's never-before-seen inbound must record its own "
            "ack, not collide with another account's ledger row"
        )

    async def test_same_tenant_same_event_id_still_dedups(
        self, conn_two_accounts: asyncpg.Connection[Any]
    ) -> None:
        """The dedup contract within a single tenant is preserved: a true
        duplicate (same account + identity + event_id, e.g. a connector
        re-emitting after a crash-before-ack) still hits ON CONFLICT and
        returns ``False`` so no second event is appended."""
        first = await queries.try_record_inbound_ack(
            conn_two_accounts,
            account_id="acc_a",
            connector="telegram",
            external_account_id="bot1",
            event_id="telegram-chat1-200",
            appended_seq=1,
        )
        assert first is True

        duplicate = await queries.try_record_inbound_ack(
            conn_two_accounts,
            account_id="acc_a",
            connector="telegram",
            external_account_id="bot1",
            event_id="telegram-chat1-200",
            appended_seq=2,
        )
        assert duplicate is False
