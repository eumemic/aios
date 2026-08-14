"""The audited ledger — not the operator-writable policy list — decides admission.

#1503 requires an inbound approval to be **server-stamped and audited**. The
first cut of ``RequireApproval`` admitted on ``policy.approved`` alone and never
consulted an ``inbound_grants`` row, so two holes were open:

* ``PUT /v1/connections/{id}/inbound-policy`` could write
  ``{"kind": "require_approval", "approved": ["alice"]}`` and admit ``alice``
  with **no grant row and no approval provenance** — an authority control
  bypassable through the operator door, which is the door that matters.
* a **dangling** ``approved`` entry stayed admitted after its grant was
  revoked / reaped, because revocation's policy-list rewrite is a second write
  that a crash, a concurrent Replace, or a hand-rolled policy PUT can skip.

These tests drive the REAL path — ``handle_inbound`` against a real database —
not the pure ``_admits`` predicate and not ``upsert_pending_inbound_grant``
directly. The negative case pins "approved-without-grant does not admit"; the
positive control pins "granted + approved still delivers" so the negative case
cannot pass on a build that admits nothing; and the pending-upsert assertions
are reached only through production code, so deleting the production
pending-upsert branch turns them red.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, patch

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.models.inbound_policy import RequireApproval
from aios.services import connections as connections_service
from aios.services.inbound import InboundDrop, handle_inbound
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration

_ACCOUNT = "acc_admission_ledger"


@pytest.fixture
async def bound_connection(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str]]:
    """``(pool, connection_id, session_id)`` — a single_session binding whose
    connection carries ``RequireApproval`` with an EMPTY approved list."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                   VALUES ($1, NULL, TRUE, 'admission-ledger-test')""",
                _ACCOUNT,
            )
        _agent, _env, session = await seed_agent_env_session(
            pool, account_id=_ACCOUNT, prefix="admission-ledger"
        )
        async with pool.acquire() as conn:
            connection = await queries.insert_connection(
                conn,
                account_id=_ACCOUNT,
                connector="echo",
                external_account_id="admission-ledger",
                metadata={},
            )
        await connections_service.attach_connection(
            pool, connection.id, account_id=_ACCOUNT, session_id=session.id
        )
        await connections_service.set_inbound_policy(
            pool, connection.id, account_id=_ACCOUNT, policy=RequireApproval(approved=[])
        )
        yield pool, connection.id, session.id
    finally:
        await pool.close()


async def _inbound(
    pool: asyncpg.Pool[Any], connection_id: str, *, chat_id: str, event_id: str
) -> Any:
    """Drive the real ``handle_inbound``, stubbing only the job-queue wake."""
    with patch("aios.services.inbound.defer_wake", AsyncMock()) as wake:
        result = await handle_inbound(
            pool,
            account_id=_ACCOUNT,
            connection_id=connection_id,
            event_id=event_id,
            chat_id=chat_id,
            sender={"id": chat_id, "display_name": chat_id},
            content="hello",
        )
    return result, wake


async def _event_count(pool: asyncpg.Pool[Any], session_id: str) -> int:
    async with pool.acquire() as conn:
        return int(
            await conn.fetchval("SELECT count(*) FROM events WHERE session_id = $1", session_id)
        )


async def _grant_rows(pool: asyncpg.Pool[Any], connection_id: str, chat_id: str) -> list[Any]:
    async with pool.acquire() as conn:
        return list(
            await conn.fetch(
                """SELECT status, approved_by, approved_via_channel FROM inbound_grants
                    WHERE connection_id = $1 AND chat_id = $2 ORDER BY created_at""",
                connection_id,
                chat_id,
            )
        )


# ─── HIGH 1 (negative): the operator-writable list alone must NOT admit ──────


async def test_approved_entry_without_active_grant_does_not_admit(
    bound_connection: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """``set_inbound_policy`` writing ``approved=["alice"]`` with NO ledger row
    must not admit alice: the audited grant is authoritative, the policy list is
    a mirror. Pre-fix this admits and appends an event."""
    pool, connection_id, session_id = bound_connection
    await connections_service.set_inbound_policy(
        pool, connection_id, account_id=_ACCOUNT, policy=RequireApproval(approved=["alice"])
    )
    assert await _grant_rows(pool, connection_id, "alice") == [], (
        "precondition: the operator policy endpoint writes NO grant row"
    )

    result, wake = await _inbound(pool, connection_id, chat_id="alice", event_id="evt-dangling")

    assert result.drop_reason is not None, (
        "an approved entry with no active grant must NOT admit — admission has to "
        "consult the audited inbound_grants ledger, not just policy.approved"
    )
    assert result.appended_event_id is None
    assert await _event_count(pool, session_id) == 0, "no event may be appended"
    wake.assert_not_awaited()


async def test_revoked_grant_leaves_dangling_approved_entry_inert(
    bound_connection: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """A stale ``approved`` entry whose grant is revoked must not admit even if
    the policy-list rewrite never happened (crash / concurrent Replace)."""
    pool, connection_id, session_id = bound_connection
    async with pool.acquire() as conn:
        await queries.upsert_pending_inbound_grant(
            conn, account_id=_ACCOUNT, connection_id=connection_id, chat_id="bob"
        )
    await connections_service.approve_inbound_grant(pool, connection_id, "bob", account_id=_ACCOUNT)
    await connections_service.revoke_inbound_grant(pool, connection_id, "bob", account_id=_ACCOUNT)
    # Simulate the dangling entry: policy still lists bob, ledger says revoked.
    await connections_service.set_inbound_policy(
        pool, connection_id, account_id=_ACCOUNT, policy=RequireApproval(approved=["bob"])
    )

    result, wake = await _inbound(pool, connection_id, chat_id="bob", event_id="evt-revoked")

    assert result.drop_reason is not None, "a revoked grant must not stay admitted"
    assert await _event_count(pool, session_id) == 0
    wake.assert_not_awaited()


# ─── POSITIVE CONTROL: granted + approved still delivers ─────────────────────


async def test_granted_and_approved_chat_still_delivers(
    bound_connection: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """The whole audited round trip through production code: a stranger's first
    message is refused AND registers a pending grant (this assertion is reached
    only via the production pending-upsert branch — deleting it turns this red),
    the operator approves, and the next message is DELIVERED.

    Without this control, "we no longer admit without a grant" would pass on a
    build that admits nothing at all.
    """
    pool, connection_id, session_id = bound_connection

    # 1. Stranger's first message: refused, and the production path registers
    #    the pending grant that makes approval possible.
    first, first_wake = await _inbound(pool, connection_id, chat_id="carol", event_id="evt-carol-1")
    assert first.drop_reason is not None
    assert await _event_count(pool, session_id) == 0
    first_wake.assert_not_awaited()
    pending = await _grant_rows(pool, connection_id, "carol")
    assert [row["status"] for row in pending] == ["pending"], (
        "handle_inbound must register a PENDING grant for an unapproved sender "
        "under RequireApproval — this is the production pending-upsert branch"
    )

    # 2. Operator approves through the audited endpoint (server-stamped).
    grant = await connections_service.approve_inbound_grant(
        pool, connection_id, "carol", account_id=_ACCOUNT
    )
    assert grant.status == "active"
    assert grant.approved_by == _ACCOUNT
    assert grant.approved_via_channel == "operator_api"

    # 3. The same sender now DELIVERS.
    second, second_wake = await _inbound(
        pool, connection_id, chat_id="carol", event_id="evt-carol-2"
    )
    assert second.drop_reason is None, (
        f"a properly granted + approved chat must still deliver; got {second!r}"
    )
    assert second.appended_event_id == "evt-carol-2"
    assert second.session_id == session_id
    assert await _event_count(pool, session_id) == 1
    second_wake.assert_awaited_once()


async def test_pending_grant_alone_does_not_admit(
    bound_connection: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """A pending (un-approved) grant is not an admission — only 'active' is."""
    pool, connection_id, session_id = bound_connection
    async with pool.acquire() as conn:
        await queries.upsert_pending_inbound_grant(
            conn, account_id=_ACCOUNT, connection_id=connection_id, chat_id="dave"
        )
    # Policy list says yes; the ledger says merely 'pending'.
    await connections_service.set_inbound_policy(
        pool, connection_id, account_id=_ACCOUNT, policy=RequireApproval(approved=["dave"])
    )

    result, wake = await _inbound(pool, connection_id, chat_id="dave", event_id="evt-dave")

    assert result.drop_reason is not None, "a pending grant is not an approval"
    assert await _event_count(pool, session_id) == 0
    wake.assert_not_awaited()


async def test_refusal_is_distinguishable_from_delivery(
    bound_connection: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """A refused inbound carries a machine-readable reason a caller can act on:
    an unapproved sender under RequireApproval is PENDING_APPROVAL, distinct
    from a flat DENIED_BY_POLICY, and distinct from a delivery."""
    pool, connection_id, _session_id = bound_connection

    refused, _ = await _inbound(pool, connection_id, chat_id="erin", event_id="evt-erin")

    assert refused.drop_reason is InboundDrop.PENDING_APPROVAL
    assert refused.appended_event_id is None
