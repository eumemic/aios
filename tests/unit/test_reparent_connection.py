"""Unit coverage for :func:`aios.services.connections.reparent_connection`.

Pins the service-layer business rules:

* Only the root operator account may reparent (caller's
  ``parent_account_id`` must be ``None``) — every other tenant gets a
  :class:`ForbiddenError` regardless of whether the destination is
  cross-tenant or in their own subtree.
* The destination must exist; missing destination → :class:`NotFoundError`.
* The connection must exist and be non-archived; missing →
  :class:`NotFoundError`, archived → :class:`ConflictError`.
* Happy path updates ``account_id`` via the underlying query and
  returns the projected :class:`Connection`.

Cross-tenant uniqueness collisions at the destination are checked at
the integration tier (real Postgres + the per-account UNIQUE), since
they're the database's job — surfacing the right error class from
``asyncpg.UniqueViolationError`` lives in
``aios.db.queries.reparent_connection``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import asyncpg
import pytest

from aios.errors import ConflictError, ForbiddenError, NotFoundError
from aios.models.accounts import Account
from aios.models.connections import Connection
from aios.services.connections import reparent_connection
from tests.unit.conftest import fake_pool_yielding_conn

_NOW = datetime.now(UTC)


def _root_account(account_id: str = "acc_root") -> Account:
    return Account(
        id=account_id,
        parent_account_id=None,
        can_mint_children=True,
        display_name="root",
        metadata={},
        created_at=_NOW,
        archived_at=None,
    )


def _child_account(account_id: str, parent: str = "acc_root") -> Account:
    return Account(
        id=account_id,
        parent_account_id=parent,
        can_mint_children=False,
        display_name=f"child-{account_id}",
        metadata={},
        created_at=_NOW,
        archived_at=None,
    )


def _connection(
    *,
    connection_id: str = "conn_x",
    account_id: str = "acc_src",
    archived_at: datetime | None = None,
) -> Connection:
    return Connection(
        id=connection_id,
        connector="signal",
        external_account_id="+15551234567",
        session_id=None,
        session_template_id=None,
        metadata={},
        secrets_set=False,
        created_at=_NOW,
        attached_at=None,
        updated_at=_NOW,
        archived_at=archived_at,
    )


def _conn_with_transaction() -> MagicMock:
    """Return a MagicMock conn whose ``conn.transaction()`` is a working async CM."""
    conn = MagicMock()
    txn_cm = MagicMock()
    txn_cm.__aenter__ = AsyncMock(return_value=None)
    txn_cm.__aexit__ = AsyncMock(return_value=None)
    conn.transaction.return_value = txn_cm
    return conn


class TestReparentConnection:
    async def test_reparent_rejects_non_root_caller(self) -> None:
        """A non-root caller (``parent_account_id`` set) is forbidden."""
        conn = _conn_with_transaction()
        pool = cast("asyncpg.Pool[Any]", fake_pool_yielding_conn(conn))

        with (
            patch(
                "aios.services.connections.queries.get_account",
                AsyncMock(return_value=_child_account("acc_caller")),
            ),
            pytest.raises(ForbiddenError),
        ):
            await reparent_connection(
                pool,
                "conn_x",
                destination_account_id="acc_dest",
                requester_account_id="acc_caller",
            )

    async def test_reparent_rejects_unknown_caller(self) -> None:
        """A caller that doesn't resolve to any account row is forbidden.

        Defensive: the auth dep should never hand us an unknown
        ``account_id``, but treating it as forbidden rather than
        crashing keeps the failure mode predictable.
        """
        conn = _conn_with_transaction()
        pool = cast("asyncpg.Pool[Any]", fake_pool_yielding_conn(conn))

        with (
            patch(
                "aios.services.connections.queries.get_account",
                AsyncMock(return_value=None),
            ),
            pytest.raises(ForbiddenError),
        ):
            await reparent_connection(
                pool,
                "conn_x",
                destination_account_id="acc_dest",
                requester_account_id="acc_ghost",
            )

    async def test_reparent_missing_destination_account_raises_not_found(self) -> None:
        """Destination lookup returning ``None`` → :class:`NotFoundError`."""
        conn = _conn_with_transaction()
        pool = cast("asyncpg.Pool[Any]", fake_pool_yielding_conn(conn))

        # First call (requester) is root; second call (destination) misses.
        with (
            patch(
                "aios.services.connections.queries.get_account",
                AsyncMock(side_effect=[_root_account("acc_root"), None]),
            ),
            pytest.raises(NotFoundError),
        ):
            await reparent_connection(
                pool,
                "conn_x",
                destination_account_id="acc_dest_missing",
                requester_account_id="acc_root",
            )

    async def test_reparent_missing_connection_raises_not_found(self) -> None:
        """The pre-update SELECT FOR UPDATE returning no row → :class:`NotFoundError`."""
        conn = _conn_with_transaction()
        pool = cast("asyncpg.Pool[Any]", fake_pool_yielding_conn(conn))
        # The pre-update SELECT FOR UPDATE returns no row.
        conn.fetchrow = AsyncMock(return_value=None)

        with (
            patch(
                "aios.services.connections.queries.get_account",
                AsyncMock(side_effect=[_root_account("acc_root"), _child_account("acc_dest")]),
            ),
            pytest.raises(NotFoundError),
        ):
            await reparent_connection(
                pool,
                "conn_missing",
                destination_account_id="acc_dest",
                requester_account_id="acc_root",
            )

    async def test_reparent_archived_connection_raises_conflict(self) -> None:
        """SELECT FOR UPDATE finds the row but it's archived → :class:`ConflictError`."""
        conn = _conn_with_transaction()
        pool = cast("asyncpg.Pool[Any]", fake_pool_yielding_conn(conn))
        # SELECT FOR UPDATE returns a row whose archived_at is set.
        conn.fetchrow = AsyncMock(return_value={"archived_at": _NOW})

        with (
            patch(
                "aios.services.connections.queries.get_account",
                AsyncMock(side_effect=[_root_account("acc_root"), _child_account("acc_dest")]),
            ),
            pytest.raises(ConflictError),
        ):
            await reparent_connection(
                pool,
                "conn_archived",
                destination_account_id="acc_dest",
                requester_account_id="acc_root",
            )

    async def test_reparent_happy_path_updates_account_id(self) -> None:
        """Calls the query with the destination account id and returns its result."""
        conn = _conn_with_transaction()
        pool = cast("asyncpg.Pool[Any]", fake_pool_yielding_conn(conn))
        # SELECT FOR UPDATE returns a live (non-archived) row.
        conn.fetchrow = AsyncMock(return_value={"archived_at": None})

        updated = _connection(connection_id="conn_x", account_id="acc_dest")
        with (
            patch(
                "aios.services.connections.queries.get_account",
                AsyncMock(side_effect=[_root_account("acc_root"), _child_account("acc_dest")]),
            ),
            patch(
                "aios.services.connections.queries.reparent_connection",
                AsyncMock(return_value=updated),
            ) as reparent_query,
        ):
            result = await reparent_connection(
                pool,
                "conn_x",
                destination_account_id="acc_dest",
                requester_account_id="acc_root",
            )

        assert result is updated
        reparent_query.assert_awaited_once()
        kwargs = reparent_query.await_args.kwargs
        assert kwargs["destination_account_id"] == "acc_dest"
        assert kwargs["connection_id"] == "conn_x"
