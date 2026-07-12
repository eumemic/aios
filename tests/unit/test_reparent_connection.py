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

import os
from datetime import UTC, datetime
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import asyncpg
import pytest

from aios.crypto.vault import CryptoBox
from aios.errors import ConflictError, ForbiddenError, NotFoundError
from aios.models.accounts import Account
from aios.models.connections import Connection
from aios.services.connections import reparent_connection
from tests.unit.conftest import fake_pool_yielding_conn

_NOW = datetime.now(UTC)
_CRYPTO_BOX = CryptoBox(os.urandom(32))


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
                crypto_box=_CRYPTO_BOX,
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
                crypto_box=_CRYPTO_BOX,
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
                crypto_box=_CRYPTO_BOX,
            )

    async def test_reparent_archived_destination_account_raises_not_found(self) -> None:
        """Archived destination → :class:`NotFoundError`.

        ``queries.get_account`` returns archived rows too, so a bare
        ``is None`` check would let an operator reparent into a
        soft-archived account. No bearer can auth as an archived
        account, so the connection would be permanently inaccessible.
        From the reparent caller's perspective an archived account
        is effectively non-existent; surface the same 404.
        """
        conn = _conn_with_transaction()
        pool = cast("asyncpg.Pool[Any]", fake_pool_yielding_conn(conn))

        archived_dest = Account(
            id="acc_dest_archived",
            parent_account_id="acc_root",
            can_mint_children=False,
            display_name="archived-dest",
            metadata={},
            created_at=_NOW,
            archived_at=_NOW,
        )
        # First call (requester) is root; second call (destination) is archived.
        with (
            patch(
                "aios.services.connections.queries.get_account",
                AsyncMock(side_effect=[_root_account("acc_root"), archived_dest]),
            ),
            pytest.raises(NotFoundError) as excinfo,
        ):
            await reparent_connection(
                pool,
                "conn_x",
                destination_account_id="acc_dest_archived",
                requester_account_id="acc_root",
                crypto_box=_CRYPTO_BOX,
            )
        assert excinfo.value.detail == {"destination_account_id": "acc_dest_archived"}

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
                crypto_box=_CRYPTO_BOX,
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
                crypto_box=_CRYPTO_BOX,
            )

    async def test_reparent_happy_path_updates_account_id(self) -> None:
        """Calls the query with the destination account id and returns its result.

        No secrets configured → ``secrets_blob`` arg is ``None`` (the
        UPDATE still rewrites both secret columns, both to NULL, which
        satisfies ``connections_secrets_pair_ck``).
        """
        conn = _conn_with_transaction()
        pool = cast("asyncpg.Pool[Any]", fake_pool_yielding_conn(conn))
        # SELECT FOR UPDATE returns a live (non-archived) row, with no
        # secrets configured — re-encryption path is skipped.
        conn.fetchrow = AsyncMock(
            return_value={
                "archived_at": None,
                "account_id": "acc_src",
                "secrets_ciphertext": None,
                "secrets_nonce": None,
            }
        )

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
                crypto_box=_CRYPTO_BOX,
            )

        assert result is updated
        reparent_query.assert_awaited_once()
        assert reparent_query.await_args is not None
        kwargs = reparent_query.await_args.kwargs
        assert kwargs["destination_account_id"] == "acc_dest"
        assert kwargs["connection_id"] == "conn_x"
        # No secrets configured → no re-key, blob is None.
        assert kwargs["secrets_blob"] is None

    async def test_reparent_query_sql_updates_account_scoped_children(self) -> None:
        """The query layer's SQL must rewrite ``account_id`` on every
        account-scoped child of the connection: ``bindings``,
        ``chat_sessions``, ``routing_rules``. Without this, the
        resolver tiers (every filter on ``account_id``) would silently
        DETACH-drop the destination's next inbound. Pin the SQL shape
        directly — an integration test covers the runtime effect, but
        only a SQL-shape pin catches a refactor that removes the CTEs.
        """
        from aios.db.queries import reparent_connection as query_reparent

        captured: dict[str, str] = {}

        async def fake_fetchrow(sql: str, *args: Any, **kwargs: Any) -> Any:
            captured["sql"] = sql
            # Return a row that ``_row_to_connection`` can parse — enough
            # fields for the projection.
            return {
                "id": "conn_x",
                "connector": "signal",
                "external_account_id": "+15551234",
                "binding_session_id": None,
                "binding_session_template_id": None,
                "binding_created_at": None,
                # The pool's jsonb codec decodes JSONB to native Python, so the
                # query layer sees a dict here (was the raw "{}" text pre-codec).
                "metadata": {},
                "secrets_ciphertext": None,
                "created_by_type": None,
                "created_by_ref": None,
                "created_at": _NOW,
                "updated_at": _NOW,
                "archived_at": None,
                # Rides ``c.*`` on every real read (migration 0121); NULL → the
                # connection has no explicit policy (resolves to DenyAll).
                "inbound_policy": None,
            }

        conn = MagicMock()
        conn.fetchrow = fake_fetchrow
        await query_reparent(
            conn,
            "conn_x",
            destination_account_id="acc_dest",
            secrets_blob=None,
        )
        sql = captured["sql"]
        assert "UPDATE connections" in sql
        assert "UPDATE bindings" in sql, (
            "bindings.account_id must be rewritten alongside connections.account_id "
            "or the resolver's account-scoped get_active_binding returns None "
            "under the destination scope and inbounds silently DETACH-drop"
        )
        assert "UPDATE chat_sessions" in sql, (
            "chat_sessions.account_id must be rewritten — lookup_chat_session "
            "filters on account_id and would otherwise miss"
        )
        assert "UPDATE routing_rules" in sql, (
            "routing_rules.account_id must be rewritten — "
            "list_routing_rules_for_connection joins bindings on account_id"
        )

    async def test_reparent_rekeys_secrets_when_present(self) -> None:
        """When the source row has encrypted secrets, the service decrypts
        with the source subkey and re-encrypts with the destination
        subkey before handing the blob to the query. Without this, the
        next ``get_connection_secrets`` (scoped to the destination)
        would attempt to decrypt with the wrong subkey and raise
        :class:`CryptoDecryptError`.
        """
        conn = _conn_with_transaction()
        pool = cast("asyncpg.Pool[Any]", fake_pool_yielding_conn(conn))

        # Encrypt a known secrets dict with the source account's subkey
        # so the SELECT FOR UPDATE returns a plausible at-rest blob.
        source_account_id = "acc_src"
        destination_account_id = "acc_dest"
        original_plaintext = {"bot_token": "tg-secret-abc123"}
        source_blob = _CRYPTO_BOX.derive_account_subkey(source_account_id).encrypt_dict(
            original_plaintext
        )
        conn.fetchrow = AsyncMock(
            return_value={
                "archived_at": None,
                "account_id": source_account_id,
                "secrets_ciphertext": source_blob.ciphertext,
                "secrets_nonce": source_blob.nonce,
            }
        )

        updated = _connection(connection_id="conn_x", account_id=destination_account_id)
        with (
            patch(
                "aios.services.connections.queries.get_account",
                AsyncMock(
                    side_effect=[
                        _root_account("acc_root"),
                        _child_account(destination_account_id),
                    ]
                ),
            ),
            patch(
                "aios.services.connections.queries.reparent_connection",
                AsyncMock(return_value=updated),
            ) as reparent_query,
        ):
            await reparent_connection(
                pool,
                "conn_x",
                destination_account_id=destination_account_id,
                requester_account_id="acc_root",
                crypto_box=_CRYPTO_BOX,
            )

        reparent_query.assert_awaited_once()
        assert reparent_query.await_args is not None
        kwargs = reparent_query.await_args.kwargs
        rekeyed = kwargs["secrets_blob"]
        assert rekeyed is not None
        # Ciphertext must differ from the source blob (re-encryption
        # rolls a fresh nonce; even on the wildly unlikely nonce
        # collision the ciphertexts under different keys differ).
        assert rekeyed.ciphertext != source_blob.ciphertext
        # Decrypting the rekeyed blob with the DESTINATION subkey must
        # recover the original plaintext.
        recovered = _CRYPTO_BOX.derive_account_subkey(destination_account_id).decrypt_dict(rekeyed)
        assert recovered == original_plaintext
