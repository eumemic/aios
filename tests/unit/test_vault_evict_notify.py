"""The vault credential-mutation service paths fire the MCP-pool eviction
NOTIFY (#1030).

An operator rotating/retiring a vault credential runs in the API process,
but the live MCP ``ClientSession`` baked in the OLD bearer lives in the
worker's pool. These tests pin that each mutation path emits a
``pg_notify`` on ``aios_mcp_evict_vault`` with the ``vault_id`` so the
worker evicts the stale session instead of waiting out the idle TTL.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.crypto.vault import KEY_BYTES, CryptoBox
from aios.db import queries
from aios.db.listen import MCP_EVICT_VAULT_CHANNEL
from aios.models.vaults import Vault, VaultCredential, VaultCredentialUpdate
from aios.services import vaults as vaults_service

pytestmark = pytest.mark.asyncio


@pytest.fixture
def crypto_box() -> CryptoBox:
    return CryptoBox(os.urandom(KEY_BYTES))


def _fake_pool_with_conn(conn: Any) -> Any:
    """Pool whose ``acquire()`` yields *conn* and whose ``execute`` (the NOTIFY
    call site) is an awaitable spy."""
    pool = MagicMock()
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=None)
    pool.acquire.return_value = cm
    pool.execute = AsyncMock()
    return pool


def _conn_with_transaction() -> MagicMock:
    conn = MagicMock()
    txn_cm = MagicMock()
    txn_cm.__aenter__ = AsyncMock(return_value=None)
    txn_cm.__aexit__ = AsyncMock(return_value=None)
    conn.transaction = MagicMock(return_value=txn_cm)
    return conn


def _credential() -> VaultCredential:
    return VaultCredential(
        id="vc_1",
        vault_id="vlt_1",
        display_name="orig",
        target_url="https://mcp.example.com",
        auth_type="bearer_header",
        metadata={},
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


def _vault() -> Vault:
    return Vault(
        id="vlt_1",
        display_name="v",
        metadata={},
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


def _assert_notified(pool: Any, vault_id: str) -> None:
    pool.execute.assert_awaited_once()
    args = pool.execute.await_args.args
    assert args[0] == "SELECT pg_notify($1, $2)"
    assert args[1] == MCP_EVICT_VAULT_CHANNEL
    assert args[2] == vault_id


async def test_update_credential_fires_evict_notify(crypto_box: CryptoBox) -> None:
    account_id = "acc_1"
    blob = crypto_box.derive_account_subkey(account_id).encrypt(json.dumps({"token": "old"}))
    conn = _conn_with_transaction()
    pool = _fake_pool_with_conn(conn)

    with (
        patch.object(
            queries,
            "get_vault_credential_with_blob",
            AsyncMock(return_value=(_credential(), blob)),
        ),
        patch.object(queries, "update_vault_credential", AsyncMock(return_value=_credential())),
    ):
        await vaults_service.update_vault_credential(
            pool,
            crypto_box,
            account_id=account_id,
            vault_id="vlt_1",
            credential_id="vc_1",
            body=VaultCredentialUpdate(token="new"),
        )

    _assert_notified(pool, "vlt_1")


async def test_archive_credential_fires_evict_notify() -> None:
    conn = MagicMock()
    pool = _fake_pool_with_conn(conn)
    with patch.object(queries, "archive_vault_credential", AsyncMock(return_value=_credential())):
        await vaults_service.archive_vault_credential(pool, "vlt_1", "vc_1", account_id="acc_1")
    _assert_notified(pool, "vlt_1")


async def test_delete_credential_fires_evict_notify() -> None:
    conn = MagicMock()
    pool = _fake_pool_with_conn(conn)
    with patch.object(queries, "delete_vault_credential", AsyncMock(return_value=None)):
        await vaults_service.delete_vault_credential(pool, "vlt_1", "vc_1", account_id="acc_1")
    _assert_notified(pool, "vlt_1")


async def test_archive_vault_fires_evict_notify() -> None:
    conn = MagicMock()
    pool = _fake_pool_with_conn(conn)
    with patch.object(queries, "archive_vault", AsyncMock(return_value=_vault())):
        await vaults_service.archive_vault(pool, "vlt_1", account_id="acc_1")
    _assert_notified(pool, "vlt_1")


async def test_delete_vault_fires_evict_notify() -> None:
    conn = MagicMock()
    pool = _fake_pool_with_conn(conn)
    with patch.object(queries, "delete_vault", AsyncMock(return_value=None)):
        await vaults_service.delete_vault(pool, "vlt_1", account_id="acc_1")
    _assert_notified(pool, "vlt_1")


async def test_create_credential_does_not_fire_evict_notify(crypto_box: CryptoBox) -> None:
    """A brand-new credential has no pooled session yet — no eviction needed,
    so create must NOT fire the NOTIFY (scope discipline)."""
    from aios.models.vaults import VaultCredentialCreate

    conn = _conn_with_transaction()
    conn.fetchrow = AsyncMock(return_value={"?column?": 1})
    pool = _fake_pool_with_conn(conn)

    with (
        patch.object(queries, "count_active_vault_credentials", AsyncMock(return_value=0)),
        patch.object(queries, "insert_vault_credential", AsyncMock(return_value=_credential())),
    ):
        await vaults_service.create_vault_credential(
            pool,
            crypto_box,
            account_id="acc_1",
            vault_id="vlt_1",
            body=VaultCredentialCreate(
                display_name="new",
                target_url="https://new.example.com",
                auth_type="bearer_header",
                token="tok",
            ),
        )

    pool.execute.assert_not_awaited()
