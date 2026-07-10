"""Integration tests for the model_providers resource: CRUD, tenant scoping,
archive-zeroing, and the `aios rekey` wiring.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.cli.commands.ops import _REKEY_COLUMNS, _rekey_column
from aios.crypto.vault import KEY_BYTES, CryptoBox, EncryptedBlob
from aios.db import queries
from aios.db.pool import create_pool
from aios.errors import ConflictError, CryptoDecryptError, NotFoundError
from aios.services import model_providers as service

pytestmark = pytest.mark.integration

_MODEL_PROVIDERS_COL = next(c for c in _REKEY_COLUMNS if c.table == "model_providers")


@pytest.fixture
async def pool_with_account(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str]]:
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_mp_test', NULL, TRUE, 'model-providers-test')
                """
            )
        yield pool, "acc_mp_test"
    finally:
        await pool.close()


@pytest.fixture
def crypto_box() -> CryptoBox:
    return CryptoBox(os.urandom(KEY_BYTES))


async def test_create_get_list(
    pool_with_account: tuple[asyncpg.Pool[Any], str], crypto_box: CryptoBox
) -> None:
    pool, account_id = pool_with_account
    created = await service.create_model_provider(
        pool,
        crypto_box,
        account_id=account_id,
        provider="anthropic",
        api_key="sk-real",
        api_base="https://proxy.example",
    )
    assert created.api_key_set is True
    assert created.api_base == "https://proxy.example"

    fetched = await service.get_model_provider(pool, created.id, account_id=account_id)
    assert fetched == created

    listed = await service.list_model_providers(pool, account_id=account_id)
    assert [p.id for p in listed] == [created.id]


async def test_duplicate_active_provider_conflicts(
    pool_with_account: tuple[asyncpg.Pool[Any], str], crypto_box: CryptoBox
) -> None:
    pool, account_id = pool_with_account
    await service.create_model_provider(
        pool, crypto_box, account_id=account_id, provider="anthropic", api_key="k1", api_base=None
    )
    with pytest.raises(ConflictError):
        await service.create_model_provider(
            pool,
            crypto_box,
            account_id=account_id,
            provider="anthropic",
            api_key="k2",
            api_base=None,
        )


async def test_create_after_archive_succeeds(
    pool_with_account: tuple[asyncpg.Pool[Any], str], crypto_box: CryptoBox
) -> None:
    pool, account_id = pool_with_account
    first = await service.create_model_provider(
        pool, crypto_box, account_id=account_id, provider="anthropic", api_key="k1", api_base=None
    )
    await service.archive_model_provider(pool, first.id, account_id=account_id)
    second = await service.create_model_provider(
        pool, crypto_box, account_id=account_id, provider="anthropic", api_key="k2", api_base=None
    )
    assert second.id != first.id
    assert second.archived_at is None


async def test_archive_zeroes_ciphertext(
    pool_with_account: tuple[asyncpg.Pool[Any], str], crypto_box: CryptoBox
) -> None:
    pool, account_id = pool_with_account
    created = await service.create_model_provider(
        pool, crypto_box, account_id=account_id, provider="anthropic", api_key="k1", api_base=None
    )
    archived = await service.archive_model_provider(pool, created.id, account_id=account_id)
    assert archived.archived_at is not None
    assert archived.api_key_set is False

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT ciphertext, nonce FROM model_providers WHERE id = $1", created.id
        )
    assert row is not None
    assert bytes(row["ciphertext"]) == b""
    assert bytes(row["nonce"]) == b""


async def test_archive_is_idempotent(
    pool_with_account: tuple[asyncpg.Pool[Any], str], crypto_box: CryptoBox
) -> None:
    pool, account_id = pool_with_account
    created = await service.create_model_provider(
        pool, crypto_box, account_id=account_id, provider="anthropic", api_key="k1", api_base=None
    )
    first = await service.archive_model_provider(pool, created.id, account_id=account_id)
    second = await service.archive_model_provider(
        pool, created.id, account_id=account_id, idempotent=True
    )
    assert second.id == first.id
    assert second.archived_at == first.archived_at

    with pytest.raises(NotFoundError):
        await service.archive_model_provider(
            pool, created.id, account_id=account_id, idempotent=False
        )


async def test_cross_tenant_isolation(
    conn_two_accounts: asyncpg.Connection[Any], crypto_box: CryptoBox
) -> None:
    """A row created for one account is invisible to (and unmodifiable by) another."""
    blob = crypto_box.derive_account_subkey("acc_a").encrypt("sk-a")
    row = await queries.insert_model_provider(
        conn_two_accounts, account_id="acc_a", provider="anthropic", api_base=None, blob=blob
    )

    with pytest.raises(NotFoundError):
        await queries.get_model_provider(conn_two_accounts, row.id, account_id="acc_b")

    listed_b = await queries.list_model_providers(conn_two_accounts, account_id="acc_b")
    assert listed_b == []

    with pytest.raises(NotFoundError):
        await queries.archive_model_provider(conn_two_accounts, row.id, account_id="acc_b")

    # The row is untouched — still active under its real owner.
    still_active = await queries.get_model_provider(conn_two_accounts, row.id, account_id="acc_a")
    assert still_active.archived_at is None


async def test_rekey_round_trip(
    pool_with_account: tuple[asyncpg.Pool[Any], str],
) -> None:
    """`aios rekey`'s generic column-rewriter correctly covers model_providers:
    a row encrypted under an old master key decrypts under the rotated one
    after `_rekey_column` runs, and no longer decrypts under the old key.
    """
    pool, account_id = pool_with_account
    old_master = CryptoBox(os.urandom(KEY_BYTES))
    new_master = CryptoBox(os.urandom(KEY_BYTES))

    old_blob = old_master.derive_account_subkey(account_id).encrypt("sk-pre-rotation")
    async with pool.acquire() as conn:
        row = await queries.insert_model_provider(
            conn, account_id=account_id, provider="anthropic", api_base=None, blob=old_blob
        )

        rekeyed_count = await _rekey_column(conn, _MODEL_PROVIDERS_COL, new_master, old_master)
        assert rekeyed_count == 1

        refreshed = await conn.fetchrow(
            "SELECT ciphertext, nonce FROM model_providers WHERE id = $1", row.id
        )
    assert refreshed is not None
    new_blob = EncryptedBlob(bytes(refreshed["ciphertext"]), bytes(refreshed["nonce"]))
    assert new_master.derive_account_subkey(account_id).decrypt(new_blob) == "sk-pre-rotation"

    with pytest.raises(CryptoDecryptError):
        old_master.derive_account_subkey(account_id).decrypt(new_blob)
