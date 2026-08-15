"""Account scoping on the vault-credential paths is enforced by the SQL predicate.

Every test here is written to FAIL if the ``AND account_id = $N`` predicate is
removed from the query it exercises.  That is the point: the pre-existing
coverage asserted cross-account failure over *mocked* queries (or a mocked
query *result*), so it would still have passed with the predicate deleted --
the tests-that-cannot-fail class, sitting on an authority boundary.

The credential PUT reaches the database through two different shapes -- the
ordinary in-place update and the rescope archive+insert replacement -- so both
are driven here, plus the query layer directly.  A foreign ``account_id`` must
produce ``NotFoundError`` (the authority answer), not a decrypt failure and not
a mutated row: crypto failing *incidentally* is not access control, and a
credential must not be read, archived, or replaced across a tenant boundary.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest
from pydantic import SecretStr

from aios.crypto.vault import CryptoBox
from aios.db import queries as db_queries
from aios.db.pool import create_pool
from aios.errors import NotFoundError
from aios.models.vaults import VaultCredentialCreate, VaultCredentialUpdate
from aios.services import vaults as vaults_service

pytestmark = pytest.mark.integration

ACC_OWNER = "acc_vc_scope_owner"
ACC_OTHER = "acc_vc_scope_other"


@pytest.fixture
def crypto_box() -> CryptoBox:
    return CryptoBox(os.urandom(32))


@pytest.fixture
async def vault_pool(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[asyncpg.Pool[Any]]:
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, "
                "display_name) VALUES ($1, NULL, TRUE, 'vc-scope-root')",
                ACC_OWNER,
            )
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, "
                "display_name) VALUES ($1, $2, FALSE, 'vc-scope-other')",
                ACC_OTHER,
                ACC_OWNER,
            )
        yield pool
    finally:
        await pool.close()


async def _seed_credential(pool: asyncpg.Pool[Any], crypto_box: CryptoBox) -> tuple[str, str]:
    """Create a vault + environment_variable credential owned by ``ACC_OWNER``."""
    vault = await vaults_service.create_vault(
        pool, display_name="scoped-vault", metadata={}, account_id=ACC_OWNER
    )
    cred = await vaults_service.create_vault_credential(
        pool,
        crypto_box,
        vault_id=vault.id,
        body=VaultCredentialCreate(
            auth_type="environment_variable",
            secret_name="SCOPED_KEY",
            allowed_hosts=["owner.example.com"],
            secret_value=SecretStr("owner-secret"),
            metadata={"owner": "ops"},
        ),
        account_id=ACC_OWNER,
    )
    return vault.id, cred.id


async def _assert_row_untouched(pool: asyncpg.Pool[Any], vault_id: str, credential_id: str) -> None:
    """The owner's credential is still active, still scoped, still holds its secret."""
    cred = await vaults_service.get_vault_credential(
        pool, vault_id, credential_id, account_id=ACC_OWNER
    )
    assert cred.archived_at is None, "foreign account archived the owner's credential"
    assert cred.allowed_hosts == ["owner.example.com"]
    assert cred.secret_name == "SCOPED_KEY"
    async with pool.acquire() as conn:
        ciphertext = await conn.fetchval(
            "SELECT ciphertext FROM vault_credentials WHERE id = $1", credential_id
        )
    assert bytes(ciphertext) != b"", "foreign account scrubbed the owner's ciphertext"


class TestVaultCredentialAccountScoping:
    """A foreign account must not reach another tenant's credential."""

    async def test_rescope_path_refuses_foreign_account(
        self, vault_pool: asyncpg.Pool[Any], crypto_box: CryptoBox
    ) -> None:
        """The archive+insert replacement path is account-scoped.

        This is the path this PR adds.  Removing the account predicate lets the
        foreign caller archive the owner's row and mint a replacement under its
        own account -- destroying another tenant's credential.
        """
        vault_id, credential_id = await _seed_credential(vault_pool, crypto_box)

        with pytest.raises(NotFoundError):
            await vaults_service.update_vault_credential(
                vault_pool,
                crypto_box,
                vault_id=vault_id,
                credential_id=credential_id,
                body=VaultCredentialUpdate(allowed_hosts=["attacker.example.com"]),
                account_id=ACC_OTHER,
            )

        await _assert_row_untouched(vault_pool, vault_id, credential_id)

    async def test_ordinary_update_path_refuses_foreign_account(
        self, vault_pool: asyncpg.Pool[Any], crypto_box: CryptoBox
    ) -> None:
        """The in-place update path is account-scoped too.

        Same request shape, no rescope -- the other input that reaches the same
        wrong outcome if the predicate is dropped.
        """
        vault_id, credential_id = await _seed_credential(vault_pool, crypto_box)

        with pytest.raises(NotFoundError):
            await vaults_service.update_vault_credential(
                vault_pool,
                crypto_box,
                vault_id=vault_id,
                credential_id=credential_id,
                body=VaultCredentialUpdate(display_name="pwned"),
                account_id=ACC_OTHER,
            )

        cred = await vaults_service.get_vault_credential(
            vault_pool, vault_id, credential_id, account_id=ACC_OWNER
        )
        assert cred.display_name != "pwned"
        await _assert_row_untouched(vault_pool, vault_id, credential_id)

    async def test_metadata_clear_refuses_foreign_account(
        self, vault_pool: asyncpg.Pool[Any], crypto_box: CryptoBox
    ) -> None:
        """The explicit ``metadata: null`` clear is not a way around the predicate.

        The clear is the behaviour this PR changed; it must still be gated by
        account, on the rescope path where the two features combine.
        """
        vault_id, credential_id = await _seed_credential(vault_pool, crypto_box)

        with pytest.raises(NotFoundError):
            await vaults_service.update_vault_credential(
                vault_pool,
                crypto_box,
                vault_id=vault_id,
                credential_id=credential_id,
                body=VaultCredentialUpdate(allowed_hosts=["attacker.example.com"], metadata=None),
                account_id=ACC_OTHER,
            )

        cred = await vaults_service.get_vault_credential(
            vault_pool, vault_id, credential_id, account_id=ACC_OWNER
        )
        assert cred.metadata == {"owner": "ops"}, "foreign account cleared owner metadata"
        await _assert_row_untouched(vault_pool, vault_id, credential_id)

    async def test_owner_rescope_succeeds(
        self, vault_pool: asyncpg.Pool[Any], crypto_box: CryptoBox
    ) -> None:
        """Positive control: the predicate blocks the foreigner, not everyone.

        Without this, every assertion above would still pass if the service
        simply raised ``NotFoundError`` unconditionally.
        """
        vault_id, credential_id = await _seed_credential(vault_pool, crypto_box)

        replacement = await vaults_service.update_vault_credential(
            vault_pool,
            crypto_box,
            vault_id=vault_id,
            credential_id=credential_id,
            body=VaultCredentialUpdate(allowed_hosts=["new.example.com"]),
            account_id=ACC_OWNER,
        )

        assert replacement.id != credential_id
        assert replacement.allowed_hosts == ["new.example.com"]


class TestVaultCredentialQueryLayerScoping:
    """The predicate lives in SQL, so assert it in SQL -- one test per query.

    Driving the service only ever proves the *first* gate it hits.  These pin
    each credential query independently, so deleting the predicate from any one
    of them turns a test red rather than being masked by an earlier check.
    """

    async def test_get_with_blob_is_scoped(
        self, vault_pool: asyncpg.Pool[Any], crypto_box: CryptoBox
    ) -> None:
        vault_id, credential_id = await _seed_credential(vault_pool, crypto_box)
        async with vault_pool.acquire() as conn:
            with pytest.raises(NotFoundError):
                await db_queries.get_vault_credential_with_blob(
                    conn, vault_id, credential_id, account_id=ACC_OTHER
                )

    async def test_get_is_scoped(
        self, vault_pool: asyncpg.Pool[Any], crypto_box: CryptoBox
    ) -> None:
        vault_id, credential_id = await _seed_credential(vault_pool, crypto_box)
        async with vault_pool.acquire() as conn:
            with pytest.raises(NotFoundError):
                await db_queries.get_vault_credential(
                    conn, vault_id, credential_id, account_id=ACC_OTHER
                )

    async def test_archive_is_scoped(
        self, vault_pool: asyncpg.Pool[Any], crypto_box: CryptoBox
    ) -> None:
        """Archive scrubs the ciphertext -- an unscoped archive is destructive."""
        vault_id, credential_id = await _seed_credential(vault_pool, crypto_box)
        async with vault_pool.acquire() as conn:
            with pytest.raises(NotFoundError):
                await db_queries.archive_vault_credential(
                    conn, vault_id, credential_id, account_id=ACC_OTHER
                )
        await _assert_row_untouched(vault_pool, vault_id, credential_id)

    async def test_update_is_scoped(
        self, vault_pool: asyncpg.Pool[Any], crypto_box: CryptoBox
    ) -> None:
        vault_id, credential_id = await _seed_credential(vault_pool, crypto_box)
        async with vault_pool.acquire() as conn:
            with pytest.raises(NotFoundError):
                await db_queries.update_vault_credential(
                    conn,
                    vault_id,
                    credential_id,
                    account_id=ACC_OTHER,
                    display_name="pwned",
                )
        cred = await vaults_service.get_vault_credential(
            vault_pool, vault_id, credential_id, account_id=ACC_OWNER
        )
        assert cred.display_name != "pwned"

    async def test_list_is_scoped(
        self, vault_pool: asyncpg.Pool[Any], crypto_box: CryptoBox
    ) -> None:
        vault_id, _ = await _seed_credential(vault_pool, crypto_box)
        async with vault_pool.acquire() as conn:
            assert (
                await db_queries.list_vault_credentials(conn, vault_id, account_id=ACC_OTHER) == []
            )
            assert (
                await db_queries.list_vault_credentials(conn, vault_id, account_id=ACC_OWNER) != []
            )
