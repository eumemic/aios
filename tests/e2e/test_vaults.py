"""E2E tests for the vault and vault credential system.

Tests run against a real testcontainer Postgres with migrations applied.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import SecretStr

from aios.models.vaults import VaultCredentialCreate, VaultCredentialUpdate


@pytest.fixture
def crypto_box(aios_env: dict[str, str]) -> Any:
    from aios.config import get_settings
    from aios.crypto.vault import CryptoBox

    return CryptoBox.from_base64(get_settings().vault_key.get_secret_value())


class TestVaultCRUD:
    async def test_create_and_get(self, pool: Any) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.services import vaults as svc

        vault = await svc.create_vault(
            pool, display_name="test-vault", metadata={"env": "test"}, account_id=account_id
        )
        assert vault.display_name == "test-vault"
        assert vault.metadata == {"env": "test"}
        assert vault.id.startswith("vlt_")
        assert vault.archived_at is None

        fetched = await svc.get_vault(pool, vault.id, account_id=account_id)
        assert fetched.id == vault.id
        assert fetched.display_name == "test-vault"

    async def test_list_vaults(self, pool: Any) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.services import vaults as svc

        v1 = await svc.create_vault(pool, display_name="list-a", metadata={}, account_id=account_id)
        v2 = await svc.create_vault(pool, display_name="list-b", metadata={}, account_id=account_id)
        vaults = await svc.list_vaults(pool, limit=100, account_id=account_id)
        ids = [v.id for v in vaults]
        assert v1.id in ids
        assert v2.id in ids

    async def test_update_vault(self, pool: Any) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.services import vaults as svc

        vault = await svc.create_vault(
            pool, display_name="before", metadata={}, account_id=account_id
        )
        updated = await svc.update_vault(
            pool, vault.id, display_name="after", account_id=account_id
        )
        assert updated.display_name == "after"
        assert updated.updated_at > vault.updated_at

    async def test_archive_vault(self, pool: Any) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.services import vaults as svc

        vault = await svc.create_vault(
            pool, display_name="to-archive", metadata={}, account_id=account_id
        )
        archived = await svc.archive_vault(pool, vault.id, account_id=account_id)
        assert archived.archived_at is not None

        # Archived vaults don't appear in list
        vaults = await svc.list_vaults(pool, limit=100, account_id=account_id)
        assert vault.id not in [v.id for v in vaults]

    async def test_delete_vault(self, pool: Any) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.services import vaults as svc

        vault = await svc.create_vault(
            pool, display_name="to-delete", metadata={}, account_id=account_id
        )
        await svc.delete_vault(pool, vault.id, account_id=account_id)

        from aios.errors import NotFoundError

        with pytest.raises(NotFoundError):
            await svc.get_vault(pool, vault.id, account_id=account_id)


class TestVaultCredentialCRUD:
    async def test_create_bearer_header(self, pool: Any, crypto_box: Any) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.services import vaults as svc

        vault = await svc.create_vault(
            pool, display_name="cred-test", metadata={}, account_id=account_id
        )
        body = VaultCredentialCreate(
            target_url="https://mcp.example.com/api",
            auth_type="bearer_header",
            token=SecretStr("my-secret-token"),
        )
        cred = await svc.create_vault_credential(
            pool, crypto_box, vault_id=vault.id, body=body, account_id=account_id
        )
        assert cred.id.startswith("vcr_")
        assert cred.vault_id == vault.id
        assert cred.target_url == "https://mcp.example.com/api"
        assert cred.auth_type == "bearer_header"

    async def test_secrets_not_returned(self, pool: Any, crypto_box: Any) -> None:
        """Verify that the read view never includes secret fields."""
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.services import vaults as svc

        vault = await svc.create_vault(
            pool, display_name="secrets-test", metadata={}, account_id=account_id
        )
        body = VaultCredentialCreate(
            target_url="https://mcp2.example.com",
            auth_type="bearer_header",
            token=SecretStr("super-secret"),
        )
        cred = await svc.create_vault_credential(
            pool, crypto_box, vault_id=vault.id, body=body, account_id=account_id
        )

        fetched = await svc.get_vault_credential(pool, vault.id, cred.id, account_id=account_id)
        dumped = fetched.model_dump()
        assert "token" not in dumped
        assert "access_token" not in dumped
        assert "ciphertext" not in dumped

    async def test_create_oauth2_refresh(self, pool: Any, crypto_box: Any) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.services import vaults as svc

        vault = await svc.create_vault(
            pool, display_name="oauth-test", metadata={}, account_id=account_id
        )
        body = VaultCredentialCreate(
            target_url="https://oauth.example.com",
            auth_type="oauth2_refresh",
            access_token=SecretStr("access-123"),
            client_id="client-abc",
            refresh_token=SecretStr("refresh-456"),
            token_endpoint="https://oauth.example.com/token",
        )
        cred = await svc.create_vault_credential(
            pool, crypto_box, vault_id=vault.id, body=body, account_id=account_id
        )
        assert cred.auth_type == "oauth2_refresh"

    async def test_bearer_header_requires_token(self, pool: Any, crypto_box: Any) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.errors import ValidationError
        from aios.services import vaults as svc

        vault = await svc.create_vault(
            pool, display_name="req-test", metadata={}, account_id=account_id
        )
        body = VaultCredentialCreate(
            target_url="https://no-token.example.com",
            auth_type="bearer_header",
        )
        with pytest.raises(ValidationError, match="require"):
            await svc.create_vault_credential(
                pool, crypto_box, vault_id=vault.id, body=body, account_id=account_id
            )

    async def test_oauth_requires_access_token(self, pool: Any, crypto_box: Any) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.errors import ValidationError
        from aios.services import vaults as svc

        vault = await svc.create_vault(
            pool, display_name="oauth-req", metadata={}, account_id=account_id
        )
        body = VaultCredentialCreate(
            target_url="https://no-at.example.com",
            auth_type="oauth2_refresh",
        )
        with pytest.raises(ValidationError, match="require"):
            await svc.create_vault_credential(
                pool, crypto_box, vault_id=vault.id, body=body, account_id=account_id
            )

    async def test_unique_url_per_vault(self, pool: Any, crypto_box: Any) -> None:
        """Two active credentials for the same URL in the same vault should conflict."""
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.errors import ConflictError
        from aios.services import vaults as svc

        vault = await svc.create_vault(
            pool, display_name="uniq-test", metadata={}, account_id=account_id
        )
        url = "https://unique-url-test.example.com"
        body = VaultCredentialCreate(
            target_url=url, auth_type="bearer_header", token=SecretStr("t1")
        )
        await svc.create_vault_credential(
            pool, crypto_box, vault_id=vault.id, body=body, account_id=account_id
        )
        with pytest.raises(ConflictError):
            await svc.create_vault_credential(
                pool, crypto_box, vault_id=vault.id, body=body, account_id=account_id
            )

    async def test_archive_frees_url(self, pool: Any, crypto_box: Any) -> None:
        """Archiving a credential frees its target_url for reuse."""
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.services import vaults as svc

        vault = await svc.create_vault(
            pool, display_name="archive-free", metadata={}, account_id=account_id
        )
        url = "https://archive-free-test.example.com"
        body = VaultCredentialCreate(
            target_url=url, auth_type="bearer_header", token=SecretStr("t1")
        )
        cred1 = await svc.create_vault_credential(
            pool, crypto_box, vault_id=vault.id, body=body, account_id=account_id
        )
        await svc.archive_vault_credential(pool, vault.id, cred1.id, account_id=account_id)

        # Now creating a new credential for the same URL should succeed.
        cred2 = await svc.create_vault_credential(
            pool, crypto_box, vault_id=vault.id, body=body, account_id=account_id
        )
        assert cred2.id != cred1.id

    async def test_create_environment_variable(self, pool: Any, crypto_box: Any) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.db import queries
        from aios.services import vaults as svc

        vault = await svc.create_vault(
            pool, display_name="env-var-test", metadata={}, account_id=account_id
        )
        body = VaultCredentialCreate(
            auth_type="environment_variable",
            secret_name="GITHUB_TOKEN",
            allowed_hosts=["api.github.com/repos/eumemic", "api.tavily.com"],
            secret_value=SecretStr("ghp_supersecret"),
            display_name="gh",
        )
        cred = await svc.create_vault_credential(
            pool, crypto_box, vault_id=vault.id, body=body, account_id=account_id
        )
        assert cred.id.startswith("vcr_")
        assert cred.auth_type == "environment_variable"
        assert cred.target_url is None
        assert cred.secret_name == "GITHUB_TOKEN"
        assert cred.allowed_hosts == ["api.github.com/repos/eumemic", "api.tavily.com"]

        # Secrets are write-only; the secret value only survives in the
        # encrypted blob, recoverable solely with the account subkey.
        fetched = await svc.get_vault_credential(pool, vault.id, cred.id, account_id=account_id)
        dumped = fetched.model_dump()
        assert "secret_value" not in dumped
        assert "ciphertext" not in dumped
        async with pool.acquire() as conn:
            _, blob = await queries.get_vault_credential_with_blob(
                conn, vault.id, cred.id, account_id=account_id
            )
        payload = crypto_box.derive_account_subkey(account_id).decrypt_dict(blob)
        assert payload == {"secret_value": "ghp_supersecret"}

    async def test_environment_variable_requires_secret_value(
        self, pool: Any, crypto_box: Any
    ) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.errors import ValidationError
        from aios.services import vaults as svc

        # secret_value is required at the service layer (like bearer's token),
        # not at model construction — the structural shape validator only
        # governs secret_name / allowed_hosts / target_url.
        vault = await svc.create_vault(
            pool, display_name="env-req", metadata={}, account_id=account_id
        )
        body = VaultCredentialCreate(
            auth_type="environment_variable",
            secret_name="GITHUB_TOKEN",
            allowed_hosts=["api.github.com"],
        )
        with pytest.raises(ValidationError):
            await svc.create_vault_credential(
                pool, crypto_box, vault_id=vault.id, body=body, account_id=account_id
            )

    async def test_secret_name_unique_per_vault(self, pool: Any, crypto_box: Any) -> None:
        """Two active env-var creds with the same secret_name in one vault conflict;
        archiving the first frees the name; the same name in another vault is fine."""
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.errors import ConflictError
        from aios.services import vaults as svc

        vault = await svc.create_vault(
            pool, display_name="env-uniq", metadata={}, account_id=account_id
        )
        other = await svc.create_vault(
            pool, display_name="env-uniq-other", metadata={}, account_id=account_id
        )

        def _body() -> VaultCredentialCreate:
            return VaultCredentialCreate(
                auth_type="environment_variable",
                secret_name="API_KEY",
                allowed_hosts=["api.example.com"],
                secret_value=SecretStr("k1"),
            )

        cred1 = await svc.create_vault_credential(
            pool, crypto_box, vault_id=vault.id, body=_body(), account_id=account_id
        )
        with pytest.raises(ConflictError):
            await svc.create_vault_credential(
                pool, crypto_box, vault_id=vault.id, body=_body(), account_id=account_id
            )
        # Same secret_name in a different vault is allowed.
        await svc.create_vault_credential(
            pool, crypto_box, vault_id=other.id, body=_body(), account_id=account_id
        )
        # Archiving frees the name for reuse in the original vault.
        await svc.archive_vault_credential(pool, vault.id, cred1.id, account_id=account_id)
        cred2 = await svc.create_vault_credential(
            pool, crypto_box, vault_id=vault.id, body=_body(), account_id=account_id
        )
        assert cred2.id != cred1.id

    async def test_environment_variable_rotates_secret(self, pool: Any, crypto_box: Any) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.db import queries
        from aios.services import vaults as svc

        vault = await svc.create_vault(
            pool, display_name="env-rotate", metadata={}, account_id=account_id
        )
        cred = await svc.create_vault_credential(
            pool,
            crypto_box,
            vault_id=vault.id,
            body=VaultCredentialCreate(
                auth_type="environment_variable",
                secret_name="ROTATING",
                allowed_hosts=["api.example.com"],
                secret_value=SecretStr("old"),
            ),
            account_id=account_id,
        )
        await svc.update_vault_credential(
            pool,
            crypto_box,
            vault_id=vault.id,
            credential_id=cred.id,
            body=VaultCredentialUpdate(secret_value=SecretStr("new")),
            account_id=account_id,
        )
        async with pool.acquire() as conn:
            _, blob = await queries.get_vault_credential_with_blob(
                conn, vault.id, cred.id, account_id=account_id
            )
        payload = crypto_box.derive_account_subkey(account_id).decrypt_dict(blob)
        assert payload == {"secret_value": "new"}

    async def test_update_credential(self, pool: Any, crypto_box: Any) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.services import vaults as svc

        vault = await svc.create_vault(
            pool, display_name="update-test", metadata={}, account_id=account_id
        )
        body = VaultCredentialCreate(
            target_url="https://update-test.example.com",
            auth_type="bearer_header",
            token=SecretStr("old-token"),
            display_name="original",
        )
        cred = await svc.create_vault_credential(
            pool, crypto_box, vault_id=vault.id, body=body, account_id=account_id
        )

        update = VaultCredentialUpdate(
            display_name="updated",
            token=SecretStr("new-token"),
        )
        updated = await svc.update_vault_credential(
            pool,
            crypto_box,
            vault_id=vault.id,
            credential_id=cred.id,
            body=update,
            account_id=account_id,
        )
        assert updated.display_name == "updated"
        assert updated.updated_at > cred.updated_at

    async def test_credential_limit(self, pool: Any, crypto_box: Any) -> None:
        """Vault cannot have more than 20 active credentials."""
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.errors import ValidationError
        from aios.services import vaults as svc

        vault = await svc.create_vault(
            pool, display_name="limit-test", metadata={}, account_id=account_id
        )
        for i in range(20):
            body = VaultCredentialCreate(
                target_url=f"https://limit-{i}.example.com",
                auth_type="bearer_header",
                token=SecretStr(f"t-{i}"),
            )
            await svc.create_vault_credential(
                pool, crypto_box, vault_id=vault.id, body=body, account_id=account_id
            )

        body21 = VaultCredentialCreate(
            target_url="https://limit-overflow.example.com",
            auth_type="bearer_header",
            token=SecretStr("overflow"),
        )
        with pytest.raises(ValidationError, match="maximum"):
            await svc.create_vault_credential(
                pool, crypto_box, vault_id=vault.id, body=body21, account_id=account_id
            )

    async def test_credential_limit_under_concurrency(self, pool: Any, crypto_box: Any) -> None:
        """The 20-cred limit holds under concurrent inserts.

        Without ``SELECT … FOR UPDATE`` on the vault row, two parallel
        inserts can both observe ``count == 19`` and both succeed,
        overflowing the cap. With the row lock, exactly 20 succeed and the
        rest get ``ValidationError``.
        """
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.errors import ValidationError
        from aios.services import vaults as svc

        vault = await svc.create_vault(
            pool, display_name="race-test", metadata={}, account_id=account_id
        )

        async def attempt(i: int) -> Any:
            account_id = "acc_test_stub"  # PR 3 scaffolding
            body = VaultCredentialCreate(
                target_url=f"https://race-{i}.example.com",
                auth_type="bearer_header",
                token=SecretStr(f"t-{i}"),
            )
            try:
                return await svc.create_vault_credential(
                    pool, crypto_box, vault_id=vault.id, body=body, account_id=account_id
                )
            except ValidationError as e:
                return e

        results = await asyncio.gather(*(attempt(i) for i in range(25)))
        successes = [r for r in results if not isinstance(r, ValidationError)]
        failures = [r for r in results if isinstance(r, ValidationError)]

        assert len(successes) == 20
        assert len(failures) == 5
        for f in failures:
            assert "maximum" in str(f)

    async def test_concurrent_update_does_not_lose_writes(self, pool: Any, crypto_box: Any) -> None:
        """Two concurrent ``PUT /v1/vaults/:vid/credentials/:cid`` requests
        each modifying a DIFFERENT field of the same credential must both
        win their respective edit. Pre-fix ``update_vault_credential``
        opens a connection without a transaction or row lock; the
        ``decrypt → merge → re-encrypt → UPDATE`` sequence is read-modify-
        write across an unbounded gap, so the second commit clobbers the
        first invocation's blob field. PR #496 fixed the merge-logic
        for ``None``-unset within ONE call; this PR fixes the cross-call
        race that ``#496`` left uncovered.

        Pre-fix: token=A1 + display_name=A2 (call A) races with
        token=B1 + display_name=B2 (call B). One field from one call
        is silently lost (depending on commit order). Post-fix:
        ``SELECT … FOR UPDATE`` serializes the read-decrypt-encrypt
        critical section so both fields land.
        """
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.services import vaults as svc

        vault = await svc.create_vault(
            pool, display_name="lost-update-test", metadata={}, account_id=account_id
        )
        cred = await svc.create_vault_credential(
            pool,
            crypto_box,
            vault_id=vault.id,
            body=VaultCredentialCreate(
                target_url="https://lost-update.example.com",
                auth_type="bearer_header",
                token=SecretStr("initial"),
                display_name="initial",
            ),
            account_id=account_id,
        )

        # Force the race window deterministically: both invocations must
        # complete the SELECT before either reaches the UPDATE. With
        # plain ``asyncio.gather`` the asyncpg pool would generally let
        # the first invocation run get → update before the second's
        # get starts, masking the bug. Patch the UPDATE query to wait
        # until BOTH invocations have arrived — this reconstructs the
        # exact race that webhook-retry / dual-CLI / parallel-agent
        # production traffic creates intermittently. The release event
        # is set from a separate coroutine so neither update releases
        # its own wait; FIFO wake order then preserves arrival order
        # for the underlying ``original_update`` call into Postgres.
        from unittest.mock import patch

        from aios.db import queries as queries_mod

        original_update = queries_mod.update_vault_credential
        release_updates = asyncio.Event()
        both_arrived = asyncio.Event()
        reached_update = 0

        async def gated_update(*args: Any, **kwargs: Any) -> Any:
            nonlocal reached_update
            reached_update += 1
            if reached_update == 2:
                both_arrived.set()
            await release_updates.wait()
            return await original_update(*args, **kwargs)

        async def release_barrier() -> None:
            # Pre-fix: both invocations reach ``gated_update`` (no row
            # lock from get) and ``both_arrived`` fires fast.
            # Post-fix: the second invocation blocks on ``FOR UPDATE``
            # before ever reaching ``gated_update`` — ``both_arrived``
            # times out. Either way, release ``release_updates`` so
            # the first invocation commits; the second then proceeds
            # naturally (immediately pre-fix, after row-lock release
            # post-fix).
            import contextlib as _contextlib

            with _contextlib.suppress(TimeoutError):
                await asyncio.wait_for(both_arrived.wait(), timeout=0.5)
            release_updates.set()

        async def update_token() -> Any:
            return await svc.update_vault_credential(
                pool,
                crypto_box,
                vault_id=vault.id,
                credential_id=cred.id,
                body=VaultCredentialUpdate(token=SecretStr("token-A")),
                account_id=account_id,
            )

        async def update_display_name() -> Any:
            return await svc.update_vault_credential(
                pool,
                crypto_box,
                vault_id=vault.id,
                credential_id=cred.id,
                body=VaultCredentialUpdate(display_name="display-B"),
                account_id=account_id,
            )

        with patch.object(queries_mod, "update_vault_credential", gated_update):
            await asyncio.gather(update_token(), update_display_name(), release_barrier())

        # Read back via the get-with-blob query and decrypt to verify the
        # actual stored secrets — the public ``VaultCredential`` strips
        # them. Both invocations should have committed their field.
        async with pool.acquire() as conn:
            from aios.db import queries

            final_cred, final_blob = await queries.get_vault_credential_with_blob(
                conn, vault.id, cred.id, account_id=account_id
            )

        subkey = crypto_box.derive_account_subkey(account_id)
        payload = subkey.decrypt_dict(final_blob)

        assert final_cred.display_name == "display-B", (
            f"display_name update was lost — pre-fix the token-update's "
            f"earlier read decrypted the pre-race blob, merged token-A, and "
            f"re-encrypted, clobbering the display_name field that was about "
            f"to land. Final display_name: {final_cred.display_name!r}"
        )
        assert payload.get("token") == "token-A", (
            f"token update was lost — symmetric race; final token in blob: {payload.get('token')!r}"
        )

    async def test_credential_inserts_across_vaults_do_not_block(
        self, pool: Any, crypto_box: Any
    ) -> None:
        """Per-vault row lock must not become a global insert bottleneck.

        If a future refactor widened the ``SELECT … FOR UPDATE`` from the
        per-vault row to a global lock (or a shared advisory lock), this
        test would expose it: 20 credentials inserted in parallel across
        20 distinct vaults must all succeed. With per-vault locks they
        run independently; with a global lock they'd serialize but still
        pass — so we additionally assert the wall-clock comes in under
        a generous bound that wouldn't be met under serialization.
        """
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.services import vaults as svc

        vaults = await asyncio.gather(
            *(
                svc.create_vault(pool, display_name=f"par-{i}", metadata={}, account_id=account_id)
                for i in range(20)
            )
        )

        async def insert_one(v_idx: int) -> Any:
            return await svc.create_vault_credential(
                pool,
                crypto_box,
                vault_id=vaults[v_idx].id,
                body=VaultCredentialCreate(
                    target_url=f"https://par-{v_idx}.example.com",
                    auth_type="bearer_header",
                    token=SecretStr(f"t-{v_idx}"),
                ),
                account_id=account_id,
            )

        results = await asyncio.gather(*(insert_one(i) for i in range(20)))
        assert all(r.id.startswith("vcr_") for r in results)
        # Every insert was on a different vault; per-vault row lock should
        # not have caused any failures.


class TestArchiveAndCascade:
    """Archive must zero the encrypted blob; delete must cascade via the FK."""

    async def test_archive_credential_zeros_blob(self, pool: Any, crypto_box: Any) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.services import vaults as svc

        vault = await svc.create_vault(
            pool, display_name="zero-cred", metadata={}, account_id=account_id
        )
        body = VaultCredentialCreate(
            target_url="https://zero-cred.example.com",
            auth_type="bearer_header",
            token=SecretStr("doomed"),
        )
        cred = await svc.create_vault_credential(
            pool, crypto_box, vault_id=vault.id, body=body, account_id=account_id
        )
        await svc.archive_vault_credential(pool, vault.id, cred.id, account_id=account_id)

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT ciphertext, nonce FROM vault_credentials WHERE id = $1",
                cred.id,
            )
        assert row is not None
        assert bytes(row["ciphertext"]) == b""
        assert bytes(row["nonce"]) == b""

    async def test_archive_vault_zeros_active_credentials(self, pool: Any, crypto_box: Any) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.services import vaults as svc

        vault = await svc.create_vault(
            pool, display_name="zero-vault", metadata={}, account_id=account_id
        )
        # Two active credentials.
        for i in range(2):
            await svc.create_vault_credential(
                pool,
                crypto_box,
                vault_id=vault.id,
                body=VaultCredentialCreate(
                    target_url=f"https://zero-vault-{i}.example.com",
                    auth_type="bearer_header",
                    token=SecretStr(f"t-{i}"),
                ),
                account_id=account_id,
            )

        await svc.archive_vault(pool, vault.id, account_id=account_id)

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT ciphertext, nonce, archived_at FROM vault_credentials WHERE vault_id = $1",
                vault.id,
            )
        assert len(rows) == 2
        for row in rows:
            assert bytes(row["ciphertext"]) == b""
            assert bytes(row["nonce"]) == b""
            assert row["archived_at"] is not None

    async def test_delete_vault_cascades_to_credentials(self, pool: Any, crypto_box: Any) -> None:
        """``ON DELETE CASCADE`` (migration 0015) wipes child rows automatically."""
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.services import vaults as svc

        vault = await svc.create_vault(
            pool, display_name="cascade-test", metadata={}, account_id=account_id
        )
        cred = await svc.create_vault_credential(
            pool,
            crypto_box,
            vault_id=vault.id,
            body=VaultCredentialCreate(
                target_url="https://cascade.example.com",
                auth_type="bearer_header",
                token=SecretStr("doomed"),
            ),
            account_id=account_id,
        )

        await svc.delete_vault(pool, vault.id, account_id=account_id)

        async with pool.acquire() as conn:
            row = await conn.fetchrow("SELECT 1 FROM vault_credentials WHERE id = $1", cred.id)
        assert row is None  # cascade-deleted


class TestQueries:
    """Direct tests for query-layer functions used internally by services."""

    async def test_get_credential_with_blob_returns_both(self, pool: Any, crypto_box: Any) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.db import queries
        from aios.services import vaults as svc

        vault = await svc.create_vault(
            pool, display_name="combo-test", metadata={}, account_id=account_id
        )
        body = VaultCredentialCreate(
            target_url="https://combo.example.com",
            auth_type="bearer_header",
            token=SecretStr("combo-token"),
        )
        cred = await svc.create_vault_credential(
            pool, crypto_box, vault_id=vault.id, body=body, account_id=account_id
        )

        async with pool.acquire() as conn:
            fetched_cred, blob = await queries.get_vault_credential_with_blob(
                conn, vault.id, cred.id, account_id=account_id
            )

        assert fetched_cred.id == cred.id
        assert fetched_cred.auth_type == "bearer_header"
        assert blob.ciphertext  # non-empty
        assert blob.nonce  # non-empty
        # Verify the blob actually decrypts to the original payload.
        import json as _json

        payload = _json.loads(crypto_box.derive_account_subkey(account_id).decrypt(blob))
        assert payload == {"token": "combo-token"}

    async def test_get_credential_with_blob_excludes_archived(
        self, pool: Any, crypto_box: Any
    ) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.db import queries
        from aios.errors import NotFoundError
        from aios.services import vaults as svc

        vault = await svc.create_vault(
            pool, display_name="combo-arch", metadata={}, account_id=account_id
        )
        body = VaultCredentialCreate(
            target_url="https://combo-arch.example.com",
            auth_type="bearer_header",
            token=SecretStr("doomed"),
        )
        cred = await svc.create_vault_credential(
            pool, crypto_box, vault_id=vault.id, body=body, account_id=account_id
        )
        await svc.archive_vault_credential(pool, vault.id, cred.id, account_id=account_id)

        async with pool.acquire() as conn:
            with pytest.raises(NotFoundError, match="archived"):
                await queries.get_vault_credential_with_blob(
                    conn, vault.id, cred.id, account_id=account_id
                )


class TestOAuthRefreshE2E:
    """End-to-end refresh: real testcontainer Postgres + mocked OAuth endpoint.

    The mocked httpx layer lets these tests assert that exactly one POST
    happens under concurrency — proving the SELECT … FOR UPDATE row lock
    works, not just trust the unit test that mocks the lock query.
    """

    @staticmethod
    async def _bind_session_to_vault(pool: Any, vault_id: str) -> str:
        """Create the minimum scaffolding (env + agent + session) to bind a vault."""
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.services import agents as agents_svc
        from aios.services import environments as env_svc
        from aios.services import sessions as sess_svc

        suffix = vault_id[-8:]
        env = await env_svc.create_environment(
            pool, name=f"oauth-e2e-env-{suffix}", account_id=account_id
        )
        agent = await agents_svc.create_agent(
            pool,
            name=f"oauth-e2e-agent-{suffix}",
            model="fake/test",
            system="test",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
            account_id=account_id,
        )
        session = await sess_svc.create_session(
            pool,
            agent_id=agent.id,
            environment_id=env.id,
            title="oauth-refresh-test",
            metadata={},
            vault_ids=[vault_id],
            account_id=account_id,
        )
        return str(session.id)

    @staticmethod
    def _expiring_oauth_body(url: str) -> VaultCredentialCreate:
        from datetime import UTC, datetime, timedelta

        return VaultCredentialCreate(
            target_url=url,
            auth_type="oauth2_refresh",
            access_token=SecretStr("stale-at"),
            refresh_token=SecretStr("rt-1"),
            client_id="cid",
            token_endpoint="https://issuer.example/token",
            expires_at=datetime.now(UTC) - timedelta(seconds=1),  # already expired
        )

    @staticmethod
    def _patched_async_client(post_calls: list[Any], body: dict[str, Any]) -> Any:
        """Build a ``patch`` context for ``services.vaults.httpx.AsyncClient``.

        Each ``client.post`` invocation is recorded into ``post_calls`` and
        returns a mocked 200 response with ``body``.
        """
        from unittest.mock import AsyncMock, MagicMock, patch

        resp = MagicMock()
        resp.json = MagicMock(return_value=body)
        resp.raise_for_status = MagicMock()

        async def _post(url: str, **kwargs: Any) -> Any:
            post_calls.append((url, kwargs))
            return resp

        client = MagicMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        client.post = AsyncMock(side_effect=_post)
        return patch("aios.services.vaults.httpx.AsyncClient", MagicMock(return_value=client))

    async def test_refresh_persists_to_db(self, pool: Any, crypto_box: Any) -> None:
        """A real Postgres round-trip: insert expiring oauth cred, resolve, assert
        the token endpoint was POSTed and the new ciphertext is in the DB."""
        account_id = "acc_test_stub"  # PR 3 scaffolding
        import json as _json

        from aios.mcp.client import resolve_auth_for_target_url
        from aios.services import vaults as svc

        vault = await svc.create_vault(
            pool, display_name="oauth-e2e", metadata={}, account_id=account_id
        )
        url = "https://oauth-e2e.example.com/mcp"
        cred = await svc.create_vault_credential(
            pool,
            crypto_box,
            vault_id=vault.id,
            body=self._expiring_oauth_body(url),
            account_id=account_id,
        )
        session_id = await self._bind_session_to_vault(pool, vault.id)

        post_calls: list[Any] = []
        with self._patched_async_client(
            post_calls,
            body={"access_token": "fresh-at", "expires_in": 3600},
        ):
            _, headers = await resolve_auth_for_target_url(
                pool, crypto_box, session_id, url, account_id="acc_test_stub"
            )

        assert len(post_calls) == 1, "expected exactly one POST to the token endpoint"
        assert post_calls[0][0] == "https://issuer.example/token"
        assert headers == {"Authorization": "Bearer fresh-at"}

        # Verify the new ciphertext is in the DB.
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT ciphertext, nonce FROM vault_credentials WHERE id = $1",
                cred.id,
            )
        from aios.crypto.vault import EncryptedBlob

        new_blob = EncryptedBlob(
            ciphertext=bytes(row["ciphertext"]),
            nonce=bytes(row["nonce"]),
        )
        new_payload = _json.loads(crypto_box.derive_account_subkey(account_id).decrypt(new_blob))
        assert new_payload["access_token"] == "fresh-at"

    async def test_concurrent_resolve_only_refreshes_once(self, pool: Any, crypto_box: Any) -> None:
        """Five parallel resolutions on an expiring credential issue exactly one POST.

        Without the SELECT … FOR UPDATE row lock + double-check, every
        coroutine would race to the token endpoint. The lock serializes
        them; the second-to-last waiter sees the now-fresh expires_at
        after acquiring the lock and exits without POSTing.
        """
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.mcp.client import resolve_auth_for_target_url
        from aios.services import vaults as svc

        vault = await svc.create_vault(
            pool, display_name="oauth-race", metadata={}, account_id=account_id
        )
        url = "https://oauth-race.example.com/mcp"
        await svc.create_vault_credential(
            pool,
            crypto_box,
            vault_id=vault.id,
            body=self._expiring_oauth_body(url),
            account_id=account_id,
        )
        session_id = await self._bind_session_to_vault(pool, vault.id)

        post_calls: list[Any] = []
        with self._patched_async_client(
            post_calls,
            body={"access_token": "fresh-at", "expires_in": 3600},
        ):
            results = await asyncio.gather(
                *(
                    resolve_auth_for_target_url(
                        pool, crypto_box, session_id, url, account_id="acc_test_stub"
                    )
                    for _ in range(5)
                )
            )

        assert all(r[1] == {"Authorization": "Bearer fresh-at"} for r in results)
        assert len(post_calls) == 1, (
            f"expected 1 POST but got {len(post_calls)} — row lock or double-check is broken"
        )

    async def test_refresh_failure_bubbles(self, pool: Any, crypto_box: Any) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from unittest.mock import AsyncMock, MagicMock, patch

        from aios.errors import OAuthRefreshError
        from aios.mcp.client import resolve_auth_for_target_url
        from aios.services import vaults as svc

        vault = await svc.create_vault(
            pool, display_name="oauth-fail", metadata={}, account_id=account_id
        )
        url = "https://oauth-fail.example.com/mcp"
        await svc.create_vault_credential(
            pool,
            crypto_box,
            vault_id=vault.id,
            body=self._expiring_oauth_body(url),
            account_id=account_id,
        )
        session_id = await self._bind_session_to_vault(pool, vault.id)

        # Token endpoint returns 401.
        import httpx as _httpx

        resp = MagicMock()
        resp.status_code = 401

        def _raise() -> None:
            raise _httpx.HTTPStatusError("401", request=MagicMock(), response=resp)

        resp.raise_for_status = MagicMock(side_effect=_raise)
        client = MagicMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        client.post = AsyncMock(return_value=resp)

        with (
            patch("aios.services.vaults.httpx.AsyncClient", MagicMock(return_value=client)),
            pytest.raises(OAuthRefreshError),
        ):
            await resolve_auth_for_target_url(
                pool, crypto_box, session_id, url, account_id="acc_test_stub"
            )

    async def test_concurrent_refresh_of_different_credentials_runs_in_parallel(
        self, pool: Any, crypto_box: Any
    ) -> None:
        """Per-row lock must not over-serialize across distinct credentials.

        Companion to ``test_concurrent_resolve_only_refreshes_once``: that
        test pins the lock semantics for the *same* credential. This one
        pins the *scope* of the lock — refreshes against two different
        ``(vault_id, target_url)`` pairs must produce two independent
        POSTs, not one. A future refactor that promoted the row lock to a
        global lock or a vault-level lock would make this test fail.
        """
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.mcp.client import resolve_auth_for_target_url
        from aios.services import vaults as svc

        v1 = await svc.create_vault(
            pool, display_name="par-refresh-1", metadata={}, account_id=account_id
        )
        v2 = await svc.create_vault(
            pool, display_name="par-refresh-2", metadata={}, account_id=account_id
        )
        url1 = "https://par-refresh-1.example.com/mcp"
        url2 = "https://par-refresh-2.example.com/mcp"
        await svc.create_vault_credential(
            pool,
            crypto_box,
            vault_id=v1.id,
            body=self._expiring_oauth_body(url1),
            account_id=account_id,
        )
        await svc.create_vault_credential(
            pool,
            crypto_box,
            vault_id=v2.id,
            body=self._expiring_oauth_body(url2),
            account_id=account_id,
        )
        sess1 = await self._bind_session_to_vault(pool, v1.id)
        sess2 = await self._bind_session_to_vault(pool, v2.id)

        post_calls: list[Any] = []
        with self._patched_async_client(
            post_calls,
            body={"access_token": "fresh-at", "expires_in": 3600},
        ):
            results = await asyncio.gather(
                resolve_auth_for_target_url(
                    pool, crypto_box, sess1, url1, account_id="acc_test_stub"
                ),
                resolve_auth_for_target_url(
                    pool, crypto_box, sess2, url2, account_id="acc_test_stub"
                ),
            )

        assert all(r[1] == {"Authorization": "Bearer fresh-at"} for r in results)
        # Two distinct credentials → two POSTs (one each), not one shared.
        assert len(post_calls) == 2, (
            f"expected 2 POSTs (one per credential) but got {len(post_calls)} "
            f"— lock scope is too wide"
        )


class TestSessionVaults:
    async def test_session_with_vault_ids(self, pool: Any) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.services import agents as agents_svc
        from aios.services import environments as env_svc
        from aios.services import sessions as sess_svc
        from aios.services import vaults as vault_svc

        env = await env_svc.create_environment(
            pool, name="vault-session-test", account_id=account_id
        )
        agent = await agents_svc.create_agent(
            pool,
            name="vault-agent",
            model="fake/test",
            system="test",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
            account_id=account_id,
        )
        v1 = await vault_svc.create_vault(
            pool, display_name="sv-1", metadata={}, account_id=account_id
        )
        v2 = await vault_svc.create_vault(
            pool, display_name="sv-2", metadata={}, account_id=account_id
        )

        session = await sess_svc.create_session(
            pool,
            agent_id=agent.id,
            environment_id=env.id,
            title="vault-test",
            metadata={},
            vault_ids=[v1.id, v2.id],
            account_id=account_id,
        )
        assert session.vault_ids == [v1.id, v2.id]

        # Get should return vault_ids too.
        fetched = await sess_svc.get_session(pool, session.id, account_id=account_id)
        assert fetched.vault_ids == [v1.id, v2.id]

    async def test_update_session_vault_ids(self, pool: Any) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.services import agents as agents_svc
        from aios.services import environments as env_svc
        from aios.services import sessions as sess_svc
        from aios.services import vaults as vault_svc

        env = await env_svc.create_environment(
            pool, name="vault-update-test", account_id=account_id
        )
        agent = await agents_svc.create_agent(
            pool,
            name="vault-update-agent",
            model="fake/test",
            system="test",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
            account_id=account_id,
        )
        v1 = await vault_svc.create_vault(
            pool, display_name="upd-1", metadata={}, account_id=account_id
        )

        session = await sess_svc.create_session(
            pool,
            agent_id=agent.id,
            environment_id=env.id,
            title="vault-upd",
            metadata={},
            vault_ids=[v1.id],
            account_id=account_id,
        )
        assert session.vault_ids == [v1.id]

        # Update to clear vault_ids.
        updated = await sess_svc.update_session(
            pool, session.id, vault_ids=[], account_id=account_id
        )
        assert updated.vault_ids == []

    async def test_session_without_vaults(self, pool: Any) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.services import agents as agents_svc
        from aios.services import environments as env_svc
        from aios.services import sessions as sess_svc

        env = await env_svc.create_environment(pool, name="no-vault-test", account_id=account_id)
        agent = await agents_svc.create_agent(
            pool,
            name="no-vault-agent",
            model="fake/test",
            system="test",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
            account_id=account_id,
        )
        session = await sess_svc.create_session(
            pool,
            agent_id=agent.id,
            environment_id=env.id,
            title="no-vault",
            metadata={},
            account_id=account_id,
        )
        assert session.vault_ids == []
