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
async def pool(aios_env: dict[str, str]) -> Any:
    from aios.config import get_settings
    from aios.db.pool import create_pool

    settings = get_settings()
    p = await create_pool(settings.db_url, min_size=1, max_size=4)
    yield p
    await p.close()


@pytest.fixture
def crypto_box(aios_env: dict[str, str]) -> Any:
    from aios.config import get_settings
    from aios.crypto.vault import CryptoBox

    return CryptoBox.from_base64(get_settings().vault_key.get_secret_value())


class TestVaultCRUD:
    async def test_create_and_get(self, pool: Any) -> None:
        from aios.services import vaults as svc

        vault = await svc.create_vault(pool, display_name="test-vault", metadata={"env": "test"})
        assert vault.display_name == "test-vault"
        assert vault.metadata == {"env": "test"}
        assert vault.id.startswith("vlt_")
        assert vault.archived_at is None

        fetched = await svc.get_vault(pool, vault.id)
        assert fetched.id == vault.id
        assert fetched.display_name == "test-vault"

    async def test_list_vaults(self, pool: Any) -> None:
        from aios.services import vaults as svc

        v1 = await svc.create_vault(pool, display_name="list-a", metadata={})
        v2 = await svc.create_vault(pool, display_name="list-b", metadata={})
        vaults = await svc.list_vaults(pool, limit=100)
        ids = [v.id for v in vaults]
        assert v1.id in ids
        assert v2.id in ids

    async def test_update_vault(self, pool: Any) -> None:
        from aios.services import vaults as svc

        vault = await svc.create_vault(pool, display_name="before", metadata={})
        updated = await svc.update_vault(pool, vault.id, display_name="after")
        assert updated.display_name == "after"
        assert updated.updated_at > vault.updated_at

    async def test_archive_vault(self, pool: Any) -> None:
        from aios.services import vaults as svc

        vault = await svc.create_vault(pool, display_name="to-archive", metadata={})
        archived = await svc.archive_vault(pool, vault.id)
        assert archived.archived_at is not None

        # Archived vaults don't appear in list
        vaults = await svc.list_vaults(pool, limit=100)
        assert vault.id not in [v.id for v in vaults]

    async def test_delete_vault(self, pool: Any) -> None:
        from aios.services import vaults as svc

        vault = await svc.create_vault(pool, display_name="to-delete", metadata={})
        await svc.delete_vault(pool, vault.id)

        from aios.errors import NotFoundError

        with pytest.raises(NotFoundError):
            await svc.get_vault(pool, vault.id)


class TestVaultCredentialCRUD:
    async def test_create_static_bearer(self, pool: Any, crypto_box: Any) -> None:
        from aios.services import vaults as svc

        vault = await svc.create_vault(pool, display_name="cred-test", metadata={})
        body = VaultCredentialCreate(
            mcp_server_url="https://mcp.example.com/api",
            auth_type="static_bearer",
            token=SecretStr("my-secret-token"),
        )
        cred = await svc.create_vault_credential(pool, crypto_box, vault_id=vault.id, body=body)
        assert cred.id.startswith("vcr_")
        assert cred.vault_id == vault.id
        assert cred.mcp_server_url == "https://mcp.example.com/api"
        assert cred.auth_type == "static_bearer"

    async def test_secrets_not_returned(self, pool: Any, crypto_box: Any) -> None:
        """Verify that the read view never includes secret fields."""
        from aios.services import vaults as svc

        vault = await svc.create_vault(pool, display_name="secrets-test", metadata={})
        body = VaultCredentialCreate(
            mcp_server_url="https://mcp2.example.com",
            auth_type="static_bearer",
            token=SecretStr("super-secret"),
        )
        cred = await svc.create_vault_credential(pool, crypto_box, vault_id=vault.id, body=body)

        fetched = await svc.get_vault_credential(pool, vault.id, cred.id)
        dumped = fetched.model_dump()
        assert "token" not in dumped
        assert "access_token" not in dumped
        assert "ciphertext" not in dumped

    async def test_create_mcp_oauth(self, pool: Any, crypto_box: Any) -> None:
        from aios.services import vaults as svc

        vault = await svc.create_vault(pool, display_name="oauth-test", metadata={})
        body = VaultCredentialCreate(
            mcp_server_url="https://oauth.example.com",
            auth_type="mcp_oauth",
            access_token=SecretStr("access-123"),
            client_id="client-abc",
            refresh_token=SecretStr("refresh-456"),
            token_endpoint="https://oauth.example.com/token",
        )
        cred = await svc.create_vault_credential(pool, crypto_box, vault_id=vault.id, body=body)
        assert cred.auth_type == "mcp_oauth"

    async def test_static_bearer_requires_token(self, pool: Any, crypto_box: Any) -> None:
        from aios.errors import ValidationError
        from aios.services import vaults as svc

        vault = await svc.create_vault(pool, display_name="req-test", metadata={})
        body = VaultCredentialCreate(
            mcp_server_url="https://no-token.example.com",
            auth_type="static_bearer",
        )
        with pytest.raises(ValidationError, match="require"):
            await svc.create_vault_credential(pool, crypto_box, vault_id=vault.id, body=body)

    async def test_oauth_requires_access_token(self, pool: Any, crypto_box: Any) -> None:
        from aios.errors import ValidationError
        from aios.services import vaults as svc

        vault = await svc.create_vault(pool, display_name="oauth-req", metadata={})
        body = VaultCredentialCreate(
            mcp_server_url="https://no-at.example.com",
            auth_type="mcp_oauth",
        )
        with pytest.raises(ValidationError, match="require"):
            await svc.create_vault_credential(pool, crypto_box, vault_id=vault.id, body=body)

    async def test_unique_url_per_vault(self, pool: Any, crypto_box: Any) -> None:
        """Two active credentials for the same URL in the same vault should conflict."""
        from aios.errors import ConflictError
        from aios.services import vaults as svc

        vault = await svc.create_vault(pool, display_name="uniq-test", metadata={})
        url = "https://unique-url-test.example.com"
        body = VaultCredentialCreate(
            mcp_server_url=url, auth_type="static_bearer", token=SecretStr("t1")
        )
        await svc.create_vault_credential(pool, crypto_box, vault_id=vault.id, body=body)
        with pytest.raises(ConflictError):
            await svc.create_vault_credential(pool, crypto_box, vault_id=vault.id, body=body)

    async def test_archive_frees_url(self, pool: Any, crypto_box: Any) -> None:
        """Archiving a credential frees its mcp_server_url for reuse."""
        from aios.services import vaults as svc

        vault = await svc.create_vault(pool, display_name="archive-free", metadata={})
        url = "https://archive-free-test.example.com"
        body = VaultCredentialCreate(
            mcp_server_url=url, auth_type="static_bearer", token=SecretStr("t1")
        )
        cred1 = await svc.create_vault_credential(pool, crypto_box, vault_id=vault.id, body=body)
        await svc.archive_vault_credential(pool, vault.id, cred1.id)

        # Now creating a new credential for the same URL should succeed.
        cred2 = await svc.create_vault_credential(pool, crypto_box, vault_id=vault.id, body=body)
        assert cred2.id != cred1.id

    async def test_update_credential(self, pool: Any, crypto_box: Any) -> None:
        from aios.services import vaults as svc

        vault = await svc.create_vault(pool, display_name="update-test", metadata={})
        body = VaultCredentialCreate(
            mcp_server_url="https://update-test.example.com",
            auth_type="static_bearer",
            token=SecretStr("old-token"),
            display_name="original",
        )
        cred = await svc.create_vault_credential(pool, crypto_box, vault_id=vault.id, body=body)

        update = VaultCredentialUpdate(
            display_name="updated",
            token=SecretStr("new-token"),
        )
        updated = await svc.update_vault_credential(
            pool, crypto_box, vault_id=vault.id, credential_id=cred.id, body=update
        )
        assert updated.display_name == "updated"
        assert updated.updated_at > cred.updated_at

    async def test_credential_limit(self, pool: Any, crypto_box: Any) -> None:
        """Vault cannot have more than 20 active credentials."""
        from aios.errors import ValidationError
        from aios.services import vaults as svc

        vault = await svc.create_vault(pool, display_name="limit-test", metadata={})
        for i in range(20):
            body = VaultCredentialCreate(
                mcp_server_url=f"https://limit-{i}.example.com",
                auth_type="static_bearer",
                token=SecretStr(f"t-{i}"),
            )
            await svc.create_vault_credential(pool, crypto_box, vault_id=vault.id, body=body)

        body21 = VaultCredentialCreate(
            mcp_server_url="https://limit-overflow.example.com",
            auth_type="static_bearer",
            token=SecretStr("overflow"),
        )
        with pytest.raises(ValidationError, match="maximum"):
            await svc.create_vault_credential(pool, crypto_box, vault_id=vault.id, body=body21)

    async def test_credential_limit_under_concurrency(self, pool: Any, crypto_box: Any) -> None:
        """The 20-cred limit holds under concurrent inserts.

        Without ``SELECT … FOR UPDATE`` on the vault row, two parallel
        inserts can both observe ``count == 19`` and both succeed,
        overflowing the cap. With the row lock, exactly 20 succeed and the
        rest get ``ValidationError``.
        """
        from aios.errors import ValidationError
        from aios.services import vaults as svc

        vault = await svc.create_vault(pool, display_name="race-test", metadata={})

        async def attempt(i: int) -> Any:
            body = VaultCredentialCreate(
                mcp_server_url=f"https://race-{i}.example.com",
                auth_type="static_bearer",
                token=SecretStr(f"t-{i}"),
            )
            try:
                return await svc.create_vault_credential(
                    pool, crypto_box, vault_id=vault.id, body=body
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


class TestQueries:
    """Direct tests for query-layer functions used internally by services."""

    async def test_get_credential_with_blob_returns_both(self, pool: Any, crypto_box: Any) -> None:
        from aios.db import queries
        from aios.services import vaults as svc

        vault = await svc.create_vault(pool, display_name="combo-test", metadata={})
        body = VaultCredentialCreate(
            mcp_server_url="https://combo.example.com",
            auth_type="static_bearer",
            token=SecretStr("combo-token"),
        )
        cred = await svc.create_vault_credential(pool, crypto_box, vault_id=vault.id, body=body)

        async with pool.acquire() as conn:
            fetched_cred, blob = await queries.get_vault_credential_with_blob(
                conn, vault.id, cred.id
            )

        assert fetched_cred.id == cred.id
        assert fetched_cred.auth_type == "static_bearer"
        assert blob.ciphertext  # non-empty
        assert blob.nonce  # non-empty
        # Verify the blob actually decrypts to the original payload.
        import json as _json

        payload = _json.loads(crypto_box.decrypt(blob))
        assert payload == {"token": "combo-token"}

    async def test_get_credential_with_blob_excludes_archived(
        self, pool: Any, crypto_box: Any
    ) -> None:
        from aios.db import queries
        from aios.errors import NotFoundError
        from aios.services import vaults as svc

        vault = await svc.create_vault(pool, display_name="combo-arch", metadata={})
        body = VaultCredentialCreate(
            mcp_server_url="https://combo-arch.example.com",
            auth_type="static_bearer",
            token=SecretStr("doomed"),
        )
        cred = await svc.create_vault_credential(pool, crypto_box, vault_id=vault.id, body=body)
        await svc.archive_vault_credential(pool, vault.id, cred.id)

        async with pool.acquire() as conn:
            with pytest.raises(NotFoundError, match="archived"):
                await queries.get_vault_credential_with_blob(conn, vault.id, cred.id)


class TestSessionVaults:
    async def test_session_with_vault_ids(self, pool: Any) -> None:
        from aios.services import agents as agents_svc
        from aios.services import environments as env_svc
        from aios.services import sessions as sess_svc
        from aios.services import vaults as vault_svc

        env = await env_svc.create_environment(pool, name="vault-session-test")
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
        )
        v1 = await vault_svc.create_vault(pool, display_name="sv-1", metadata={})
        v2 = await vault_svc.create_vault(pool, display_name="sv-2", metadata={})

        session = await sess_svc.create_session(
            pool,
            agent_id=agent.id,
            environment_id=env.id,
            title="vault-test",
            metadata={},
            vault_ids=[v1.id, v2.id],
        )
        assert session.vault_ids == [v1.id, v2.id]

        # Get should return vault_ids too.
        fetched = await sess_svc.get_session(pool, session.id)
        assert fetched.vault_ids == [v1.id, v2.id]

    async def test_update_session_vault_ids(self, pool: Any) -> None:
        from aios.services import agents as agents_svc
        from aios.services import environments as env_svc
        from aios.services import sessions as sess_svc
        from aios.services import vaults as vault_svc

        env = await env_svc.create_environment(pool, name="vault-update-test")
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
        )
        v1 = await vault_svc.create_vault(pool, display_name="upd-1", metadata={})

        session = await sess_svc.create_session(
            pool,
            agent_id=agent.id,
            environment_id=env.id,
            title="vault-upd",
            metadata={},
            vault_ids=[v1.id],
        )
        assert session.vault_ids == [v1.id]

        # Update to clear vault_ids.
        updated = await sess_svc.update_session(pool, session.id, vault_ids=[])
        assert updated.vault_ids == []

    async def test_session_without_vaults(self, pool: Any) -> None:
        from aios.services import agents as agents_svc
        from aios.services import environments as env_svc
        from aios.services import sessions as sess_svc

        env = await env_svc.create_environment(pool, name="no-vault-test")
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
        )
        session = await sess_svc.create_session(
            pool,
            agent_id=agent.id,
            environment_id=env.id,
            title="no-vault",
            metadata={},
        )
        assert session.vault_ids == []
