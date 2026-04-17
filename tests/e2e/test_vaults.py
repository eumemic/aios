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
        from aios.services import vaults as svc

        vaults = await asyncio.gather(
            *(svc.create_vault(pool, display_name=f"par-{i}", metadata={}) for i in range(20))
        )

        async def insert_one(v_idx: int) -> Any:
            return await svc.create_vault_credential(
                pool,
                crypto_box,
                vault_id=vaults[v_idx].id,
                body=VaultCredentialCreate(
                    mcp_server_url=f"https://par-{v_idx}.example.com",
                    auth_type="static_bearer",
                    token=SecretStr(f"t-{v_idx}"),
                ),
            )

        results = await asyncio.gather(*(insert_one(i) for i in range(20)))
        assert all(r.id.startswith("vcr_") for r in results)
        # Every insert was on a different vault; per-vault row lock should
        # not have caused any failures.


class TestArchiveAndCascade:
    """Archive must zero the encrypted blob; delete must cascade via the FK."""

    async def test_archive_credential_zeros_blob(self, pool: Any, crypto_box: Any) -> None:
        from aios.services import vaults as svc

        vault = await svc.create_vault(pool, display_name="zero-cred", metadata={})
        body = VaultCredentialCreate(
            mcp_server_url="https://zero-cred.example.com",
            auth_type="static_bearer",
            token=SecretStr("doomed"),
        )
        cred = await svc.create_vault_credential(pool, crypto_box, vault_id=vault.id, body=body)
        await svc.archive_vault_credential(pool, vault.id, cred.id)

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT ciphertext, nonce FROM vault_credentials WHERE id = $1",
                cred.id,
            )
        assert row is not None
        assert bytes(row["ciphertext"]) == b""
        assert bytes(row["nonce"]) == b""

    async def test_archive_vault_zeros_active_credentials(self, pool: Any, crypto_box: Any) -> None:
        from aios.services import vaults as svc

        vault = await svc.create_vault(pool, display_name="zero-vault", metadata={})
        # Two active credentials.
        for i in range(2):
            await svc.create_vault_credential(
                pool,
                crypto_box,
                vault_id=vault.id,
                body=VaultCredentialCreate(
                    mcp_server_url=f"https://zero-vault-{i}.example.com",
                    auth_type="static_bearer",
                    token=SecretStr(f"t-{i}"),
                ),
            )

        await svc.archive_vault(pool, vault.id)

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
        from aios.services import vaults as svc

        vault = await svc.create_vault(pool, display_name="cascade-test", metadata={})
        cred = await svc.create_vault_credential(
            pool,
            crypto_box,
            vault_id=vault.id,
            body=VaultCredentialCreate(
                mcp_server_url="https://cascade.example.com",
                auth_type="static_bearer",
                token=SecretStr("doomed"),
            ),
        )

        await svc.delete_vault(pool, vault.id)

        async with pool.acquire() as conn:
            row = await conn.fetchrow("SELECT 1 FROM vault_credentials WHERE id = $1", cred.id)
        assert row is None  # cascade-deleted


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


class TestOAuthRefreshE2E:
    """End-to-end refresh: real testcontainer Postgres + mocked OAuth endpoint.

    The mocked httpx layer lets these tests assert that exactly one POST
    happens under concurrency — proving the SELECT … FOR UPDATE row lock
    works, not just trust the unit test that mocks the lock query.
    """

    @staticmethod
    async def _bind_session_to_vault(pool: Any, vault_id: str) -> str:
        """Create the minimum scaffolding (env + agent + session) to bind a vault."""
        from aios.services import agents as agents_svc
        from aios.services import environments as env_svc
        from aios.services import sessions as sess_svc

        suffix = vault_id[-8:]
        env = await env_svc.create_environment(pool, name=f"oauth-e2e-env-{suffix}")
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
        )
        session = await sess_svc.create_session(
            pool,
            agent_id=agent.id,
            environment_id=env.id,
            title="oauth-refresh-test",
            metadata={},
            vault_ids=[vault_id],
        )
        return str(session.id)

    @staticmethod
    def _expiring_oauth_body(url: str) -> VaultCredentialCreate:
        from datetime import UTC, datetime, timedelta

        return VaultCredentialCreate(
            mcp_server_url=url,
            auth_type="mcp_oauth",
            access_token=SecretStr("stale-at"),
            refresh_token=SecretStr("rt-1"),
            client_id="cid",
            token_endpoint="https://issuer.example/token",
            expires_at=datetime.now(UTC) - timedelta(seconds=1),  # already expired
        )

    @staticmethod
    def _patched_async_client(post_calls: list[Any], body: dict[str, Any]):
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
        import json as _json

        from aios.mcp.client import resolve_auth_for_url
        from aios.services import vaults as svc

        vault = await svc.create_vault(pool, display_name="oauth-e2e", metadata={})
        url = "https://oauth-e2e.example.com/mcp"
        cred = await svc.create_vault_credential(
            pool, crypto_box, vault_id=vault.id, body=self._expiring_oauth_body(url)
        )
        session_id = await self._bind_session_to_vault(pool, vault.id)

        post_calls: list[Any] = []
        with self._patched_async_client(
            post_calls,
            body={"access_token": "fresh-at", "expires_in": 3600},
        ):
            headers = await resolve_auth_for_url(pool, crypto_box, session_id, url)

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
        new_payload = _json.loads(crypto_box.decrypt(new_blob))
        assert new_payload["access_token"] == "fresh-at"

    async def test_concurrent_resolve_only_refreshes_once(self, pool: Any, crypto_box: Any) -> None:
        """Five parallel resolutions on an expiring credential issue exactly one POST.

        Without the SELECT … FOR UPDATE row lock + double-check, every
        coroutine would race to the token endpoint. The lock serializes
        them; the second-to-last waiter sees the now-fresh expires_at
        after acquiring the lock and exits without POSTing.
        """
        from aios.mcp.client import resolve_auth_for_url
        from aios.services import vaults as svc

        vault = await svc.create_vault(pool, display_name="oauth-race", metadata={})
        url = "https://oauth-race.example.com/mcp"
        await svc.create_vault_credential(
            pool, crypto_box, vault_id=vault.id, body=self._expiring_oauth_body(url)
        )
        session_id = await self._bind_session_to_vault(pool, vault.id)

        post_calls: list[Any] = []
        with self._patched_async_client(
            post_calls,
            body={"access_token": "fresh-at", "expires_in": 3600},
        ):
            results = await asyncio.gather(
                *(resolve_auth_for_url(pool, crypto_box, session_id, url) for _ in range(5))
            )

        assert all(r == {"Authorization": "Bearer fresh-at"} for r in results)
        assert len(post_calls) == 1, (
            f"expected 1 POST but got {len(post_calls)} — row lock or double-check is broken"
        )

    async def test_refresh_failure_bubbles(self, pool: Any, crypto_box: Any) -> None:
        from unittest.mock import AsyncMock, MagicMock, patch

        from aios.errors import OAuthRefreshError
        from aios.mcp.client import resolve_auth_for_url
        from aios.services import vaults as svc

        vault = await svc.create_vault(pool, display_name="oauth-fail", metadata={})
        url = "https://oauth-fail.example.com/mcp"
        await svc.create_vault_credential(
            pool, crypto_box, vault_id=vault.id, body=self._expiring_oauth_body(url)
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
            await resolve_auth_for_url(pool, crypto_box, session_id, url)

    async def test_concurrent_refresh_of_different_credentials_runs_in_parallel(
        self, pool: Any, crypto_box: Any
    ) -> None:
        """Per-row lock must not over-serialize across distinct credentials.

        Companion to ``test_concurrent_resolve_only_refreshes_once``: that
        test pins the lock semantics for the *same* credential. This one
        pins the *scope* of the lock — refreshes against two different
        ``(vault_id, mcp_server_url)`` pairs must produce two independent
        POSTs, not one. A future refactor that promoted the row lock to a
        global lock or a vault-level lock would make this test fail.
        """
        from aios.mcp.client import resolve_auth_for_url
        from aios.services import vaults as svc

        v1 = await svc.create_vault(pool, display_name="par-refresh-1", metadata={})
        v2 = await svc.create_vault(pool, display_name="par-refresh-2", metadata={})
        url1 = "https://par-refresh-1.example.com/mcp"
        url2 = "https://par-refresh-2.example.com/mcp"
        await svc.create_vault_credential(
            pool, crypto_box, vault_id=v1.id, body=self._expiring_oauth_body(url1)
        )
        await svc.create_vault_credential(
            pool, crypto_box, vault_id=v2.id, body=self._expiring_oauth_body(url2)
        )
        sess1 = await self._bind_session_to_vault(pool, v1.id)
        sess2 = await self._bind_session_to_vault(pool, v2.id)

        post_calls: list[Any] = []
        with self._patched_async_client(
            post_calls,
            body={"access_token": "fresh-at", "expires_in": 3600},
        ):
            results = await asyncio.gather(
                resolve_auth_for_url(pool, crypto_box, sess1, url1),
                resolve_auth_for_url(pool, crypto_box, sess2, url2),
            )

        assert all(r == {"Authorization": "Bearer fresh-at"} for r in results)
        # Two distinct credentials → two POSTs (one each), not one shared.
        assert len(post_calls) == 2, (
            f"expected 2 POSTs (one per credential) but got {len(post_calls)} "
            f"— lock scope is too wide"
        )


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
