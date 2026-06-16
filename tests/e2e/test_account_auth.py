"""E2E tests for the account-aware bearer auth dep."""

from __future__ import annotations

import secrets
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from tests.helpers.connections import asgi_client


@pytest.fixture
async def http_client(pool: Any, aios_env: dict[str, str]) -> AsyncIterator[httpx.AsyncClient]:
    async with asgi_client(pool) as client:
        yield client


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


class TestAccountAuth:
    async def test_bootstrapped_key_authenticates(
        self, http_client: httpx.AsyncClient, aios_env: dict[str, str]
    ) -> None:
        """The plaintext key seeded into ``account_keys`` by ``aios_env``
        successfully authenticates an arbitrary protected route.
        """
        r = await http_client.get("/v1/agents", headers=_bearer(aios_env["AIOS_API_KEY"]))
        assert r.status_code == 200, r.text

    @pytest.mark.parametrize(
        "headers",
        [
            pytest.param({}, id="missing"),
            pytest.param(
                {"Authorization": f"Bearer aios_{secrets.token_urlsafe(32)}"},
                id="unknown-token",
            ),
            pytest.param({"Authorization": "Basic some-creds"}, id="wrong-scheme"),
        ],
    )
    async def test_header_shape_variants_401(
        self, http_client: httpx.AsyncClient, headers: dict[str, str]
    ) -> None:
        r = await http_client.get("/v1/agents", headers=headers)
        assert r.status_code == 401, r.text

    async def test_revoked_key_401(
        self, http_client: httpx.AsyncClient, aios_env: dict[str, str], pool: Any
    ) -> None:
        """Setting ``revoked_at`` on the matching key kicks the holder out."""
        from aios.services.accounts import hash_key

        plaintext = aios_env["AIOS_API_KEY"]
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE account_keys SET revoked_at = now() WHERE hash = $1",
                hash_key(plaintext),
            )
        r = await http_client.get("/v1/agents", headers=_bearer(plaintext))
        assert r.status_code == 401, r.text

    async def test_archived_account_401(
        self, http_client: httpx.AsyncClient, aios_env: dict[str, str], pool: Any
    ) -> None:
        """Archiving the owning account locks all its keys out, even
        without an explicit revoke on the key row.
        """
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE accounts SET archived_at = now() WHERE parent_account_id IS NULL"
            )
        r = await http_client.get("/v1/agents", headers=_bearer(aios_env["AIOS_API_KEY"]))
        assert r.status_code == 401, r.text

    async def test_aios_api_key_env_no_longer_consulted(
        self, http_client: httpx.AsyncClient, aios_env: dict[str, str], pool: Any
    ) -> None:
        """``AIOS_API_KEY`` env var alone doesn't unlock auth — the key
        must have a matching row in ``account_keys``. Delete the seeded
        row and confirm even the env-var value is rejected.
        """
        from aios.services.accounts import hash_key

        plaintext = aios_env["AIOS_API_KEY"]
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM account_keys WHERE hash = $1",
                hash_key(plaintext),
            )
        r = await http_client.get("/v1/agents", headers=_bearer(plaintext))
        assert r.status_code == 401, r.text
