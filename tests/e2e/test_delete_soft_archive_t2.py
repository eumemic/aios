"""E2E proof of the T2 DELETE-semantics normalization (#1463).

Across the four formerly-hard-delete families — vaults, vault credentials,
memory stores, and sessions — bare HTTP ``DELETE`` must now perform a
*soft-archive* (sets ``archived_at``, row + history retained), and the
prior hard-delete must be reachable only via the explicit ``POST /purge``
verb (cascade / host-mirror cleanup / secret-zeroing preserved).
"""

from __future__ import annotations

import secrets
from collections.abc import AsyncIterator
from typing import Any
from unittest import mock

import httpx
import pytest

from tests.helpers.connections import authed_client, wired_app


def _uniq() -> str:
    return secrets.token_hex(4)


@pytest.fixture
async def http_client(pool: Any, aios_env: dict[str, str]) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=wired_app(pool))
    with mock.patch("aios.api.routers.sessions.defer_wake", new_callable=mock.AsyncMock):
        async with authed_client(
            "http://testserver",
            aios_env["AIOS_API_KEY"],
            transport=transport,
        ) as client:
            yield client


class TestVaultDeleteIsSoft:
    async def test_delete_soft_archives_purge_hard_deletes(
        self, http_client: httpx.AsyncClient
    ) -> None:
        r = await http_client.post("/v1/vaults", json={"display_name": f"v-{_uniq()}"})
        assert r.status_code == 201, r.text
        vault_id = r.json()["id"]

        # Bare DELETE → soft-archive. Row persists, fetchable, archived_at set.
        r = await http_client.delete(f"/v1/vaults/{vault_id}")
        assert r.status_code == 200, r.text
        assert r.json()["archived_at"] is not None

        r = await http_client.get(f"/v1/vaults/{vault_id}")
        assert r.status_code == 200, r.text
        assert r.json()["archived_at"] is not None

        # Hidden from default list (history retained, not destroyed).
        r = await http_client.get("/v1/vaults")
        assert vault_id not in [v["id"] for v in r.json()["data"]]

        # Explicit /purge → hard-delete.
        r = await http_client.post(f"/v1/vaults/{vault_id}/purge")
        assert r.status_code == 204, r.text

        r = await http_client.get(f"/v1/vaults/{vault_id}")
        assert r.status_code == 404, r.text


class TestVaultCredentialDeleteIsSoft:
    async def test_delete_soft_archives_purge_hard_deletes(
        self, http_client: httpx.AsyncClient
    ) -> None:
        r = await http_client.post("/v1/vaults", json={"display_name": f"v-{_uniq()}"})
        assert r.status_code == 201, r.text
        vault_id = r.json()["id"]

        r = await http_client.post(
            f"/v1/vaults/{vault_id}/credentials",
            json={
                "target_url": "https://mcp.example.com/api",
                "auth_type": "bearer_header",
                "token": "secret-token",
            },
        )
        assert r.status_code == 201, r.text
        cred_id = r.json()["id"]

        # Bare DELETE → soft-archive (secret zeroed, row retained for audit).
        r = await http_client.delete(f"/v1/vaults/{vault_id}/credentials/{cred_id}")
        assert r.status_code == 200, r.text
        assert r.json()["archived_at"] is not None

        r = await http_client.get(f"/v1/vaults/{vault_id}/credentials/{cred_id}")
        assert r.status_code == 200, r.text
        assert r.json()["archived_at"] is not None

        # Explicit /purge → hard-delete.
        r = await http_client.post(f"/v1/vaults/{vault_id}/credentials/{cred_id}/purge")
        assert r.status_code == 204, r.text

        r = await http_client.get(f"/v1/vaults/{vault_id}/credentials/{cred_id}")
        assert r.status_code == 404, r.text


class TestSessionDeleteIsSoft:
    @pytest.fixture
    async def session_id(self, pool: Any) -> str:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.db import queries
        from aios.services import agents as agents_svc
        from aios.services import sessions as sessions_svc

        async with pool.acquire() as conn:
            env = await queries.insert_environment(
                conn, name=f"del-env-{_uniq()}", account_id=account_id
            )
        agent = await agents_svc.create_agent(
            pool,
            name=f"del-agent-{_uniq()}",
            model="openai/gpt-4o-mini",
            system="",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
            account_id=account_id,
        )
        session = await sessions_svc.create_session(
            pool,
            agent_id=agent.id,
            environment_id=env.id,
            title="to-delete",
            metadata={},
            account_id=account_id,
        )
        return session.id

    async def test_delete_soft_archives_purge_hard_deletes(
        self, http_client: httpx.AsyncClient, session_id: str
    ) -> None:
        # Bare DELETE → soft-archive. Row persists, archived_at set.
        r = await http_client.delete(f"/v1/sessions/{session_id}")
        assert r.status_code == 200, r.text
        assert r.json()["archived_at"] is not None

        r = await http_client.get(f"/v1/sessions/{session_id}")
        assert r.status_code == 200, r.text
        assert r.json()["archived_at"] is not None

        # Explicit /purge → hard-delete (cascade events/vaults/bindings).
        r = await http_client.post(f"/v1/sessions/{session_id}/purge")
        assert r.status_code == 204, r.text

        r = await http_client.get(f"/v1/sessions/{session_id}")
        assert r.status_code == 404, r.text


class TestBareDeleteIsIdempotent:
    """A *second* bare DELETE must return 200 + the existing archived row, not
    404, across all four archive families (#1492).

    #1486 threaded ``idempotent=True`` so the repeat-DELETE-is-200 contract
    held for memory-stores; this proves the same contract now holds for
    sessions, vaults, and vault credentials — the asymmetry #1492 closes.
    """

    @pytest.fixture
    async def session_id(self, pool: Any) -> str:
        account_id = "acc_test_stub"
        from aios.db import queries
        from aios.services import agents as agents_svc
        from aios.services import sessions as sessions_svc

        async with pool.acquire() as conn:
            env = await queries.insert_environment(
                conn, name=f"idem-env-{_uniq()}", account_id=account_id
            )
        agent = await agents_svc.create_agent(
            pool,
            name=f"idem-agent-{_uniq()}",
            model="openai/gpt-4o-mini",
            system="",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
            account_id=account_id,
        )
        session = await sessions_svc.create_session(
            pool,
            agent_id=agent.id,
            environment_id=env.id,
            title="idem-delete",
            metadata={},
            account_id=account_id,
        )
        return session.id

    async def test_second_delete_memory_store_is_200(self, http_client: httpx.AsyncClient) -> None:
        r = await http_client.post("/v1/memory-stores", json={"name": f"ms-{_uniq()}"})
        assert r.status_code == 201, r.text
        store_id = r.json()["id"]

        r1 = await http_client.delete(f"/v1/memory-stores/{store_id}")
        assert r1.status_code == 200, r1.text
        assert r1.json()["archived_at"] is not None

        r2 = await http_client.delete(f"/v1/memory-stores/{store_id}")
        assert r2.status_code == 200, r2.text
        assert r2.json()["archived_at"] is not None
        # Idempotent: the second DELETE reports the *same* archived row.
        assert r2.json()["archived_at"] == r1.json()["archived_at"]

    async def test_second_delete_vault_is_200(self, http_client: httpx.AsyncClient) -> None:
        r = await http_client.post("/v1/vaults", json={"display_name": f"v-{_uniq()}"})
        assert r.status_code == 201, r.text
        vault_id = r.json()["id"]

        r1 = await http_client.delete(f"/v1/vaults/{vault_id}")
        assert r1.status_code == 200, r1.text
        assert r1.json()["archived_at"] is not None

        r2 = await http_client.delete(f"/v1/vaults/{vault_id}")
        assert r2.status_code == 200, r2.text
        assert r2.json()["archived_at"] is not None
        assert r2.json()["archived_at"] == r1.json()["archived_at"]

    async def test_second_delete_vault_credential_is_200(
        self, http_client: httpx.AsyncClient
    ) -> None:
        r = await http_client.post("/v1/vaults", json={"display_name": f"v-{_uniq()}"})
        assert r.status_code == 201, r.text
        vault_id = r.json()["id"]

        r = await http_client.post(
            f"/v1/vaults/{vault_id}/credentials",
            json={
                "target_url": "https://mcp.example.com/api",
                "auth_type": "bearer_header",
                "token": "secret-token",
            },
        )
        assert r.status_code == 201, r.text
        cred_id = r.json()["id"]

        r1 = await http_client.delete(f"/v1/vaults/{vault_id}/credentials/{cred_id}")
        assert r1.status_code == 200, r1.text
        assert r1.json()["archived_at"] is not None

        r2 = await http_client.delete(f"/v1/vaults/{vault_id}/credentials/{cred_id}")
        assert r2.status_code == 200, r2.text
        assert r2.json()["archived_at"] is not None
        assert r2.json()["archived_at"] == r1.json()["archived_at"]

    async def test_second_delete_session_is_200(
        self, http_client: httpx.AsyncClient, session_id: str
    ) -> None:
        r1 = await http_client.delete(f"/v1/sessions/{session_id}")
        assert r1.status_code == 200, r1.text
        assert r1.json()["archived_at"] is not None

        r2 = await http_client.delete(f"/v1/sessions/{session_id}")
        assert r2.status_code == 200, r2.text
        assert r2.json()["archived_at"] is not None
        assert r2.json()["archived_at"] == r1.json()["archived_at"]
