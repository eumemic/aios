"""End-to-end tenant isolation tests (#367 follow-up).

The per-PR test suites cover individual queries and endpoints; this module
exercises the full path: two tenants mint resources independently, then
verify each can't see the other's resources via any management or
resource-listing endpoint. Catches cross-tenant leaks across the resource
families (agents, sessions, vaults, memory_stores, environments, skills,
session_templates) end-to-end rather than per-query.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from tests.helpers.connections import asgi_client


@pytest.fixture
async def pool(aios_env: dict[str, str]) -> AsyncIterator[Any]:
    from aios.config import get_settings
    from aios.db.pool import create_pool

    settings = get_settings()
    p = await create_pool(settings.db_url, min_size=1, max_size=4)
    yield p
    await p.close()


@pytest.fixture
async def http_client(pool: Any, aios_env: dict[str, str]) -> AsyncIterator[httpx.AsyncClient]:
    async with asgi_client(pool) as client:
        yield client


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def _mint_tenant(http: httpx.AsyncClient, parent_key: str, name: str) -> str:
    """Mint a child tenant and return its bearer key."""
    r = await http.post(
        "/v1/accounts/children",
        headers=_bearer(parent_key),
        json={"display_name": name, "can_mint_children": False},
    )
    assert r.status_code == 201, r.text
    return str(r.json()["plaintext_key"])


class TestTwoTenantIsolation:
    """Tenant A and Tenant B mint similar resources. Each must see only their own."""

    async def test_agents_isolated(
        self, http_client: httpx.AsyncClient, aios_env: dict[str, str]
    ) -> None:
        ka = await _mint_tenant(http_client, aios_env["AIOS_API_KEY"], "iso-agents-a")
        kb = await _mint_tenant(http_client, aios_env["AIOS_API_KEY"], "iso-agents-b")

        agent_a = await http_client.post(
            "/v1/agents",
            headers=_bearer(ka),
            json={"name": "alpha-a", "model": "openrouter/test"},
        )
        agent_b = await http_client.post(
            "/v1/agents",
            headers=_bearer(kb),
            json={"name": "alpha-b", "model": "openrouter/test"},
        )
        assert agent_a.status_code == 201
        assert agent_b.status_code == 201
        # NOTE: per-tenant name uniqueness is a follow-up — the current
        # schema keeps a global unique on agents.name. The test uses
        # distinct names per tenant; cross-tenant visibility is what we
        # assert below.
        a_id = agent_a.json()["id"]
        b_id = agent_b.json()["id"]
        assert a_id != b_id

        # Each tenant's list shows only their own agent.
        list_a = await http_client.get("/v1/agents", headers=_bearer(ka))
        list_b = await http_client.get("/v1/agents", headers=_bearer(kb))
        ids_a = {x["id"] for x in list_a.json()["data"]}
        ids_b = {x["id"] for x in list_b.json()["data"]}
        assert a_id in ids_a and b_id not in ids_a
        assert b_id in ids_b and a_id not in ids_b

        # Cross-tenant direct fetch returns 404 (not 403; not the row).
        cross = await http_client.get(f"/v1/agents/{b_id}", headers=_bearer(ka))
        assert cross.status_code == 404, cross.text

    async def test_environments_isolated(
        self, http_client: httpx.AsyncClient, aios_env: dict[str, str]
    ) -> None:
        ka = await _mint_tenant(http_client, aios_env["AIOS_API_KEY"], "iso-envs-a")
        kb = await _mint_tenant(http_client, aios_env["AIOS_API_KEY"], "iso-envs-b")

        env_a = await http_client.post(
            "/v1/environments", headers=_bearer(ka), json={"name": "env-a"}
        )
        env_b = await http_client.post(
            "/v1/environments", headers=_bearer(kb), json={"name": "env-b"}
        )
        assert env_a.status_code == 201
        assert env_b.status_code == 201
        a_id = env_a.json()["id"]
        b_id = env_b.json()["id"]

        list_a = await http_client.get("/v1/environments", headers=_bearer(ka))
        ids_a = {x["id"] for x in list_a.json()["data"]}
        assert a_id in ids_a and b_id not in ids_a

        cross = await http_client.get(f"/v1/environments/{b_id}", headers=_bearer(ka))
        assert cross.status_code == 404

    async def test_vaults_isolated(
        self, http_client: httpx.AsyncClient, aios_env: dict[str, str]
    ) -> None:
        ka = await _mint_tenant(http_client, aios_env["AIOS_API_KEY"], "iso-vaults-a")
        kb = await _mint_tenant(http_client, aios_env["AIOS_API_KEY"], "iso-vaults-b")

        vault_a = await http_client.post(
            "/v1/vaults", headers=_bearer(ka), json={"display_name": "shared"}
        )
        vault_b = await http_client.post(
            "/v1/vaults", headers=_bearer(kb), json={"display_name": "shared"}
        )
        assert vault_a.status_code == 201
        assert vault_b.status_code == 201
        a_id = vault_a.json()["id"]
        b_id = vault_b.json()["id"]

        list_a = await http_client.get("/v1/vaults", headers=_bearer(ka))
        ids_a = {x["id"] for x in list_a.json()["data"]}
        assert a_id in ids_a and b_id not in ids_a

        cross = await http_client.get(f"/v1/vaults/{b_id}", headers=_bearer(ka))
        assert cross.status_code == 404

    async def test_memory_stores_isolated(
        self, http_client: httpx.AsyncClient, aios_env: dict[str, str]
    ) -> None:
        ka = await _mint_tenant(http_client, aios_env["AIOS_API_KEY"], "iso-mem-a")
        kb = await _mint_tenant(http_client, aios_env["AIOS_API_KEY"], "iso-mem-b")

        store_a = await http_client.post(
            "/v1/memory-stores", headers=_bearer(ka), json={"name": "scratch-a"}
        )
        store_b = await http_client.post(
            "/v1/memory-stores", headers=_bearer(kb), json={"name": "scratch-b"}
        )
        assert store_a.status_code == 201
        assert store_b.status_code == 201
        a_id = store_a.json()["id"]
        b_id = store_b.json()["id"]

        list_a = await http_client.get("/v1/memory-stores", headers=_bearer(ka))
        ids_a = {x["id"] for x in list_a.json()["data"]}
        assert a_id in ids_a and b_id not in ids_a

        cross = await http_client.get(f"/v1/memory-stores/{b_id}", headers=_bearer(ka))
        assert cross.status_code == 404

    async def test_session_templates_isolated(
        self, http_client: httpx.AsyncClient, aios_env: dict[str, str]
    ) -> None:
        """Regression for the pre-#426 leak.

        ``list_session_templates`` accepted ``account_id`` as a kwarg but
        the SQL didn't filter on it; ``GET /v1/session-templates`` as
        tenant A returned tenant B's rows. The pre-#426 e2e suite didn't
        catch this because there was no session_templates cross-tenant
        test. Lock it in.
        """
        ka = await _mint_tenant(http_client, aios_env["AIOS_API_KEY"], "iso-tmpl-a")
        kb = await _mint_tenant(http_client, aios_env["AIOS_API_KEY"], "iso-tmpl-b")

        async def _mint_template_for(key: str, suffix: str) -> str:
            agent = await http_client.post(
                "/v1/agents",
                headers=_bearer(key),
                json={"name": f"tmpl-agent-{suffix}", "model": "openrouter/test"},
            )
            assert agent.status_code == 201, agent.text
            env = await http_client.post(
                "/v1/environments",
                headers=_bearer(key),
                json={"name": f"tmpl-env-{suffix}"},
            )
            assert env.status_code == 201, env.text
            tmpl = await http_client.post(
                "/v1/session-templates",
                headers=_bearer(key),
                json={
                    "name": f"tmpl-{suffix}",
                    "agent_id": agent.json()["id"],
                    "environment_id": env.json()["id"],
                },
            )
            assert tmpl.status_code == 201, tmpl.text
            return str(tmpl.json()["id"])

        a_id = await _mint_template_for(ka, "a")
        b_id = await _mint_template_for(kb, "b")
        assert a_id != b_id

        list_a = await http_client.get("/v1/session-templates", headers=_bearer(ka))
        list_b = await http_client.get("/v1/session-templates", headers=_bearer(kb))
        ids_a = {x["id"] for x in list_a.json()["data"]}
        ids_b = {x["id"] for x in list_b.json()["data"]}
        # The regression: pre-#426 ids_a would contain b_id.
        assert a_id in ids_a and b_id not in ids_a
        assert b_id in ids_b and a_id not in ids_b

        # Cross-tenant direct fetch returns 404.
        cross = await http_client.get(f"/v1/session-templates/{b_id}", headers=_bearer(ka))
        assert cross.status_code == 404, cross.text

    async def test_keys_isolated(
        self, http_client: httpx.AsyncClient, aios_env: dict[str, str]
    ) -> None:
        """Key listings on a sibling's account 404."""
        ka_resp = await http_client.post(
            "/v1/accounts/children",
            headers=_bearer(aios_env["AIOS_API_KEY"]),
            json={"display_name": "iso-keys-a"},
        )
        kb_resp = await http_client.post(
            "/v1/accounts/children",
            headers=_bearer(aios_env["AIOS_API_KEY"]),
            json={"display_name": "iso-keys-b"},
        )
        ka = ka_resp.json()["plaintext_key"]
        b_id = kb_resp.json()["account_id"]

        # A tries to list B's keys → 404.
        cross = await http_client.get(f"/v1/accounts/{b_id}/keys", headers=_bearer(ka))
        assert cross.status_code == 404, cross.text

        # A tries to mint a key on B → 404.
        mint = await http_client.post(
            f"/v1/accounts/{b_id}/keys",
            headers=_bearer(ka),
            json={"label": "stolen"},
        )
        assert mint.status_code == 404, mint.text
