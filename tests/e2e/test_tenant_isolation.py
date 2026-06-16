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

    async def test_vault_credential_create_cross_tenant_404(
        self, http_client: httpx.AsyncClient, aios_env: dict[str, str]
    ) -> None:
        """POST /v1/vaults/{B-vault}/credentials as tenant A must 404.

        ``services.vaults.create_vault_credential`` row-locks the parent
        vault with ``SELECT 1 FROM vaults WHERE id = $1 FOR UPDATE`` —
        no ``account_id`` filter. Tenant A can target tenant B's
        ``vault_id``, land the lock, then ``insert_vault_credential``
        writes a row with ``(account_id=A, vault_id=B's vault)`` —
        cross-tenant data placement. The credential is invisible to B
        (B's reads scope by ``account_id``), but A has manufactured a
        credential nested under a vault it doesn't own. Match the
        established cross-tenant posture: NotFound, not silent success.
        """
        ka = await _mint_tenant(http_client, aios_env["AIOS_API_KEY"], "iso-vcred-a")
        kb = await _mint_tenant(http_client, aios_env["AIOS_API_KEY"], "iso-vcred-b")

        vault_b = await http_client.post(
            "/v1/vaults", headers=_bearer(kb), json={"display_name": "b-vault"}
        )
        assert vault_b.status_code == 201, vault_b.text
        b_vault_id = vault_b.json()["id"]

        # Tenant A targets tenant B's vault_id; expected 404, not 201.
        cross = await http_client.post(
            f"/v1/vaults/{b_vault_id}/credentials",
            headers=_bearer(ka),
            json={
                "display_name": "stolen",
                "target_url": "https://example.com",
                "auth_type": "bearer_header",
                "token": "secret",
            },
        )
        assert cross.status_code == 404, cross.text

        # Confirm B's view of its own vault is uncontaminated.
        list_b = await http_client.get(f"/v1/vaults/{b_vault_id}/credentials", headers=_bearer(kb))
        assert list_b.status_code == 200
        assert list_b.json()["data"] == []

    async def test_session_create_cross_tenant_env_404(
        self, http_client: httpx.AsyncClient, aios_env: dict[str, str]
    ) -> None:
        """POST /v1/sessions binding tenant B's environment_id as tenant A must 404.

        ``services.sessions.create_session`` opened its insert transaction and
        called ``insert_session`` directly — the environment was validated only
        by its FK (existence, not ownership). Tenant A could pass tenant B's
        ``environment_id`` and the session was created bound to B's environment
        (its image / env-vars / networking). The sibling ``create_run`` path
        already validates the env as account-owned; match that posture here
        (issue #755). NotFound, not silent cross-tenant binding.
        """
        ka = await _mint_tenant(http_client, aios_env["AIOS_API_KEY"], "iso-sess-a")
        kb = await _mint_tenant(http_client, aios_env["AIOS_API_KEY"], "iso-sess-b")

        # Tenant B owns env_b.
        env_b = await http_client.post(
            "/v1/environments", headers=_bearer(kb), json={"name": "sess-env-b"}
        )
        assert env_b.status_code == 201, env_b.text
        env_b_id = env_b.json()["id"]

        # Tenant A owns the agent.
        agent_a = await http_client.post(
            "/v1/agents",
            headers=_bearer(ka),
            json={"name": "sess-agent-a", "model": "openrouter/test"},
        )
        assert agent_a.status_code == 201, agent_a.text
        agent_a_id = agent_a.json()["id"]

        # Tenant A targets tenant B's env_id; expected 404, not a bound session.
        cross = await http_client.post(
            "/v1/sessions",
            headers=_bearer(ka),
            json={"agent_id": agent_a_id, "environment_id": env_b_id},
        )
        assert cross.status_code == 404, cross.text

        # Same-tenant control: tenant A's own env still binds a session (201).
        env_a = await http_client.post(
            "/v1/environments", headers=_bearer(ka), json={"name": "sess-env-a"}
        )
        assert env_a.status_code == 201, env_a.text
        env_a_id = env_a.json()["id"]
        ok = await http_client.post(
            "/v1/sessions",
            headers=_bearer(ka),
            json={"agent_id": agent_a_id, "environment_id": env_a_id},
        )
        assert ok.status_code == 201, ok.text

    async def test_session_template_create_cross_tenant_env_404(
        self, http_client: httpx.AsyncClient, aios_env: dict[str, str]
    ) -> None:
        """POST /v1/session-templates binding tenant B's env as tenant A must 404.

        ``services.session_templates.create_session_template`` wrote the
        caller-supplied ``environment_id`` with no ownership check — only the
        FK guarded it (existence, not ownership). Tenant A could bind tenant B's
        environment (its image / env-vars / networking) into a template. Same
        defect class as the sessions/runs paths; match that posture
        (issue #755). NotFound, not silent cross-tenant binding.
        """
        ka = await _mint_tenant(http_client, aios_env["AIOS_API_KEY"], "iso-tmpl-a")
        kb = await _mint_tenant(http_client, aios_env["AIOS_API_KEY"], "iso-tmpl-b")

        # Tenant B owns env_b.
        env_b = await http_client.post(
            "/v1/environments", headers=_bearer(kb), json={"name": "tmpl-create-env-b"}
        )
        assert env_b.status_code == 201, env_b.text
        env_b_id = env_b.json()["id"]

        # Tenant A owns the agent.
        agent_a = await http_client.post(
            "/v1/agents",
            headers=_bearer(ka),
            json={"name": "tmpl-create-agent-a", "model": "openrouter/test"},
        )
        assert agent_a.status_code == 201, agent_a.text
        agent_a_id = agent_a.json()["id"]

        # Tenant A targets tenant B's env_id; expected 404, not a bound template.
        cross = await http_client.post(
            "/v1/session-templates",
            headers=_bearer(ka),
            json={
                "name": "tmpl-create-cross",
                "agent_id": agent_a_id,
                "environment_id": env_b_id,
            },
        )
        assert cross.status_code == 404, cross.text

        # Same-tenant control: tenant A's own env still binds a template (201).
        env_a = await http_client.post(
            "/v1/environments", headers=_bearer(ka), json={"name": "tmpl-create-env-a"}
        )
        assert env_a.status_code == 201, env_a.text
        env_a_id = env_a.json()["id"]
        ok = await http_client.post(
            "/v1/session-templates",
            headers=_bearer(ka),
            json={
                "name": "tmpl-create-own",
                "agent_id": agent_a_id,
                "environment_id": env_a_id,
            },
        )
        assert ok.status_code == 201, ok.text

    async def test_session_template_update_cross_tenant_env_404(
        self, http_client: httpx.AsyncClient, aios_env: dict[str, str]
    ) -> None:
        """PUT /v1/session-templates/{id} rebinding to tenant B's env must 404.

        ``services.session_templates.update_session_template`` wrote the
        caller-supplied ``environment_id`` with no ownership check. Tenant A
        could rebind its own template onto tenant B's environment. Validate the
        new env as account-owned only when the caller actually supplies one —
        omitting it must still allow partial updates (issue #755).
        """
        ka = await _mint_tenant(http_client, aios_env["AIOS_API_KEY"], "iso-tmplu-a")
        kb = await _mint_tenant(http_client, aios_env["AIOS_API_KEY"], "iso-tmplu-b")

        # Tenant B owns env_b.
        env_b = await http_client.post(
            "/v1/environments", headers=_bearer(kb), json={"name": "tmpl-update-env-b"}
        )
        assert env_b.status_code == 201, env_b.text
        env_b_id = env_b.json()["id"]

        # Tenant A owns the agent + its own env, and a template bound to env_a.
        agent_a = await http_client.post(
            "/v1/agents",
            headers=_bearer(ka),
            json={"name": "tmpl-update-agent-a", "model": "openrouter/test"},
        )
        assert agent_a.status_code == 201, agent_a.text
        agent_a_id = agent_a.json()["id"]
        env_a = await http_client.post(
            "/v1/environments", headers=_bearer(ka), json={"name": "tmpl-update-env-a"}
        )
        assert env_a.status_code == 201, env_a.text
        env_a_id = env_a.json()["id"]
        tmpl = await http_client.post(
            "/v1/session-templates",
            headers=_bearer(ka),
            json={
                "name": "tmpl-update",
                "agent_id": agent_a_id,
                "environment_id": env_a_id,
            },
        )
        assert tmpl.status_code == 201, tmpl.text
        template_id = tmpl.json()["id"]

        # Tenant A rebinds onto tenant B's env_id; expected 404, not a rebind.
        cross = await http_client.put(
            f"/v1/session-templates/{template_id}",
            headers=_bearer(ka),
            json={"environment_id": env_b_id},
        )
        assert cross.status_code == 404, cross.text

        # Same-tenant control: rebinding onto tenant A's own env succeeds (200).
        ok = await http_client.put(
            f"/v1/session-templates/{template_id}",
            headers=_bearer(ka),
            json={"environment_id": env_a_id},
        )
        assert ok.status_code == 200, ok.text

        # Partial update with no env change still succeeds — the conditional
        # ownership check must not block updates that omit environment_id.
        name_only = await http_client.put(
            f"/v1/session-templates/{template_id}",
            headers=_bearer(ka),
            json={"name": "tmpl-update-renamed"},
        )
        assert name_only.status_code == 200, name_only.text

    async def test_session_create_cross_tenant_agent_404(
        self, http_client: httpx.AsyncClient, aios_env: dict[str, str]
    ) -> None:
        """POST /v1/sessions binding tenant B's agent_id as tenant A must 404.

        ``services.sessions.create_session`` validated only the environment as
        account-owned; ``agent_id`` was FK-only (existence, not ownership), so
        tenant A could bind tenant B's agent (its model / surface) into a
        session. Match the env guard's posture (issue #851). NotFound, not a
        cross-tenant binding.
        """
        ka = await _mint_tenant(http_client, aios_env["AIOS_API_KEY"], "iso-sessag-a")
        kb = await _mint_tenant(http_client, aios_env["AIOS_API_KEY"], "iso-sessag-b")

        # Tenant B owns agent_b.
        agent_b = await http_client.post(
            "/v1/agents",
            headers=_bearer(kb),
            json={"name": "sessag-agent-b", "model": "openrouter/test"},
        )
        assert agent_b.status_code == 201, agent_b.text
        agent_b_id = agent_b.json()["id"]

        # Tenant A owns the environment.
        env_a = await http_client.post(
            "/v1/environments", headers=_bearer(ka), json={"name": "sessag-env-a"}
        )
        assert env_a.status_code == 201, env_a.text
        env_a_id = env_a.json()["id"]

        # Tenant A targets tenant B's agent_id; expected 404, not a bound session.
        cross = await http_client.post(
            "/v1/sessions",
            headers=_bearer(ka),
            json={"agent_id": agent_b_id, "environment_id": env_a_id},
        )
        assert cross.status_code == 404, cross.text

        # Same-tenant control: tenant A's own agent still binds a session (201).
        agent_a = await http_client.post(
            "/v1/agents",
            headers=_bearer(ka),
            json={"name": "sessag-agent-a", "model": "openrouter/test"},
        )
        assert agent_a.status_code == 201, agent_a.text
        ok = await http_client.post(
            "/v1/sessions",
            headers=_bearer(ka),
            json={"agent_id": agent_a.json()["id"], "environment_id": env_a_id},
        )
        assert ok.status_code == 201, ok.text

    async def test_session_template_create_cross_tenant_agent_404(
        self, http_client: httpx.AsyncClient, aios_env: dict[str, str]
    ) -> None:
        """POST /v1/session-templates binding tenant B's agent as tenant A must 404.

        ``create_session_template`` validated only the environment; ``agent_id``
        was FK-only. Tenant A could bind tenant B's agent into a template. Match
        the env guard's posture (issue #851).
        """
        ka = await _mint_tenant(http_client, aios_env["AIOS_API_KEY"], "iso-tmplag-a")
        kb = await _mint_tenant(http_client, aios_env["AIOS_API_KEY"], "iso-tmplag-b")

        # Tenant B owns agent_b.
        agent_b = await http_client.post(
            "/v1/agents",
            headers=_bearer(kb),
            json={"name": "tmplag-agent-b", "model": "openrouter/test"},
        )
        assert agent_b.status_code == 201, agent_b.text
        agent_b_id = agent_b.json()["id"]

        # Tenant A owns the environment.
        env_a = await http_client.post(
            "/v1/environments", headers=_bearer(ka), json={"name": "tmplag-env-a"}
        )
        assert env_a.status_code == 201, env_a.text
        env_a_id = env_a.json()["id"]

        # Tenant A targets tenant B's agent_id; expected 404, not a bound template.
        cross = await http_client.post(
            "/v1/session-templates",
            headers=_bearer(ka),
            json={
                "name": "tmplag-cross",
                "agent_id": agent_b_id,
                "environment_id": env_a_id,
            },
        )
        assert cross.status_code == 404, cross.text

        # Same-tenant control: tenant A's own agent still binds a template (201).
        agent_a = await http_client.post(
            "/v1/agents",
            headers=_bearer(ka),
            json={"name": "tmplag-agent-a", "model": "openrouter/test"},
        )
        assert agent_a.status_code == 201, agent_a.text
        ok = await http_client.post(
            "/v1/session-templates",
            headers=_bearer(ka),
            json={
                "name": "tmplag-own",
                "agent_id": agent_a.json()["id"],
                "environment_id": env_a_id,
            },
        )
        assert ok.status_code == 201, ok.text

    async def test_session_template_create_cross_tenant_vault_404(
        self, http_client: httpx.AsyncClient, aios_env: dict[str, str]
    ) -> None:
        """POST /v1/session-templates with tenant B's vault_id as tenant A must 404.

        ``vault_ids`` is a plain ``text[]`` column with NO FK — a foreign id
        would silently bind. The ownership guard rejects it (issue #851).
        """
        ka = await _mint_tenant(http_client, aios_env["AIOS_API_KEY"], "iso-tmplvt-a")
        kb = await _mint_tenant(http_client, aios_env["AIOS_API_KEY"], "iso-tmplvt-b")

        # Tenant A owns its agent + env.
        agent_a = await http_client.post(
            "/v1/agents",
            headers=_bearer(ka),
            json={"name": "tmplvt-agent-a", "model": "openrouter/test"},
        )
        assert agent_a.status_code == 201, agent_a.text
        agent_a_id = agent_a.json()["id"]
        env_a = await http_client.post(
            "/v1/environments", headers=_bearer(ka), json={"name": "tmplvt-env-a"}
        )
        assert env_a.status_code == 201, env_a.text
        env_a_id = env_a.json()["id"]

        # Tenant B owns vault_b.
        vault_b = await http_client.post(
            "/v1/vaults", headers=_bearer(kb), json={"display_name": "tmplvt-vault-b"}
        )
        assert vault_b.status_code == 201, vault_b.text
        vault_b_id = vault_b.json()["id"]

        # Tenant A targets tenant B's vault_id; expected 404.
        cross = await http_client.post(
            "/v1/session-templates",
            headers=_bearer(ka),
            json={
                "name": "tmplvt-cross",
                "agent_id": agent_a_id,
                "environment_id": env_a_id,
                "vault_ids": [vault_b_id],
            },
        )
        assert cross.status_code == 404, cross.text

        # Same-tenant control: tenant A's own vault binds a template (201).
        vault_a = await http_client.post(
            "/v1/vaults", headers=_bearer(ka), json={"display_name": "tmplvt-vault-a"}
        )
        assert vault_a.status_code == 201, vault_a.text
        ok = await http_client.post(
            "/v1/session-templates",
            headers=_bearer(ka),
            json={
                "name": "tmplvt-own",
                "agent_id": agent_a_id,
                "environment_id": env_a_id,
                "vault_ids": [vault_a.json()["id"]],
            },
        )
        assert ok.status_code == 201, ok.text

    async def test_session_template_create_cross_tenant_memory_store_404(
        self, http_client: httpx.AsyncClient, aios_env: dict[str, str]
    ) -> None:
        """POST /v1/session-templates with tenant B's memory_store_id as tenant A must 404.

        ``memory_store_ids`` is a plain ``text[]`` column with NO FK — a foreign
        id would silently bind. The ownership guard rejects it (issue #851).
        """
        ka = await _mint_tenant(http_client, aios_env["AIOS_API_KEY"], "iso-tmplms-a")
        kb = await _mint_tenant(http_client, aios_env["AIOS_API_KEY"], "iso-tmplms-b")

        # Tenant A owns its agent + env.
        agent_a = await http_client.post(
            "/v1/agents",
            headers=_bearer(ka),
            json={"name": "tmplms-agent-a", "model": "openrouter/test"},
        )
        assert agent_a.status_code == 201, agent_a.text
        agent_a_id = agent_a.json()["id"]
        env_a = await http_client.post(
            "/v1/environments", headers=_bearer(ka), json={"name": "tmplms-env-a"}
        )
        assert env_a.status_code == 201, env_a.text
        env_a_id = env_a.json()["id"]

        # Tenant B owns store_b.
        store_b = await http_client.post(
            "/v1/memory-stores", headers=_bearer(kb), json={"name": "tmplms-store-b"}
        )
        assert store_b.status_code == 201, store_b.text
        store_b_id = store_b.json()["id"]

        # Tenant A targets tenant B's memory_store_id; expected 404.
        cross = await http_client.post(
            "/v1/session-templates",
            headers=_bearer(ka),
            json={
                "name": "tmplms-cross",
                "agent_id": agent_a_id,
                "environment_id": env_a_id,
                "memory_store_ids": [store_b_id],
            },
        )
        assert cross.status_code == 404, cross.text

        # Same-tenant control: tenant A's own store binds a template (201).
        store_a = await http_client.post(
            "/v1/memory-stores", headers=_bearer(ka), json={"name": "tmplms-store-a"}
        )
        assert store_a.status_code == 201, store_a.text
        ok = await http_client.post(
            "/v1/session-templates",
            headers=_bearer(ka),
            json={
                "name": "tmplms-own",
                "agent_id": agent_a_id,
                "environment_id": env_a_id,
                "memory_store_ids": [store_a.json()["id"]],
            },
        )
        assert ok.status_code == 201, ok.text

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
