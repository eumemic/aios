"""E2E: ``POST /v1/invocations`` env-ownership gate uses the *authenticated* account (#1130).

Part of #1122. The invoke edge (#1128) lets an external/operator caller supply
an ``environment_id`` for the servicer session it spawns. #1130 gates that
supply with an **ownership-only** check: the env must be owned by the
authenticated account (``get_environment(.., account_id=)`` — the same gate
``create_session`` / ``create_run`` already enforce).

These tests drive the HTTP endpoint end-to-end through bearer auth, so they pin
the load-bearing invariant the service-level tests can't: the gate scopes on the
account_id resolved from the **bearer token** (``AccountIdDep``), never a
body-supplied account. A cross-tenant ``environment_id`` 404s; a self-owned one
binds the servicer session.

The wakes are mocked (``wired_app`` injects a MagicMock procrastinate handle and
the auth fixture's app never runs a worker), so this exercises the edge-write +
ownership gate, not the worker stepping.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from aios.services import agents as agents_service
from aios.services import environments as environments_service
from tests.helpers.connections import asgi_client

pytestmark = pytest.mark.e2e

# The bearer key seeded by ``aios_env`` binds to this account (tests/conftest.py).
_CALLER = "acc_test_stub"
_OTHER = "acc_invoke_other_tenant"


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
async def pool(aios_env: dict[str, str]) -> AsyncIterator[Any]:
    from aios.config import get_settings
    from aios.db.pool import create_pool

    settings = get_settings()
    p = await create_pool(settings.db_url, min_size=1, max_size=4)
    try:
        yield p
    finally:
        await p.close()


@pytest.fixture
async def http_client(pool: Any) -> AsyncIterator[httpx.AsyncClient]:
    async with asgi_client(pool) as client:
        yield client


@pytest.fixture
async def caller_agent_id(pool: Any) -> str:
    """An agent owned by the bearer-auth caller (``acc_test_stub``)."""
    agent = await agents_service.create_agent(
        pool,
        account_id=_CALLER,
        name="invoke-owner-agent",
        model="openrouter/test",
        system="",
        tools=[],
        description=None,
        metadata={},
        window_min=50_000,
        window_max=150_000,
    )
    return agent.id


@pytest.fixture
async def caller_env_id(pool: Any) -> str:
    """An environment owned by the bearer-auth caller (``acc_test_stub``)."""
    env = await environments_service.create_environment(
        pool, account_id=_CALLER, name="invoke-owner-env"
    )
    return env.id


@pytest.fixture
async def other_tenant_env_id(pool: Any) -> str:
    """An environment owned by a DIFFERENT account than the bearer caller."""
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
            "VALUES ($1, NULL, FALSE, 'invoke-other-tenant')",
            _OTHER,
        )
    env = await environments_service.create_environment(
        pool, account_id=_OTHER, name="invoke-foreign-env"
    )
    return env.id


class TestInvokeEnvOwnershipGate:
    async def test_self_owned_environment_binds_servicer_session(
        self,
        http_client: httpx.AsyncClient,
        aios_env: dict[str, str],
        caller_agent_id: str,
        caller_env_id: str,
    ) -> None:
        """An ``environment_id`` owned by the bearer caller is accepted and bound."""
        r = await http_client.post(
            "/v1/invocations",
            headers=_bearer(aios_env["AIOS_API_KEY"]),
            json={
                "target_kind": "agent",
                "target": caller_agent_id,
                "input": {"q": "hello"},
                "environment_id": caller_env_id,
            },
        )
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["servicer_kind"] == "session"
        assert body["servicer_id"].startswith("sess_")
        assert body["request_id"].startswith("req_")

    async def test_cross_tenant_environment_refused(
        self,
        http_client: httpx.AsyncClient,
        aios_env: dict[str, str],
        caller_agent_id: str,
        other_tenant_env_id: str,
    ) -> None:
        """An ``environment_id`` owned by a DIFFERENT account is refused (404).

        The gate scopes ``get_environment`` by the account_id resolved from the
        bearer token (``AccountIdDep``), so a foreign env id is not-found from
        the caller's vantage — never accepted via a body-supplied account. This
        is the #1130 ownership gate observed end-to-end through HTTP auth.
        """
        r = await http_client.post(
            "/v1/invocations",
            headers=_bearer(aios_env["AIOS_API_KEY"]),
            json={
                "target_kind": "agent",
                "target": caller_agent_id,
                "input": {"q": "hello"},
                "environment_id": other_tenant_env_id,
            },
        )
        assert r.status_code == 404, r.text
