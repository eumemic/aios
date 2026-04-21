"""E2E tests for the ``/v1/agents`` HTTP endpoints.

Currently focused on the ``?name=`` filter introduced for issue #41
(idempotent agent lookup by name).
"""

from __future__ import annotations

import secrets
from collections.abc import AsyncIterator
from typing import Any
from unittest import mock

import httpx
import pytest


def _uniq() -> str:
    return secrets.token_hex(4)


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
    from aios.api.app import create_app
    from aios.config import get_settings
    from aios.crypto.vault import CryptoBox

    settings = get_settings()
    app = create_app()
    app.state.pool = pool
    app.state.crypto_box = CryptoBox.from_base64(settings.vault_key.get_secret_value())
    app.state.db_url = settings.db_url
    app.state.procrastinate = mock.MagicMock()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
        headers={"Authorization": f"Bearer {aios_env['AIOS_API_KEY']}"},
    ) as client:
        yield client


async def _create_agent(http_client: httpx.AsyncClient, name: str) -> str:
    r = await http_client.post(
        "/v1/agents",
        json={
            "name": name,
            "model": "openai/gpt-4o-mini",
            "system": "",
            "tools": [],
            "metadata": {},
        },
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


class TestListAgentsNameFilter:
    async def test_returns_matching_agent(self, http_client: httpx.AsyncClient) -> None:
        target_name = f"target-{_uniq()}"
        other_name = f"other-{_uniq()}"
        target_id = await _create_agent(http_client, target_name)
        await _create_agent(http_client, other_name)

        r = await http_client.get("/v1/agents", params={"name": target_name})

        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["data"]) == 1
        assert body["data"][0]["id"] == target_id
        assert body["data"][0]["name"] == target_name

    async def test_returns_empty_when_no_match(self, http_client: httpx.AsyncClient) -> None:
        r = await http_client.get("/v1/agents", params={"name": f"nonexistent-{_uniq()}"})

        assert r.status_code == 200, r.text
        assert r.json()["data"] == []

    async def test_excludes_archived(self, http_client: httpx.AsyncClient) -> None:
        name = f"archive-me-{_uniq()}"
        agent_id = await _create_agent(http_client, name)

        r = await http_client.delete(f"/v1/agents/{agent_id}")
        assert r.status_code == 204, r.text

        r = await http_client.get("/v1/agents", params={"name": name})
        assert r.status_code == 200, r.text
        assert r.json()["data"] == []
