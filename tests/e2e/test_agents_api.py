"""E2E tests for the ``/v1/agents`` HTTP endpoints.

Currently focused on the ``?name=`` filter introduced for issue #41
(idempotent agent lookup by name).
"""

from __future__ import annotations

import secrets
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from tests.helpers.connections import authed_client, wired_app


def _uniq() -> str:
    return secrets.token_hex(4)


@pytest.fixture
async def http_client(pool: Any, aios_env: dict[str, str]) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=wired_app(pool))
    async with authed_client(
        "http://testserver",
        aios_env["AIOS_API_KEY"],
        transport=transport,
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
    agent_id: str = r.json()["id"]
    return agent_id


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
