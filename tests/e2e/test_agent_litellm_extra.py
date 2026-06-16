"""E2E tests for the ``litellm_extra`` agent field (issue #79)."""

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


async def _create(
    client: httpx.AsyncClient, name: str, *, litellm_extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "name": name,
        "model": "openai/gpt-4o-mini",
        "system": "",
        "tools": [],
        "metadata": {},
    }
    if litellm_extra is not None:
        body["litellm_extra"] = litellm_extra
    r = await client.post("/v1/agents", json=body)
    assert r.status_code == 201, r.text
    result: dict[str, Any] = r.json()
    return result


class TestLitellmExtraRoundTrip:
    async def test_default_is_empty_dict(self, http_client: httpx.AsyncClient) -> None:
        """Omitting the field yields ``{}`` on read back."""
        agent = await _create(http_client, f"no-extras-{_uniq()}")
        assert agent["litellm_extra"] == {}

    async def test_create_with_extras_persists(self, http_client: httpx.AsyncClient) -> None:
        extras = {
            "extra_body": {"provider": {"order": ["friendli"]}},
            "reasoning_effort": "high",
            "temperature": 0.2,
        }
        agent = await _create(http_client, f"with-extras-{_uniq()}", litellm_extra=extras)
        assert agent["litellm_extra"] == extras

        r = await http_client.get(f"/v1/agents/{agent['id']}")
        assert r.status_code == 200
        assert r.json()["litellm_extra"] == extras


class TestLitellmExtraVersionBump:
    async def test_changing_extras_creates_new_version(
        self, http_client: httpx.AsyncClient
    ) -> None:
        agent = await _create(http_client, f"v-bump-{_uniq()}")
        assert agent["version"] == 1

        r = await http_client.put(
            f"/v1/agents/{agent['id']}",
            json={"version": 1, "litellm_extra": {"thinking": {"type": "enabled"}}},
        )
        assert r.status_code == 200, r.text
        assert r.json()["version"] == 2
        assert r.json()["litellm_extra"] == {"thinking": {"type": "enabled"}}

    async def test_noop_update_does_not_bump(self, http_client: httpx.AsyncClient) -> None:
        agent = await _create(http_client, f"noop-{_uniq()}", litellm_extra={"temperature": 0.5})
        assert agent["version"] == 1

        r = await http_client.put(
            f"/v1/agents/{agent['id']}",
            json={"version": 1, "litellm_extra": {"temperature": 0.5}},
        )
        assert r.status_code == 200, r.text
        assert r.json()["version"] == 1

    async def test_agent_version_snapshot_records_extras(
        self, http_client: httpx.AsyncClient
    ) -> None:
        agent = await _create(http_client, f"snap-{_uniq()}", litellm_extra={"temperature": 0.9})

        r = await http_client.get(f"/v1/agents/{agent['id']}/versions/1")
        assert r.status_code == 200, r.text
        assert r.json()["litellm_extra"] == {"temperature": 0.9}
