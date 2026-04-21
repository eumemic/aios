"""E2E tests for the ``litellm_extra`` agent field (issue #79)."""

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
    return r.json()


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
