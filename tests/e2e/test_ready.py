"""E2E tests for the readiness/liveness probes (Docker: testcontainer Postgres).

``/ready`` does a real ``SELECT 1`` against the live pool; ``/health`` stays
pure liveness. Fixtures mirror ``tests/e2e/test_account_auth.py`` (a real pool
+ an in-process ASGI client wired to the shared app).
"""

from __future__ import annotations

import logging
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


async def test_ready_200_db_up(http_client: httpx.AsyncClient) -> None:
    """With Postgres up, ``/ready`` runs ``SELECT 1`` and returns 200."""
    r = await http_client.get("/ready")
    assert r.status_code == 200, r.text
    assert r.json() == {"status": "ready"}


async def test_health_still_200(http_client: httpx.AsyncClient) -> None:
    """``/health`` stays pure liveness — 200 with the running version."""
    r = await http_client.get("/health")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


async def test_401_logged(http_client: httpx.AsyncClient, caplog: pytest.LogCaptureFixture) -> None:
    """An unauthenticated protected request 401s, and the request-logging
    middleware still emits an ``api.request`` line carrying status 401 (no
    account_id, since auth never resolved)."""
    from aios.logging import configure_logging

    configure_logging("INFO")
    with caplog.at_level(logging.INFO):
        r = await http_client.get("/v1/agents")
    assert r.status_code == 401, r.text

    request_lines = [
        m for r in caplog.records if (m := r.getMessage()).startswith("{") and '"api.request"' in m
    ]
    assert any('"status": 401' in line and '"path": "/v1/agents"' in line for line in request_lines)
