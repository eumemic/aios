"""E2E test: every list endpoint rejects out-of-range ``?limit=`` values.

Pre-fix: every paginated route declared ``limit: int = N`` with no
Pydantic constraints. Three symptoms:

* ``?limit=0`` returned ``has_more=True, data=[]`` (the standard
  ``has_more=len(items) == limit`` shortcut evaluated to ``0 == 0``
  → ``True``; clients that loop on ``has_more`` without null-checking
  ``next_after`` would spin).
* ``?limit=-1`` reached Postgres as ``LIMIT -1`` → ``asyncpg`` raised
  ``InvalidRowCountInLimitClauseError`` → unhandled 500.
* No upper bound — a single client request could pull unbounded rows
  (DoS surface; one request fetching ~1 GB of event data).

Fix: every paginated route now declares
``limit: Annotated[int, Query(ge=1, le=200)]`` (or ``le=500`` for the
events endpoint). Pydantic / FastAPI rejects at the request boundary
with 422; the routes never see invalid values.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from tests.helpers.connections import authed_client, wired_app


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
    transport = httpx.ASGITransport(app=wired_app(pool))
    async with authed_client(
        "http://testserver",
        aios_env["AIOS_API_KEY"],
        transport=transport,
    ) as client:
        yield client


# Path-free GET-list endpoints. Excludes endpoints that need an existing
# session_id / store_id (events list, memories list, etc.) — those go
# through the same Pydantic validation, the path-free set is enough to
# pin the contract that the validation is uniformly applied.
_LIST_PATHS = [
    "/v1/agents",
    "/v1/sessions",
    "/v1/skills",
    "/v1/environments",
    "/v1/vaults",
    "/v1/connections",
    "/v1/memory-stores",
    "/v1/session-templates",
]


class TestPaginationLimitValidation:
    @pytest.mark.parametrize("path", _LIST_PATHS)
    async def test_limit_zero_rejected_with_422(
        self, http_client: httpx.AsyncClient, path: str
    ) -> None:
        r = await http_client.get(path, params={"limit": 0})
        assert r.status_code == 422, (
            f"{path}?limit=0 should reject with 422 (Pydantic Field ge=1); "
            f"got {r.status_code} body={r.text[:200]}"
        )

    @pytest.mark.parametrize("path", _LIST_PATHS)
    async def test_limit_negative_rejected_with_422(
        self, http_client: httpx.AsyncClient, path: str
    ) -> None:
        r = await http_client.get(path, params={"limit": -1})
        assert r.status_code == 422, (
            f"{path}?limit=-1 should reject with 422 (Pydantic Field ge=1); "
            f"got {r.status_code} body={r.text[:200]} — without the "
            f"constraint the value reaches Postgres as ``LIMIT -1`` and "
            f"raises asyncpg.InvalidRowCountInLimitClauseError → 500"
        )

    @pytest.mark.parametrize("path", _LIST_PATHS)
    async def test_limit_over_max_rejected_with_422(
        self, http_client: httpx.AsyncClient, path: str
    ) -> None:
        r = await http_client.get(path, params={"limit": 201})
        assert r.status_code == 422, (
            f"{path}?limit=201 should reject with 422 (Pydantic Field le=200); "
            f"got {r.status_code} body={r.text[:200]} — without the cap "
            f"a single request can pull unbounded rows (DoS surface)"
        )

    @pytest.mark.parametrize("path", _LIST_PATHS)
    async def test_limit_at_default_works(self, http_client: httpx.AsyncClient, path: str) -> None:
        """Regression guard: omitting limit (default) succeeds."""
        r = await http_client.get(path)
        assert r.status_code == 200, (
            f"{path} default-limit request should succeed; got {r.status_code} body={r.text[:200]}"
        )
        body = r.json()
        assert "data" in body
        assert isinstance(body["data"], list)
