"""E2E tests for ``POST /v1/connections/{id}/reparent`` (issue #694).

The reparent primitive moves ``connection.account_id`` in place,
preserving ``connection.id`` so connector-daemon state keyed by it
survives. Auth (v1) is root-operator-only: the caller's account must
have ``parent_account_id IS NULL``.

These tests drive the route through a real in-process FastAPI app
against a testcontainer Postgres, covering the four wire-visible
outcomes:

* 200 — root key reparenting between two of its child accounts.
* 404 — unknown connection (the destination account exists).
* 409 — the destination already holds an active connection on the same
  ``(connector, external_account_id)``.
* 403 — a non-root caller (a child account's own key).

All requests ride the single in-process ``http_client`` with a
per-request ``Authorization`` header rather than spinning up a fresh
``httpx.AsyncClient`` per actor — the latter would need its own
``ASGITransport`` bound to the same app, which is what ``bearer()`` on
the shared client sidesteps.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from tests.helpers.connections import asgi_client, bearer


@pytest.fixture
async def http_client(pool: Any, aios_env: dict[str, str]) -> AsyncIterator[httpx.AsyncClient]:
    """In-process API client (``asgi_client``).

    The reparent suite hits ``/v1/accounts/children`` and
    ``/v1/connections`` — neither defers a wake — so the lighter
    ``asgi_client`` (which doesn't mock ``defer_wake``) suffices.
    """
    async with asgi_client(pool) as client:
        yield client


async def _mint_child(http_client: httpx.AsyncClient, root_key: str, name: str) -> tuple[str, str]:
    """Create a child account under the bootstrapped root, returning
    ``(account_id, plaintext_key)``."""
    r = await http_client.post(
        "/v1/accounts/children",
        headers=bearer(root_key),
        json={"display_name": name, "can_mint_children": False},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    return body["account_id"], body["plaintext_key"]


async def _create_connection(
    http_client: httpx.AsyncClient,
    api_key: str,
    *,
    connector: str = "signal",
    external_account_id: str = "+15550001",
) -> str:
    """Create a detached connection scoped to ``api_key``'s account
    (the create route scopes the row's ``account_id`` to the bearer)."""
    r = await http_client.post(
        "/v1/connections",
        headers=bearer(api_key),
        json={"connector": connector, "external_account_id": external_account_id},
    )
    assert r.status_code == 201, r.text
    return str(r.json()["id"])


class TestReparentRoundTrip:
    async def test_root_reparents_between_two_children(
        self,
        http_client: httpx.AsyncClient,
        aios_env: dict[str, str],
        pool: Any,
    ) -> None:
        """Root key moves a child's connection to the sibling child; 200,
        ``id`` preserved, ``account_id`` flipped in the DB."""
        root_key = aios_env["AIOS_API_KEY"]
        _child_a, child_a_key = await _mint_child(http_client, root_key, "tenant-a")
        child_b, _child_b_key = await _mint_child(http_client, root_key, "tenant-b")

        connection_id = await _create_connection(http_client, child_a_key)

        r = await http_client.post(
            f"/v1/connections/{connection_id}/reparent",
            headers=bearer(root_key),
            json={"destination_account_id": child_b},
        )
        assert r.status_code == 200, r.text
        assert r.json()["id"] == connection_id

        # Cross-check via direct SQL: ``account_id`` actually moved.
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT account_id FROM connections WHERE id = $1", connection_id
            )
        assert row is not None
        assert row["account_id"] == child_b

    async def test_unknown_connection_returns_404(
        self,
        http_client: httpx.AsyncClient,
        aios_env: dict[str, str],
    ) -> None:
        """Reparenting a non-existent connection 404s. The destination
        account is well-formed, so it clears the destination-validation
        step and the miss surfaces on the source connection lookup."""
        root_key = aios_env["AIOS_API_KEY"]
        child_b, _ = await _mint_child(http_client, root_key, "tenant-b")

        r = await http_client.post(
            "/v1/connections/conn_does_not_exist/reparent",
            headers=bearer(root_key),
            json={"destination_account_id": child_b},
        )
        assert r.status_code == 404, r.text

    async def test_destination_collision_returns_409(
        self,
        http_client: httpx.AsyncClient,
        aios_env: dict[str, str],
    ) -> None:
        """When the destination already holds an active connection on
        the same ``(connector, external_account_id)``, the per-account
        UNIQUE bites and the route returns 409."""
        root_key = aios_env["AIOS_API_KEY"]
        _child_a, child_a_key = await _mint_child(http_client, root_key, "tenant-a")
        child_b, child_b_key = await _mint_child(http_client, root_key, "tenant-b")

        source_id = await _create_connection(http_client, child_a_key)
        # Destination already holds the same identity.
        await _create_connection(http_client, child_b_key)

        r = await http_client.post(
            f"/v1/connections/{source_id}/reparent",
            headers=bearer(root_key),
            json={"destination_account_id": child_b},
        )
        assert r.status_code == 409, r.text

    async def test_non_root_caller_returns_403(
        self,
        http_client: httpx.AsyncClient,
        aios_env: dict[str, str],
    ) -> None:
        """A child account's key cannot reparent — even within its own
        connections — under v1's root-only authz."""
        root_key = aios_env["AIOS_API_KEY"]
        _child_a, child_a_key = await _mint_child(http_client, root_key, "tenant-a")
        child_b, _ = await _mint_child(http_client, root_key, "tenant-b")

        source_id = await _create_connection(http_client, child_a_key)

        r = await http_client.post(
            f"/v1/connections/{source_id}/reparent",
            headers=bearer(child_a_key),
            json={"destination_account_id": child_b},
        )
        assert r.status_code == 403, r.text
