"""E2E tests for connector tokens (#301).

Pins the per-connection scoped bearer token contract:

* Plaintext is returned ONCE on issue and never persisted.
* The bearer resolves to one ``connection_id`` via ``ConnectorAuthDep``.
* Revocation is soft (audit trail) and immediately blocks resolution.
* Operator endpoints (issue / list / revoke) require the global key.
* Connector endpoints (``/whoami``) require a valid token; the global
  key is REJECTED — issue tokens are not interchangeable with the
  operator key.
"""

from __future__ import annotations

import httpx

from tests.conftest import needs_docker


async def _create_connection(http_client: httpx.AsyncClient, account: str) -> str:
    """Create a detached connection, return its id."""
    r = await http_client.post(
        "/v1/connections",
        json={"connector": "echo", "account": account},
    )
    assert r.status_code == 201, r.text
    return str(r.json()["id"])


@needs_docker
class TestIssueAndResolve:
    async def test_issue_returns_plaintext(self, http_client: httpx.AsyncClient) -> None:
        connection_id = await _create_connection(http_client, "acct-issue-1")
        r = await http_client.post(
            "/v1/connector-tokens",
            json={"connection_id": connection_id, "label": "test-bot"},
        )
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["connection_id"] == connection_id
        assert body["label"] == "test-bot"
        assert body["plaintext"].startswith("aios_conn_")
        assert len(body["plaintext"]) > 30  # 32 bytes base64url = ~43 chars + prefix
        assert body["id"].startswith("ctok_")

    async def test_whoami_resolves_token(self, http_client: httpx.AsyncClient) -> None:
        connection_id = await _create_connection(http_client, "acct-whoami-1")
        issue_r = await http_client.post(
            "/v1/connector-tokens",
            json={"connection_id": connection_id, "label": None},
        )
        plaintext = issue_r.json()["plaintext"]

        # Use the connector token to hit /whoami — it should resolve.
        whoami_r = await http_client.get(
            "/v1/connector-tokens/whoami",
            headers={"Authorization": f"Bearer {plaintext}"},
        )
        assert whoami_r.status_code == 200, whoami_r.text
        assert whoami_r.json()["connection_id"] == connection_id

    async def test_operator_key_rejected_on_connector_route(
        self, http_client: httpx.AsyncClient, aios_env: dict[str, str]
    ) -> None:
        """The global API key is operator-only; connector routes must
        reject it.  Otherwise scope is meaningless."""
        r = await http_client.get(
            "/v1/connector-tokens/whoami",
            headers={"Authorization": f"Bearer {aios_env['AIOS_API_KEY']}"},
        )
        assert r.status_code == 401, r.text

    async def test_garbage_token_rejected(self, http_client: httpx.AsyncClient) -> None:
        r = await http_client.get(
            "/v1/connector-tokens/whoami",
            headers={"Authorization": "Bearer not-even-our-prefix-xyz"},
        )
        assert r.status_code == 401, r.text

    async def test_missing_auth_rejected(self, http_client: httpx.AsyncClient) -> None:
        # Build a fresh client without default auth headers.
        async with httpx.AsyncClient(
            transport=http_client._transport,
            base_url="http://testserver",
        ) as anon:
            r = await anon.get("/v1/connector-tokens/whoami")
            assert r.status_code == 401, r.text


@needs_docker
class TestRevoke:
    async def test_revoked_token_rejected(self, http_client: httpx.AsyncClient) -> None:
        connection_id = await _create_connection(http_client, "acct-revoke-1")
        issue_r = await http_client.post(
            "/v1/connector-tokens",
            json={"connection_id": connection_id, "label": None},
        )
        token_id = issue_r.json()["id"]
        plaintext = issue_r.json()["plaintext"]

        # Token works before revoke.
        ok = await http_client.get(
            "/v1/connector-tokens/whoami",
            headers={"Authorization": f"Bearer {plaintext}"},
        )
        assert ok.status_code == 200

        # Revoke it.
        rev = await http_client.post(f"/v1/connector-tokens/{token_id}/revoke")
        assert rev.status_code == 200, rev.text
        assert rev.json()["revoked_at"] is not None

        # Token rejected after revoke.
        denied = await http_client.get(
            "/v1/connector-tokens/whoami",
            headers={"Authorization": f"Bearer {plaintext}"},
        )
        assert denied.status_code == 401

    async def test_revoke_idempotent(self, http_client: httpx.AsyncClient) -> None:
        connection_id = await _create_connection(http_client, "acct-revoke-idem")
        issue_r = await http_client.post(
            "/v1/connector-tokens",
            json={"connection_id": connection_id},
        )
        token_id = issue_r.json()["id"]

        first = await http_client.post(f"/v1/connector-tokens/{token_id}/revoke")
        assert first.status_code == 200
        first_revoked_at = first.json()["revoked_at"]

        second = await http_client.post(f"/v1/connector-tokens/{token_id}/revoke")
        assert second.status_code == 200
        assert second.json()["revoked_at"] == first_revoked_at  # unchanged


@needs_docker
class TestListConnectorTokens:
    async def test_list_includes_revoked(self, http_client: httpx.AsyncClient) -> None:
        connection_id = await _create_connection(http_client, "acct-list-1")
        live_id = (
            await http_client.post(
                "/v1/connector-tokens",
                json={"connection_id": connection_id, "label": "live"},
            )
        ).json()["id"]
        dead_id = (
            await http_client.post(
                "/v1/connector-tokens",
                json={"connection_id": connection_id, "label": "dead"},
            )
        ).json()["id"]
        await http_client.post(f"/v1/connector-tokens/{dead_id}/revoke")

        r = await http_client.get("/v1/connector-tokens", params={"connection_id": connection_id})
        assert r.status_code == 200
        ids_to_revoked = {item["id"]: item["revoked_at"] for item in r.json()["data"]}
        assert ids_to_revoked[live_id] is None
        assert ids_to_revoked[dead_id] is not None

    async def test_list_scoped_per_connection(self, http_client: httpx.AsyncClient) -> None:
        c1 = await _create_connection(http_client, "acct-list-2a")
        c2 = await _create_connection(http_client, "acct-list-2b")
        await http_client.post("/v1/connector-tokens", json={"connection_id": c1})
        await http_client.post("/v1/connector-tokens", json={"connection_id": c2})

        r = await http_client.get("/v1/connector-tokens", params={"connection_id": c1})
        for item in r.json()["data"]:
            assert item["connection_id"] == c1


@needs_docker
class TestPlaintextNotPersisted:
    async def test_read_view_omits_plaintext(self, http_client: httpx.AsyncClient) -> None:
        """Defensive check: the listing surface NEVER contains plaintext."""
        connection_id = await _create_connection(http_client, "acct-noleak")
        await http_client.post(
            "/v1/connector-tokens",
            json={"connection_id": connection_id},
        )
        r = await http_client.get("/v1/connector-tokens", params={"connection_id": connection_id})
        for item in r.json()["data"]:
            assert "plaintext" not in item
