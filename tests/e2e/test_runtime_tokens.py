"""Runtime-token issue → use → revoke roundtrip (#328 PR 5).

Runtime tokens are the per-connector-type successor to per-connection
:mod:`connector_tokens`; this file exercises the operator surface (issue,
list, revoke) plus the runtime-scoped routes that the tokens
authenticate, asserting:

* A freshly-issued runtime token authenticates the runtime-scoped routes
  for its connector type.
* A runtime token does NOT authenticate the legacy per-connection
  routes (and vice versa) — the two auth deps are isolated.
* Revoked tokens 401.
"""

from __future__ import annotations

from tests.conftest import needs_docker
from tests.helpers.connections import bearer


async def _issue_runtime_token(http_client: object, connector: str) -> tuple[str, str]:
    """``(token_id, plaintext)`` for a fresh runtime token scoped to ``connector``."""
    r = await http_client.post(  # type: ignore[attr-defined]
        "/v1/runtime-tokens", json={"connector": connector}
    )
    r.raise_for_status()
    body = r.json()
    return str(body["id"]), str(body["plaintext"])


async def _issue_connector_token(http_client: object, connection_id: str) -> str:
    """Plaintext for a fresh legacy per-connection token."""
    r = await http_client.post(  # type: ignore[attr-defined]
        "/v1/connector-tokens", json={"connection_id": connection_id}
    )
    r.raise_for_status()
    return str(r.json()["plaintext"])


@needs_docker
class TestRuntimeTokensRoundtrip:
    async def test_issue_list_revoke(
        self,
        http_client: object,
        aios_env: dict[str, str],
    ) -> None:
        """Issue, list (newest-first), revoke, list-again (revoked_at set)."""
        token_id, plaintext = await _issue_runtime_token(http_client, "echo")
        assert plaintext.startswith("aios_runtime_")

        r = await http_client.get("/v1/runtime-tokens", params={"connector": "echo"})  # type: ignore[attr-defined]
        r.raise_for_status()
        items = r.json()["data"]
        assert any(t["id"] == token_id and t["revoked_at"] is None for t in items)

        r = await http_client.post(f"/v1/runtime-tokens/{token_id}/revoke")  # type: ignore[attr-defined]
        r.raise_for_status()
        assert r.json()["revoked_at"] is not None

        r = await http_client.get("/v1/runtime-tokens", params={"connector": "echo"})  # type: ignore[attr-defined]
        r.raise_for_status()
        items = r.json()["data"]
        revoked = next(t for t in items if t["id"] == token_id)
        assert revoked["revoked_at"] is not None

    async def test_runtime_token_rejects_legacy_route(
        self,
        http_client: object,
        aios_env: dict[str, str],
    ) -> None:
        """A runtime bearer must NOT authenticate ``GET /v1/connectors/secrets``
        (the legacy per-connection route).  The two auth deps share no
        token namespace; cross-pollination would be a privilege widening."""
        # Pre-create a connection so the legacy route has something to scope to.
        r = await http_client.post(  # type: ignore[attr-defined]
            "/v1/connections", json={"connector": "echo", "account": "acct-iso"}
        )
        r.raise_for_status()

        _token_id, plaintext = await _issue_runtime_token(http_client, "echo")
        # Per-request bearer override so we reuse ``http_client``'s ASGI
        # transport — building a fresh client against ``base_url`` would
        # try real DNS resolution on the in-process ``testserver`` host.
        r = await http_client.get(  # type: ignore[attr-defined]
            "/v1/connectors/secrets", headers=bearer(plaintext)
        )
        assert r.status_code == 401

    async def test_legacy_token_rejects_runtime_route(
        self,
        http_client: object,
        aios_env: dict[str, str],
    ) -> None:
        """And the reverse: a per-connection token can't reach the
        runtime-scoped routes."""
        r = await http_client.post(  # type: ignore[attr-defined]
            "/v1/connections", json={"connector": "echo", "account": "acct-iso2"}
        )
        r.raise_for_status()
        connection_id = r.json()["id"]
        plaintext = await _issue_connector_token(http_client, connection_id)

        r = await http_client.get(  # type: ignore[attr-defined]
            "/v1/connectors/runtime/secrets",
            params={"connection_id": connection_id},
            headers=bearer(plaintext),
        )
        assert r.status_code == 401

    async def test_revoked_runtime_token_401s(
        self,
        http_client: object,
        aios_env: dict[str, str],
    ) -> None:
        token_id, plaintext = await _issue_runtime_token(http_client, "echo")
        r = await http_client.post(f"/v1/runtime-tokens/{token_id}/revoke")  # type: ignore[attr-defined]
        r.raise_for_status()

        # Pre-create a connection of type "echo" so the runtime route
        # has a valid connection_id; the runtime token is revoked, so
        # the failure must come from auth, not from the connection check.
        r = await http_client.post(  # type: ignore[attr-defined]
            "/v1/connections", json={"connector": "echo", "account": "acct-rev"}
        )
        r.raise_for_status()
        connection_id = r.json()["id"]

        r = await http_client.get(  # type: ignore[attr-defined]
            "/v1/connectors/runtime/secrets",
            params={"connection_id": connection_id},
            headers=bearer(plaintext),
        )
        assert r.status_code == 401

    async def test_runtime_token_rejects_cross_type_connection(
        self,
        http_client: object,
        aios_env: dict[str, str],
    ) -> None:
        """A runtime bearer scoped to ``"echo"`` may not reach a
        ``"telegram"`` connection's secrets via the runtime-secrets
        route.  This is the headline guard on the new auth path."""
        # Echo connection + telegram connection (different connector types).
        r = await http_client.post(  # type: ignore[attr-defined]
            "/v1/connections", json={"connector": "echo", "account": "acct-cross-e"}
        )
        r.raise_for_status()
        r = await http_client.post(  # type: ignore[attr-defined]
            "/v1/connections", json={"connector": "telegram", "account": "acct-cross-t"}
        )
        r.raise_for_status()
        telegram_conn_id = r.json()["id"]

        _token_id, plaintext = await _issue_runtime_token(http_client, "echo")
        r = await http_client.get(  # type: ignore[attr-defined]
            "/v1/connectors/runtime/secrets",
            params={"connection_id": telegram_conn_id},
            headers=bearer(plaintext),
        )
        assert r.status_code == 403
