"""End-to-end coverage for connection-attached secrets.

The flow under test (post-#328-PR-7):

* Operator creates a connection with a ``secrets`` dict.
* Operator GET shows ``secrets_set: true``; the values never appear.
* A runtime bearer for the connector type can fetch the decrypted dict
  via ``GET /v1/connectors/runtime/secrets?connection_id=...``.
* Operator rotates secrets via PUT; the runtime re-fetch sees the new values.
* A runtime token can read each connection of its type independently —
  it never crosses the type boundary.

These exercise the full encrypt-at-rest + connector-only-decrypt boundary
across ``CryptoBox`` + queries + service + routes.
"""

from __future__ import annotations

import httpx

from tests.conftest import needs_docker
from tests.e2e.test_echo_http_connector import live_server  # noqa: F401  fixture re-export
from tests.helpers.connections import authed_client, issue_runtime_token


async def _create_with_secrets(
    api_key: str,
    base_url: str,
    *,
    account: str,
    secrets: dict[str, str] | None,
) -> str:
    body: dict[str, object] = {"connector": "echo", "external_account_id": account}
    if secrets is not None:
        body["secrets"] = secrets
    async with authed_client(base_url, api_key) as c:
        r = await c.post("/v1/connections", json=body)
        r.raise_for_status()
        return str(r.json()["id"])


async def _get_connection(api_key: str, base_url: str, connection_id: str) -> dict[str, object]:
    async with authed_client(base_url, api_key) as c:
        r = await c.get(f"/v1/connections/{connection_id}")
        r.raise_for_status()
        return dict(r.json())


async def _set_secrets(
    api_key: str, base_url: str, connection_id: str, secrets: dict[str, str]
) -> None:
    async with authed_client(base_url, api_key) as c:
        r = await c.put(f"/v1/connections/{connection_id}/secrets", json={"secrets": secrets})
        r.raise_for_status()


async def _runtime_get_secrets(
    base_url: str, runtime_token: str, connection_id: str
) -> tuple[int, dict[str, object]]:
    async with authed_client(base_url, runtime_token) as c:
        r = await c.get("/v1/connectors/runtime/secrets", params={"connection_id": connection_id})
        # Successful responses always come back as JSON.  4xx/5xx may not —
        # let callers introspect by status alone.
        if r.status_code >= 400:
            return r.status_code, {}
        return r.status_code, dict(r.json())


@needs_docker
class TestConnectionSecretsRoundTrip:
    async def test_create_with_secrets_sets_flag_but_hides_values(
        self,
        live_server: str,  # noqa: F811
        aios_env: dict[str, str],
    ) -> None:
        api_key = aios_env["AIOS_API_KEY"]
        secret_value = "secret-canary-XYZ-123"
        cid = await _create_with_secrets(
            api_key,
            live_server,
            account=f"acct-flag-{id(self)}",
            secrets={"bot_token": secret_value},
        )
        view = await _get_connection(api_key, live_server, cid)
        assert view["secrets_set"] is True
        # No key called ``secrets`` / ``secret`` / ``bot_token`` on the
        # operator-facing read; the plaintext canary value must not appear
        # anywhere in the serialized response.
        assert "secrets" not in view
        assert "bot_token" not in str(view)
        assert secret_value not in str(view)

    async def test_create_without_secrets_leaves_flag_false(
        self,
        live_server: str,  # noqa: F811
        aios_env: dict[str, str],
    ) -> None:
        api_key = aios_env["AIOS_API_KEY"]
        cid = await _create_with_secrets(
            api_key, live_server, account=f"acct-noflag-{id(self)}", secrets=None
        )
        view = await _get_connection(api_key, live_server, cid)
        assert view["secrets_set"] is False

    async def test_create_with_empty_secrets_dict_equals_no_secrets(
        self,
        live_server: str,  # noqa: F811
        aios_env: dict[str, str],
    ) -> None:
        """Regression: ``create(secrets={})`` and ``set_secrets({})`` must
        produce the same row state. Both paths now treat empty as clear.
        """
        api_key = aios_env["AIOS_API_KEY"]
        cid = await _create_with_secrets(
            api_key, live_server, account=f"acct-empty-{id(self)}", secrets={}
        )
        view = await _get_connection(api_key, live_server, cid)
        assert view["secrets_set"] is False

    async def test_runtime_token_decrypts_secrets(
        self,
        live_server: str,  # noqa: F811
        aios_env: dict[str, str],
    ) -> None:
        api_key = aios_env["AIOS_API_KEY"]
        cid = await _create_with_secrets(
            api_key,
            live_server,
            account=f"acct-decrypt-{id(self)}",
            secrets={"bot_token": "abc123"},
        )
        token = await issue_runtime_token(api_key, live_server, "echo")
        status, body = await _runtime_get_secrets(live_server, token, cid)
        assert status == 200
        assert body == {"secrets": {"bot_token": "abc123"}}

    async def test_rotate_via_put_visible_to_runtime(
        self,
        live_server: str,  # noqa: F811
        aios_env: dict[str, str],
    ) -> None:
        api_key = aios_env["AIOS_API_KEY"]
        cid = await _create_with_secrets(
            api_key,
            live_server,
            account=f"acct-rot-{id(self)}",
            secrets={"bot_token": "old"},
        )
        token = await issue_runtime_token(api_key, live_server, "echo")
        _, before = await _runtime_get_secrets(live_server, token, cid)
        assert before == {"secrets": {"bot_token": "old"}}

        await _set_secrets(api_key, live_server, cid, {"bot_token": "new"})

        _, after = await _runtime_get_secrets(live_server, token, cid)
        assert after == {"secrets": {"bot_token": "new"}}

    async def test_clear_via_put_empty_dict(
        self,
        live_server: str,  # noqa: F811
        aios_env: dict[str, str],
    ) -> None:
        api_key = aios_env["AIOS_API_KEY"]
        cid = await _create_with_secrets(
            api_key,
            live_server,
            account=f"acct-clear-{id(self)}",
            secrets={"bot_token": "abc"},
        )
        await _set_secrets(api_key, live_server, cid, {})
        view = await _get_connection(api_key, live_server, cid)
        assert view["secrets_set"] is False

        token = await issue_runtime_token(api_key, live_server, "echo")
        _, body = await _runtime_get_secrets(live_server, token, cid)
        assert body == {"secrets": {}}

    async def test_unauthenticated_request_rejected(
        self,
        live_server: str,  # noqa: F811
    ) -> None:
        async with httpx.AsyncClient(base_url=live_server) as c:
            r = await c.get("/v1/connectors/runtime/secrets", params={"connection_id": "x"})
        assert r.status_code in (401, 403)

    async def test_operator_token_cannot_read_secrets_endpoint(
        self,
        live_server: str,  # noqa: F811
        aios_env: dict[str, str],
    ) -> None:
        """The runtime-scoped route refuses the operator API key — secrets
        read-back is exclusively for runtime bearers, not AIOS_API_KEY.
        """
        api_key = aios_env["AIOS_API_KEY"]
        async with authed_client(live_server, api_key) as c:
            r = await c.get("/v1/connectors/runtime/secrets", params={"connection_id": "x"})
        assert r.status_code in (401, 403)
