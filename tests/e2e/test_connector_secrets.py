"""End-to-end coverage for connection-attached secrets.

The flow under test:

* Operator creates a connection with a ``secrets`` dict.
* Operator GET shows ``secrets_set: true``; the values never appear.
* Connector token resolves to that connection; ``GET /v1/connectors/secrets``
  returns the decrypted dict.
* Operator rotates secrets via PUT; connector re-fetch sees the new values.
* Connector A's bearer token cannot read connection B's secrets.

These exercise the full encrypt-at-rest + connector-only-decrypt boundary
across ``CryptoBox`` + queries + service + routes.
"""

from __future__ import annotations

import httpx
import pytest

from tests.conftest import needs_docker
from tests.e2e.test_echo_http_connector import live_server  # noqa: F401  fixture re-export


async def _create_with_secrets(
    api_key: str,
    base_url: str,
    *,
    account: str,
    secrets: dict[str, str] | None,
) -> str:
    async with httpx.AsyncClient(
        base_url=base_url, headers={"Authorization": f"Bearer {api_key}"}
    ) as c:
        body: dict[str, object] = {"connector": "echo", "account": account}
        if secrets is not None:
            body["secrets"] = secrets
        r = await c.post("/v1/connections", json=body)
        r.raise_for_status()
        return str(r.json()["id"])


async def _get_connection(api_key: str, base_url: str, connection_id: str) -> dict[str, object]:
    async with httpx.AsyncClient(
        base_url=base_url, headers={"Authorization": f"Bearer {api_key}"}
    ) as c:
        r = await c.get(f"/v1/connections/{connection_id}")
        r.raise_for_status()
        return dict(r.json())


async def _set_secrets(
    api_key: str, base_url: str, connection_id: str, secrets: dict[str, str]
) -> None:
    async with httpx.AsyncClient(
        base_url=base_url, headers={"Authorization": f"Bearer {api_key}"}
    ) as c:
        r = await c.put(f"/v1/connections/{connection_id}/secrets", json={"secrets": secrets})
        r.raise_for_status()


async def _issue_token(api_key: str, base_url: str, connection_id: str) -> str:
    async with httpx.AsyncClient(
        base_url=base_url, headers={"Authorization": f"Bearer {api_key}"}
    ) as c:
        r = await c.post("/v1/connector-tokens", json={"connection_id": connection_id})
        r.raise_for_status()
        return str(r.json()["plaintext"])


async def _connector_get_secrets(
    base_url: str, connector_token: str
) -> tuple[int, dict[str, object]]:
    async with httpx.AsyncClient(
        base_url=base_url, headers={"Authorization": f"Bearer {connector_token}"}
    ) as c:
        r = await c.get("/v1/connectors/secrets")
        try:
            return r.status_code, dict(r.json())
        except Exception:
            return r.status_code, {}


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
        produce the same row state.  The earlier draft had divergent
        semantics (create stored an encrypted-empty-dict blob, set
        cleared) — which would have made operator intent depend on
        which endpoint they used.  Both paths now treat empty as clear.
        """
        api_key = aios_env["AIOS_API_KEY"]
        cid = await _create_with_secrets(
            api_key, live_server, account=f"acct-empty-{id(self)}", secrets={}
        )
        view = await _get_connection(api_key, live_server, cid)
        assert view["secrets_set"] is False

    async def test_connector_token_decrypts_own_secrets(
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
        token = await _issue_token(api_key, live_server, cid)

        status, body = await _connector_get_secrets(live_server, token)

        assert status == 200
        assert body == {"secrets": {"bot_token": "abc123"}}

    async def test_rotate_via_put_visible_to_connector(
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
        token = await _issue_token(api_key, live_server, cid)
        _, before = await _connector_get_secrets(live_server, token)
        assert before == {"secrets": {"bot_token": "old"}}

        await _set_secrets(api_key, live_server, cid, {"bot_token": "new"})

        _, after = await _connector_get_secrets(live_server, token)
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

        token = await _issue_token(api_key, live_server, cid)
        _, body = await _connector_get_secrets(live_server, token)
        assert body == {"secrets": {}}

    async def test_one_token_cannot_read_anothers_secrets(
        self,
        live_server: str,  # noqa: F811
        aios_env: dict[str, str],
    ) -> None:
        api_key = aios_env["AIOS_API_KEY"]
        cid_a = await _create_with_secrets(
            api_key, live_server, account=f"acct-a-{id(self)}", secrets={"bot_token": "A"}
        )
        cid_b = await _create_with_secrets(
            api_key, live_server, account=f"acct-b-{id(self)}", secrets={"bot_token": "B"}
        )
        token_a = await _issue_token(api_key, live_server, cid_a)
        token_b = await _issue_token(api_key, live_server, cid_b)

        # A's token sees A's secret only.
        _, a_view = await _connector_get_secrets(live_server, token_a)
        assert a_view == {"secrets": {"bot_token": "A"}}

        _, b_view = await _connector_get_secrets(live_server, token_b)
        assert b_view == {"secrets": {"bot_token": "B"}}

    async def test_unauthenticated_request_rejected(
        self,
        live_server: str,  # noqa: F811
    ) -> None:
        async with httpx.AsyncClient(base_url=live_server) as c:
            r = await c.get("/v1/connectors/secrets")
        assert r.status_code in (401, 403)

    async def test_operator_token_cannot_read_secrets_endpoint(
        self,
        live_server: str,  # noqa: F811
        aios_env: dict[str, str],
    ) -> None:
        """The connector-scoped route refuses operator credentials —
        secrets read-back is exclusively for the connector container's
        own bearer token, not for AIOS_API_KEY.
        """
        api_key = aios_env["AIOS_API_KEY"]
        async with httpx.AsyncClient(
            base_url=live_server, headers={"Authorization": f"Bearer {api_key}"}
        ) as c:
            r = await c.get("/v1/connectors/secrets")
        assert r.status_code in (401, 403)


# Re-export marker to keep the lint/import clean.
_ = pytest
