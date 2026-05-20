"""End-to-end tests for runtime-token ``connection_ids`` allowlist scope (#350).

A runtime token issued with ``connection_ids=[...]`` is authorized only
on the listed connection IDs:
* the connection-discovery SSE backfill emits only the allowlisted set;
* runtime-scoped routes that name an out-of-scope ``connection_id``
  return 403.

Tokens issued without ``connection_ids`` retain the pre-#350 behaviour
— they see every connection of the bearer's connector type.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
from collections.abc import AsyncIterator
from unittest import mock

import httpx
import pytest
import uvicorn

from aios_sdk import Client, stream_connection_discovery
from tests.conftest import needs_docker
from tests.helpers.connections import authed_client, bearer


async def _wait_for_health(url: str, *, deadline_s: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + deadline_s
    async with httpx.AsyncClient() as client:
        while True:
            with contextlib.suppress(httpx.HTTPError):
                r = await client.get(f"{url}/v1/health", timeout=0.5)
                if r.status_code < 500:
                    return
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(f"server at {url} not ready in {deadline_s}s")
            await asyncio.sleep(0.05)


@pytest.fixture
async def live_server(aios_env: dict[str, str]) -> AsyncIterator[str]:
    """Real-uvicorn aios so SSE chunked-transfer streaming works end-to-end.

    Mirrors ``test_connection_discovery_sse.live_server`` — extracted-
    but-not-shared because per-file pytest fixtures keep the setup
    local to the assertions that depend on them.
    """
    from aios.api.app import create_app
    from aios.config import get_settings
    from aios.crypto.vault import CryptoBox
    from aios.db.pool import create_pool

    settings = get_settings()
    pool = await create_pool(settings.db_url, min_size=1, max_size=4)
    app = create_app()
    app.state.pool = pool
    app.state.crypto_box = CryptoBox.from_base64(settings.vault_key.get_secret_value())
    app.state.db_url = settings.db_url
    app.state.procrastinate = mock.MagicMock()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="off")
    server = uvicorn.Server(config)
    server.config.load()
    server.lifespan = server.config.lifespan_class(server.config)

    async def _serve() -> None:
        sock.setblocking(False)
        await server.serve(sockets=[sock])

    with (
        mock.patch("aios.api.routers.sessions.defer_wake", new_callable=mock.AsyncMock),
        mock.patch("aios.services.inbound.defer_wake", new_callable=mock.AsyncMock),
    ):
        serve_task = asyncio.create_task(_serve())
        try:
            url = f"http://127.0.0.1:{port}"
            await _wait_for_health(url)
            yield url
        finally:
            await server.shutdown()
            serve_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await serve_task
            await pool.close()


async def _create_connection(api_key: str, base_url: str, account: str) -> str:
    async with authed_client(base_url, api_key) as c:
        r = await c.post(
            "/v1/connections", json={"connector": "echo", "external_account_id": account}
        )
        r.raise_for_status()
        return str(r.json()["id"])


async def _issue_runtime_token(
    api_key: str,
    base_url: str,
    connector: str,
    *,
    connection_ids: list[str] | None = None,
) -> str:
    """Mint a runtime token; optionally bind it to a connection allowlist."""
    body: dict[str, object] = {"connector": connector}
    if connection_ids is not None:
        body["connection_ids"] = connection_ids
    async with authed_client(base_url, api_key) as c:
        r = await c.post("/v1/runtime-tokens", json=body)
        r.raise_for_status()
        return str(r.json()["plaintext"])


async def _collect_backfill_ids(
    base_url: str,
    token: str,
    *,
    expected_ids: set[str],
    timeout_s: float = 5.0,
) -> set[str]:
    """Open the discovery SSE stream and collect ``added`` IDs until
    ``expected_ids`` is a subset of what we've seen — then cancel.

    Discovery backfill emits one ``added`` per active connection of the
    bearer's type before tailing NOTIFY; once we've seen every ID we
    expect, anything still streaming is the tail, which we don't care
    about for the backfill assertion.
    """
    seen: set[str] = set()
    done = asyncio.Event()

    async def _consume() -> None:
        async with Client(base_url=base_url, token=token) as client:
            async for msg in stream_connection_discovery(client.get_async_httpx_client(), "echo"):
                if msg.event != "connection":
                    continue
                payload = json.loads(msg.data)
                if payload["event"] == "added":
                    seen.add(payload["connection_id"])
                    if expected_ids.issubset(seen):
                        done.set()

    task = asyncio.create_task(_consume())
    try:
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(done.wait(), timeout=timeout_s)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    return seen


@needs_docker
class TestRuntimeTokenConnectionScope:
    async def test_scoped_token_discovery_emits_only_allowlisted(
        self,
        live_server: str,
        aios_env: dict[str, str],
    ) -> None:
        """Three echo connections A/B/C; scope to [A, B]; backfill must
        emit A and B (in any order) and must NOT emit C."""
        api_key = aios_env["AIOS_API_KEY"]
        a = await _create_connection(api_key, live_server, "acct-scope-a")
        b = await _create_connection(api_key, live_server, "acct-scope-b")
        c = await _create_connection(api_key, live_server, "acct-scope-c")

        token = await _issue_runtime_token(api_key, live_server, "echo", connection_ids=[a, b])

        seen = await _collect_backfill_ids(live_server, token, expected_ids={a, b}, timeout_s=5.0)
        assert a in seen
        assert b in seen
        assert c not in seen

    async def test_unscoped_token_sees_all_connections(
        self,
        live_server: str,
        aios_env: dict[str, str],
    ) -> None:
        """Token issued without ``connection_ids`` retains pre-#350
        behaviour: every active connection of the bearer's type is
        emitted in backfill."""
        api_key = aios_env["AIOS_API_KEY"]
        a = await _create_connection(api_key, live_server, "acct-unscoped-a")
        b = await _create_connection(api_key, live_server, "acct-unscoped-b")
        c = await _create_connection(api_key, live_server, "acct-unscoped-c")

        token = await _issue_runtime_token(api_key, live_server, "echo")

        seen = await _collect_backfill_ids(
            live_server, token, expected_ids={a, b, c}, timeout_s=5.0
        )
        assert {a, b, c}.issubset(seen)


@needs_docker
class TestRuntimeTokenConnectionScopeRoutes:
    """403 / 200 boundary checks on the runtime-scoped routes when the
    bearer's allowlist excludes (or includes) the target connection.

    These don't need live SSE streaming, so the in-process
    ``http_client`` ASGI transport is enough.
    """

    async def test_scoped_token_403_on_inbound_for_out_of_scope(
        self,
        http_client: object,
        aios_env: dict[str, str],
    ) -> None:
        r = await http_client.post(  # type: ignore[attr-defined]
            "/v1/connections", json={"connector": "echo", "external_account_id": "acct-in-a"}
        )
        r.raise_for_status()
        a_id = r.json()["id"]
        r = await http_client.post(  # type: ignore[attr-defined]
            "/v1/connections", json={"connector": "echo", "external_account_id": "acct-in-b"}
        )
        r.raise_for_status()
        b_id = r.json()["id"]

        r = await http_client.post(  # type: ignore[attr-defined]
            "/v1/runtime-tokens",
            json={"connector": "echo", "connection_ids": [a_id]},
        )
        r.raise_for_status()
        plaintext = r.json()["plaintext"]

        r = await http_client.post(  # type: ignore[attr-defined]
            "/v1/connectors/runtime/inbound",
            data={
                "connection_id": b_id,
                "event_id": "evt-out-of-scope-1",
                "chat_id": "chat-1",
                "content": "hi",
            },
            headers=bearer(plaintext),
        )
        assert r.status_code == 403

    async def test_scoped_token_403_on_tool_results_for_out_of_scope(
        self,
        http_client: object,
        aios_env: dict[str, str],
    ) -> None:
        r = await http_client.post(  # type: ignore[attr-defined]
            "/v1/connections", json={"connector": "echo", "external_account_id": "acct-tr-a"}
        )
        r.raise_for_status()
        a_id = r.json()["id"]
        r = await http_client.post(  # type: ignore[attr-defined]
            "/v1/connections", json={"connector": "echo", "external_account_id": "acct-tr-b"}
        )
        r.raise_for_status()
        b_id = r.json()["id"]

        r = await http_client.post(  # type: ignore[attr-defined]
            "/v1/runtime-tokens",
            json={"connector": "echo", "connection_ids": [a_id]},
        )
        r.raise_for_status()
        plaintext = r.json()["plaintext"]

        r = await http_client.post(  # type: ignore[attr-defined]
            "/v1/connectors/runtime/tool-results",
            json={
                "connection_id": b_id,
                "session_id": "sess_nope",
                "tool_call_id": "tcid_nope",
                "content": "{}",
            },
            headers=bearer(plaintext),
        )
        assert r.status_code == 403

    async def test_scoped_token_403_on_secrets_for_out_of_scope(
        self,
        http_client: object,
        aios_env: dict[str, str],
    ) -> None:
        r = await http_client.post(  # type: ignore[attr-defined]
            "/v1/connections", json={"connector": "echo", "external_account_id": "acct-sec-a"}
        )
        r.raise_for_status()
        a_id = r.json()["id"]
        r = await http_client.post(  # type: ignore[attr-defined]
            "/v1/connections", json={"connector": "echo", "external_account_id": "acct-sec-b"}
        )
        r.raise_for_status()
        b_id = r.json()["id"]

        r = await http_client.post(  # type: ignore[attr-defined]
            "/v1/runtime-tokens",
            json={"connector": "echo", "connection_ids": [a_id]},
        )
        r.raise_for_status()
        plaintext = r.json()["plaintext"]

        r = await http_client.get(  # type: ignore[attr-defined]
            "/v1/connectors/runtime/secrets",
            params={"connection_id": b_id},
            headers=bearer(plaintext),
        )
        assert r.status_code == 403

    async def test_scoped_token_200_on_secrets_for_in_scope(
        self,
        http_client: object,
        aios_env: dict[str, str],
    ) -> None:
        r = await http_client.post(  # type: ignore[attr-defined]
            "/v1/connections", json={"connector": "echo", "external_account_id": "acct-sec-ok"}
        )
        r.raise_for_status()
        a_id = r.json()["id"]

        r = await http_client.post(  # type: ignore[attr-defined]
            "/v1/runtime-tokens",
            json={"connector": "echo", "connection_ids": [a_id]},
        )
        r.raise_for_status()
        plaintext = r.json()["plaintext"]

        r = await http_client.get(  # type: ignore[attr-defined]
            "/v1/connectors/runtime/secrets",
            params={"connection_id": a_id},
            headers=bearer(plaintext),
        )
        assert r.status_code == 200

    async def test_unscoped_token_inbound_succeeds_for_any_connection(
        self,
        http_client: object,
        aios_env: dict[str, str],
    ) -> None:
        """Pre-#350 behaviour preserved when ``connection_ids`` is omitted:
        the inbound route does not 403 on connection-scope (it may
        still fail later in the pipeline — we assert ``!= 403`` rather
        than ``== 201``)."""
        r = await http_client.post(  # type: ignore[attr-defined]
            "/v1/connections", json={"connector": "echo", "external_account_id": "acct-un-any"}
        )
        r.raise_for_status()
        a_id = r.json()["id"]

        r = await http_client.post(  # type: ignore[attr-defined]
            "/v1/runtime-tokens", json={"connector": "echo"}
        )
        r.raise_for_status()
        plaintext = r.json()["plaintext"]

        r = await http_client.post(  # type: ignore[attr-defined]
            "/v1/connectors/runtime/inbound",
            data={
                "connection_id": a_id,
                "event_id": "evt-un-any-1",
                "chat_id": "chat-1",
                "content": "hi",
            },
            headers=bearer(plaintext),
        )
        assert r.status_code != 403
