"""Connection discovery SSE — backfill + ``added`` / ``removed`` (#328 PR 5).

The runtime container subscribes once per ``connector`` type and learns
about the connections it should service via:

1. Subscribe-time backfill — every active connection of the bearer's
   type emits an ``added`` event.
2. Live tail — :func:`aios.services.connections.attach_connection` emits
   ``added`` on a NOTIFY channel; :func:`archive_connection` emits
   ``removed``.

These tests use a real uvicorn server (not ASGITransport) so the SSE
chunked-transfer streaming actually exercises the dual-NOTIFY-emit
plumbing end-to-end.
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
from tests.helpers.connections import authed_client


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
    """Same shape as ``test_echo_http_connector.live_server`` —
    extracted-but-not-shared because per-file pytest fixtures keep the
    setup local to the assertions that depend on them.
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
            # See test_signal_registration.py for why ``should_exit`` alone
            # loses the race against open SSE streams.
            await server.shutdown()
            serve_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await serve_task
            await pool.close()


async def _create_connection(api_key: str, base_url: str, account: str) -> str:
    async with authed_client(base_url, api_key) as c:
        r = await c.post("/v1/connections", json={"connector": "echo", "account": account})
        r.raise_for_status()
        return str(r.json()["id"])


async def _issue_runtime_token(api_key: str, base_url: str, connector: str) -> str:
    async with authed_client(base_url, api_key) as c:
        r = await c.post("/v1/runtime-tokens", json={"connector": connector})
        r.raise_for_status()
        return str(r.json()["plaintext"])


@needs_docker
class TestConnectionDiscoverySse:
    async def test_backfill_added_then_attach_emits_added_then_archive_emits_removed(
        self,
        live_server: str,
        aios_env: dict[str, str],
    ) -> None:
        """End-to-end: subscribe → confirm backfill → attach a new
        connection → see ``added`` → archive → see ``removed``.

        ``archive_connection`` refuses to archive while attached, so we
        detach first.
        """
        api_key = aios_env["AIOS_API_KEY"]

        # Pre-existing connection that should show up in backfill.
        pre_id = await _create_connection(api_key, live_server, "acct-disc-pre")

        token = await _issue_runtime_token(api_key, live_server, "echo")

        events_seen: list[dict[str, str]] = []
        backfilled = asyncio.Event()
        saw_added = asyncio.Event()
        saw_removed = asyncio.Event()
        added_id: dict[str, str] = {}

        async def _consume() -> None:
            async with Client(base_url=live_server, token=token) as client:
                async for msg in stream_connection_discovery(
                    client.get_async_httpx_client(), "echo"
                ):
                    if msg.event != "connection":
                        continue
                    payload = json.loads(msg.data)
                    events_seen.append(payload)
                    if (
                        payload["event"] == "added"
                        and payload["connection_id"] == pre_id
                        and not backfilled.is_set()
                    ):
                        backfilled.set()
                    elif (
                        payload["event"] == "added"
                        and "new_id" in added_id
                        and payload["connection_id"] == added_id["new_id"]
                    ):
                        saw_added.set()
                    elif (
                        payload["event"] == "removed"
                        and "new_id" in added_id
                        and payload["connection_id"] == added_id["new_id"]
                    ):
                        saw_removed.set()

        consumer_task = asyncio.create_task(_consume())
        try:
            await asyncio.wait_for(backfilled.wait(), timeout=5.0)

            # Now create + attach a new connection of the same type;
            # ``attach_connection`` is what fires the ``added`` NOTIFY.
            from aios.services import agents as agents_service
            from aios.services import connections as connections_service
            from aios.services import environments as env_svc
            from aios.services import sessions as sess_svc

            new_id = await _create_connection(api_key, live_server, "acct-disc-new")
            added_id["new_id"] = new_id

            from aios.config import get_settings
            from aios.db.pool import create_pool

            settings = get_settings()
            pool = await create_pool(settings.db_url, min_size=1, max_size=2)
            try:
                account_id = "acc_test_stub"  # PR 3 scaffolding
                agent = await agents_service.create_agent(
                    pool,
                    name=f"disc-{id(self)}",
                    model="fake/test",
                    system="",
                    tools=[],
                    description=None,
                    metadata={},
                    window_min=50_000,
                    window_max=150_000,
                    account_id=account_id,
                )
                env = await env_svc.create_environment(
                    pool, name=f"env-disc-{id(self)}", account_id=account_id
                )
                session = await sess_svc.create_session(
                    pool,
                    agent_id=agent.id,
                    environment_id=env.id,
                    title=None,
                    metadata={},
                    account_id=account_id,
                )
                await connections_service.attach_connection(
                    pool, new_id, session_id=session.id, account_id=account_id
                )
                await asyncio.wait_for(saw_added.wait(), timeout=5.0)

                # Detach + archive — ``archive_connection`` emits ``removed``.
                await connections_service.detach_connection(pool, new_id, account_id=account_id)
                await connections_service.archive_connection(pool, new_id, account_id=account_id)
                await asyncio.wait_for(saw_removed.wait(), timeout=5.0)
            finally:
                await pool.close()
        finally:
            consumer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await consumer_task

        # Sanity: backfilled the pre-existing connection BEFORE the new one.
        ids = [e["connection_id"] for e in events_seen if e["event"] == "added"]
        assert pre_id in ids
        assert added_id["new_id"] in ids
