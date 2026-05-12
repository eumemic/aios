"""End-to-end validation gate for the multi-connection runtime path (#328 PR 5).

Spins up the reference echo-http connector against a real aios API
instance.  Tests here:

* Spawn a uvicorn server in-process (a real socket, not ASGITransport)
  so the connector's SSE streaming actually works.
* Issue a runtime token (per-type, not per-connection); create a
  connection + attach it + set legacy tools so the model sees them.
* Run an :class:`EchoConnector` as an asyncio task.  The runner
  discovers the connection via SSE, spawns its no-op worker, then
  dispatches tool calls.
* Drive the model via the harness so it calls a connection-tool.
* Verify the connector executes the tool and POSTs the result.
* Verify ``trigger_inbound`` synthesizes a session event via the
  runtime multipart route.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

import pytest
import uvicorn
from aios_echo_http import EchoConnector

from tests.conftest import needs_docker
from tests.e2e.harness import Harness, assistant, last_assistant_content, tool_call
from tests.helpers.connections import authed_client, issue_runtime_token, wait_for_health


@pytest.fixture
async def live_server(aios_env: dict[str, str]) -> AsyncIterator[str]:
    """Run uvicorn on a free port, serving the aios app.

    Returns the base URL.  Uses defer_wake mocking the same way the
    in-process http_client fixture does — tests drive the harness
    directly for any session advancement.
    """
    import socket
    from unittest import mock

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
        # Hand uvicorn the prebound socket so we don't race on port pick.
        sock.setblocking(False)
        await server.serve(sockets=[sock])

    with (
        mock.patch("aios.api.routers.sessions.defer_wake", new_callable=mock.AsyncMock),
        mock.patch("aios.services.inbound.defer_wake", new_callable=mock.AsyncMock),
    ):
        serve_task = asyncio.create_task(_serve())
        try:
            url = f"http://127.0.0.1:{port}"
            await wait_for_health(url)
            yield url
        finally:
            server.should_exit = True
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait_for(serve_task, timeout=5.0)
            await pool.close()


async def _create_connection(api_key: str, base_url: str, account: str) -> str:
    async with authed_client(base_url, api_key) as c:
        r = await c.post("/v1/connections", json={"connector": "echo", "account": account})
        r.raise_for_status()
        return str(r.json()["id"])


async def _set_tools(api_key: str, base_url: str, connection_id: str) -> None:
    async with authed_client(base_url, api_key) as c:
        r = await c.put(
            f"/v1/connections/{connection_id}/tools",
            json={
                "tools": [
                    {
                        "type": "custom",
                        "name": "ping",
                        "description": "ping",
                        "input_schema": {"type": "object"},
                    },
                    {
                        "type": "custom",
                        "name": "echo",
                        "description": "echo",
                        "input_schema": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                        },
                    },
                    {
                        "type": "custom",
                        "name": "trigger_inbound",
                        "description": "synth inbound",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "chat_id": {"type": "string"},
                                "sender_name": {"type": "string"},
                                "content": {"type": "string"},
                            },
                            "required": ["chat_id", "sender_name", "content"],
                        },
                    },
                ]
            },
        )
        r.raise_for_status()


@needs_docker
class TestSdkAgainstLiveServer:
    """Smoke test: SDK SSE round-trip against the uvicorn fixture."""

    async def test_runtime_calls_stream_emits_backfill(
        self,
        harness: Harness,
        live_server: str,
        aios_env: dict[str, str],
    ) -> None:
        """Pre-seed a pending call → open per-type calls SSE → confirm
        we receive it with the right ``connection_id``."""
        import json as _json

        from aios.services import agents as agents_service
        from aios.services import connections as connections_service
        from aios.services import environments as env_svc
        from aios.services import sessions as sess_svc
        from aios_sdk import Client, stream_connector_calls

        agent = await agents_service.create_agent(
            harness._pool,
            name=f"sse-bf-{id(self)}",
            model="fake/test",
            system="",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        env = await env_svc.create_environment(harness._pool, name=f"env-sse-{id(self)}")
        session = await sess_svc.create_session(
            harness._pool, agent_id=agent.id, environment_id=env.id, title=None, metadata={}
        )

        api_key = aios_env["AIOS_API_KEY"]
        connection_id = await _create_connection(api_key, live_server, f"acct-sse-{id(self)}")
        await connections_service.attach_connection(
            harness._pool, connection_id, session_id=session.id
        )
        await _set_tools(api_key, live_server, connection_id)
        token = await issue_runtime_token(api_key, live_server, "echo")

        # Pre-seed a pending call so backfill has something to deliver.
        await sess_svc.append_event(
            harness._pool,
            session.id,
            "message",
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_sse_bf",
                        "type": "function",
                        "function": {"name": "ping", "arguments": "{}"},
                    }
                ],
            },
        )
        await sess_svc.set_session_status(
            harness._pool,
            session.id,
            "idle",
            stop_reason={
                "type": "requires_action",
                "event_ids": ["call_sse_bf"],
                "custom_tools": ["call_sse_bf"],
            },
        )

        async with Client(base_url=live_server, token=token) as client:

            async def _first_call() -> dict[str, object]:
                async for msg in stream_connector_calls(client.get_async_httpx_client(), "echo"):
                    if msg.event == "call":
                        parsed: dict[str, object] = _json.loads(msg.data)
                        return parsed
                raise AssertionError("stream closed before any call")

            first = await asyncio.wait_for(_first_call(), timeout=10.0)
            assert first["tool_call_id"] == "call_sse_bf"
            assert first["name"] == "ping"
            assert first["connection_id"] == connection_id


@needs_docker
class TestEchoHttpConnectorEndToEnd:
    """Drive the full HTTP-client path: model → SSE → connector → tool-results."""

    async def test_echo_tool_round_trip(
        self,
        harness: Harness,
        live_server: str,
        aios_env: dict[str, str],
    ) -> None:
        from aios.services import agents as agents_service
        from aios.services import connections as connections_service
        from aios.services import environments as env_svc
        from aios.services import sessions as sess_svc

        agent = await agents_service.create_agent(
            harness._pool,
            name=f"e2e-{id(self)}",
            model="fake/test",
            system="",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        env = await env_svc.create_environment(harness._pool, name=f"env-e2e-{id(self)}")
        session = await sess_svc.create_session(
            harness._pool,
            agent_id=agent.id,
            environment_id=env.id,
            title=None,
            metadata={},
        )

        api_key = aios_env["AIOS_API_KEY"]
        connection_id = await _create_connection(api_key, live_server, f"acct-{id(self)}")
        await connections_service.attach_connection(
            harness._pool, connection_id, session_id=session.id
        )
        await _set_tools(api_key, live_server, connection_id)
        token = await issue_runtime_token(api_key, live_server, "echo")

        connector = EchoConnector(base_url=live_server, token=token)

        # Run the connector + drive the model in parallel.  The harness's
        # scripted model calls echo, the connector executes it, posts
        # the result, the harness's next step sees the result and emits
        # the wrap-up message.
        harness.script_model(
            [
                assistant(tool_calls=[tool_call("echo", {"text": "hello"}, call_id="call_1")]),
                assistant("All done."),
            ]
        )
        await sess_svc.append_user_message(harness._pool, session.id, "say hello")

        connector_task = asyncio.create_task(connector.run())
        try:
            # Step 1: model calls echo → session parks in requires_action,
            # which fires connector_calls NOTIFY → connector receives via SSE
            # → connector posts tool_result.
            await harness.run_step(session.id)

            # Wait for the connector to post the tool result.
            async def _result_landed() -> bool:
                events = await harness.events(session.id)
                return any(
                    e.kind == "message"
                    and e.data.get("role") == "tool"
                    and e.data.get("tool_call_id") == "call_1"
                    for e in events
                )

            deadline = asyncio.get_running_loop().time() + 10.0
            while not await _result_landed():
                if asyncio.get_running_loop().time() >= deadline:
                    raise AssertionError("connector did not post tool_result in 10s")
                await asyncio.sleep(0.1)

            # Step 2: model sees the tool_result and emits wrap-up.
            await harness.run_step(session.id)
            events = await harness.events(session.id)
            assert last_assistant_content(events) == "All done."

            # The tool result content carries echo's output.
            tool_event = next(
                e
                for e in events
                if e.kind == "message"
                and e.data.get("role") == "tool"
                and e.data.get("tool_call_id") == "call_1"
            )
            import json

            assert json.loads(tool_event.data["content"]) == {"text": "hello"}
        finally:
            connector_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await connector_task

    async def test_trigger_inbound_synthesizes_event(
        self,
        harness: Harness,
        live_server: str,
        aios_env: dict[str, str],
    ) -> None:
        """Driver: model calls trigger_inbound → connector POSTs to
        /v1/connectors/runtime/inbound (multipart) → user-message event lands."""
        from aios.services import agents as agents_service
        from aios.services import connections as connections_service
        from aios.services import environments as env_svc
        from aios.services import sessions as sess_svc

        agent = await agents_service.create_agent(
            harness._pool,
            name=f"e2e-i-{id(self)}",
            model="fake/test",
            system="",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        env = await env_svc.create_environment(harness._pool, name=f"env-e2e-i-{id(self)}")
        session = await sess_svc.create_session(
            harness._pool, agent_id=agent.id, environment_id=env.id, title=None, metadata={}
        )

        api_key = aios_env["AIOS_API_KEY"]
        connection_id = await _create_connection(api_key, live_server, f"acct-i-{id(self)}")
        await connections_service.attach_connection(
            harness._pool, connection_id, session_id=session.id
        )
        await _set_tools(api_key, live_server, connection_id)
        token = await issue_runtime_token(api_key, live_server, "echo")

        connector = EchoConnector(base_url=live_server, token=token)

        harness.script_model(
            [
                assistant(
                    tool_calls=[
                        tool_call(
                            "trigger_inbound",
                            {
                                "chat_id": "chat-1",
                                "sender_name": "Alice",
                                "content": "synthesized hi",
                            },
                            call_id="call_t1",
                        )
                    ]
                ),
                assistant("ack"),
            ]
        )
        await sess_svc.append_user_message(harness._pool, session.id, "fire trigger")

        connector_task = asyncio.create_task(connector.run())
        try:
            await harness.run_step(session.id)

            async def _inbound_landed() -> bool:
                events = await harness.events(session.id)
                return any(
                    e.kind == "message"
                    and e.data.get("role") == "user"
                    and e.data.get("content") == "synthesized hi"
                    for e in events
                )

            deadline = asyncio.get_running_loop().time() + 10.0
            while not await _inbound_landed():
                if asyncio.get_running_loop().time() >= deadline:
                    raise AssertionError("inbound never landed in 10s")
                await asyncio.sleep(0.1)
        finally:
            connector_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await connector_task
