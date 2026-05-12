"""E2E coverage for the multi-connection Signal connector (#328 PR 6).

Spins up uvicorn aios in-process and runs a real ``SignalConnector``
against it with a mocked :class:`SignalDaemon`.  Asserts:

* Per-account inbound routing: an envelope tagged with phone A's
  ``account`` lands as a user-message on session A — NOT session B.
* Per-account outbound routing: a model-driven ``signal_send`` from
  session A's tool call lands as an RPC call with ``params["account"]``
  equal to phone A — NOT phone B.

The whole point of PR 6 is the demux on both directions; one e2e test
proves the wiring.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import pytest
import uvicorn

from tests.conftest import needs_docker
from tests.e2e.harness import Harness, assistant, last_assistant_content, tool_call
from tests.helpers.connections import authed_client, issue_runtime_token, wait_for_health

# Two phones, two connections — the demux subjects.
PHONE_A = "+15550000111"
PHONE_B = "+15550000222"
BOT_UUID_A = "aaaaaaaa-1111-1111-1111-111111111111"
BOT_UUID_B = "bbbbbbbb-2222-2222-2222-222222222222"
ALICE_UUID = "cccccccc-3333-3333-3333-333333333333"


@pytest.fixture
async def live_server(aios_env: dict[str, str]) -> AsyncIterator[str]:
    """Uvicorn aios on a free port — same shape as the echo e2e fixture."""
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
            await wait_for_health(url)
            yield url
        finally:
            server.should_exit = True
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait_for(serve_task, timeout=5.0)
            await pool.close()


async def _create_signal_connection(
    api_key: str, base_url: str, account: str, secrets: dict[str, str]
) -> str:
    async with authed_client(base_url, api_key) as c:
        r = await c.post(
            "/v1/connections",
            json={"connector": "signal", "account": account, "secrets": secrets},
        )
        r.raise_for_status()
        return str(r.json()["id"])


async def _set_signal_tools(api_key: str, base_url: str, connection_id: str) -> None:
    """Publish the signal tool schemas onto a connection.

    PR 7 will cut the model over to ``connectors.tools_schema``; in PR 6
    the model still sees the per-connection list, so each test connection
    needs its own (identical) copy.
    """
    async with authed_client(base_url, api_key) as c:
        r = await c.put(
            f"/v1/connections/{connection_id}/tools",
            json={
                "tools": [
                    {
                        "type": "custom",
                        "name": "signal_send",
                        "description": "Send a Signal message.",
                        "input_schema": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                        },
                    },
                ]
            },
        )
        r.raise_for_status()


class _FakeListener:
    """Stand-in for :class:`aios_signal.rpc.RpcListener`.

    Tests push envelopes into ``feed`` as ``(account, envelope)`` pairs;
    ``messages()`` is an async generator that yields them in order and
    then waits for more.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()

    async def feed(self, account: str, envelope: dict[str, Any]) -> None:
        await self._queue.put((account, envelope))

    async def messages(self) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        while True:
            yield await self._queue.get()


def _build_text_envelope(sender_uuid: str, text: str) -> dict[str, Any]:
    """Minimal signal-cli envelope shape that :func:`parse_envelope` accepts."""
    return {
        "envelope": {
            "source": sender_uuid,
            "sourceName": "Alice",
            "sourceUuid": sender_uuid,
            "sourceDevice": 1,
            "timestamp": 1700000000000,
            "dataMessage": {
                "timestamp": 1700000000000,
                "message": text,
                "expiresInSeconds": 0,
                "viewOnce": False,
            },
        }
    }


@pytest.fixture
def mocked_signal_daemon(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace :class:`SignalDaemon` with a fake. Returns the fake plus
    the per-test inbound listener so tests can feed envelopes in.

    ``verify_phone`` returns a deterministic uuid-per-phone so the
    connector's per-connection bot_uuid maps to {A: BOT_UUID_A, B: BOT_UUID_B}.
    ``rpc.call`` is a ``MagicMock`` capturing every ``send`` / ``listGroups``
    call the connector makes — tests assert ``params["account"]`` to
    verify outbound routing.
    """
    listener = _FakeListener()
    rpc = MagicMock()
    rpc.call = AsyncMock(return_value=None)

    daemon = MagicMock()
    daemon.listener = listener
    daemon.rpc = rpc
    daemon.verify_phone = AsyncMock(
        side_effect=lambda phone: {PHONE_A: BOT_UUID_A, PHONE_B: BOT_UUID_B}[phone]
    )
    daemon.list_contacts = AsyncMock(return_value={})
    daemon.list_groups = AsyncMock(return_value=[])
    daemon.__aenter__ = AsyncMock(return_value=daemon)
    daemon.__aexit__ = AsyncMock(return_value=None)

    monkeypatch.setattr(
        "aios_signal.connector.SignalDaemon",
        MagicMock(return_value=daemon),
    )
    return {"daemon": daemon, "listener": listener, "rpc_call": rpc.call}


@needs_docker
class TestSignalMultiConnection:
    async def test_inbound_routes_to_correct_session(
        self,
        harness: Harness,
        live_server: str,
        aios_env: dict[str, str],
        mocked_signal_daemon: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """An inbound envelope tagged with phone A's ``account`` lands as
        a user-message event on session A; an envelope tagged with B's
        account lands on session B."""
        from aios_signal.config import Settings
        from aios_signal.connector import SignalConnector

        from aios.services import agents as agents_service
        from aios.services import connections as connections_service
        from aios.services import environments as env_svc
        from aios.services import sessions as sess_svc

        # Two sessions, two connections, two phones.
        agent = await agents_service.create_agent(
            harness._pool,
            name=f"sig-{id(self)}",
            model="fake/test",
            system="",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        env = await env_svc.create_environment(harness._pool, name=f"env-sig-{id(self)}")
        session_a = await sess_svc.create_session(
            harness._pool, agent_id=agent.id, environment_id=env.id, title=None, metadata={}
        )
        session_b = await sess_svc.create_session(
            harness._pool, agent_id=agent.id, environment_id=env.id, title=None, metadata={}
        )

        api_key = aios_env["AIOS_API_KEY"]
        conn_a = await _create_signal_connection(api_key, live_server, PHONE_A, {"phone": PHONE_A})
        conn_b = await _create_signal_connection(api_key, live_server, PHONE_B, {"phone": PHONE_B})
        await connections_service.attach_connection(harness._pool, conn_a, session_id=session_a.id)
        await connections_service.attach_connection(harness._pool, conn_b, session_id=session_b.id)
        await _set_signal_tools(api_key, live_server, conn_a)
        await _set_signal_tools(api_key, live_server, conn_b)
        token = await issue_runtime_token(api_key, live_server, "signal")

        cfg = Settings(config_dir=tmp_path / "cfg", cli_bin="/usr/bin/signal-cli")
        connector = SignalConnector(cfg)
        connector._base_url = live_server
        connector._token = token

        connector_task = asyncio.create_task(connector.run())
        listener: _FakeListener = mocked_signal_daemon["listener"]
        try:
            # Push an envelope for phone A → expect a user-message on session A.
            await listener.feed(PHONE_A, _build_text_envelope(ALICE_UUID, "hello-A"))
            await listener.feed(PHONE_B, _build_text_envelope(ALICE_UUID, "hello-B"))

            async def _both_landed() -> bool:
                events_a = await harness.events(session_a.id)
                events_b = await harness.events(session_b.id)
                has_a = any(
                    e.kind == "message"
                    and e.data.get("role") == "user"
                    and "hello-A" in (e.data.get("content") or "")
                    for e in events_a
                )
                has_b = any(
                    e.kind == "message"
                    and e.data.get("role") == "user"
                    and "hello-B" in (e.data.get("content") or "")
                    for e in events_b
                )
                return has_a and has_b

            deadline = asyncio.get_running_loop().time() + 10.0
            while not await _both_landed():
                if asyncio.get_running_loop().time() >= deadline:
                    raise AssertionError("inbound never demuxed onto both sessions")
                await asyncio.sleep(0.1)

            # Cross-routing: session A must NOT have B's message and vice versa.
            events_a = await harness.events(session_a.id)
            events_b = await harness.events(session_b.id)
            assert not any("hello-B" in (e.data.get("content") or "") for e in events_a)
            assert not any("hello-A" in (e.data.get("content") or "") for e in events_b)
        finally:
            connector_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await connector_task

    async def test_outbound_send_routes_to_correct_phone(
        self,
        harness: Harness,
        live_server: str,
        aios_env: dict[str, str],
        mocked_signal_daemon: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """The model calling ``signal_send`` on session A's tool call
        lands as an RPC ``send`` with ``account=PHONE_A``, not PHONE_B."""
        from aios_signal.config import Settings
        from aios_signal.connector import SignalConnector

        from aios.services import agents as agents_service
        from aios.services import connections as connections_service
        from aios.services import environments as env_svc
        from aios.services import sessions as sess_svc

        agent = await agents_service.create_agent(
            harness._pool,
            name=f"sig-out-{id(self)}",
            model="fake/test",
            system="",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        env = await env_svc.create_environment(harness._pool, name=f"env-sigo-{id(self)}")
        session_a = await sess_svc.create_session(
            harness._pool, agent_id=agent.id, environment_id=env.id, title=None, metadata={}
        )
        session_b = await sess_svc.create_session(
            harness._pool, agent_id=agent.id, environment_id=env.id, title=None, metadata={}
        )

        api_key = aios_env["AIOS_API_KEY"]
        conn_a = await _create_signal_connection(
            api_key, live_server, PHONE_A + "-out", {"phone": PHONE_A}
        )
        conn_b = await _create_signal_connection(
            api_key, live_server, PHONE_B + "-out", {"phone": PHONE_B}
        )
        await connections_service.attach_connection(harness._pool, conn_a, session_id=session_a.id)
        await connections_service.attach_connection(harness._pool, conn_b, session_id=session_b.id)
        await _set_signal_tools(api_key, live_server, conn_a)
        await _set_signal_tools(api_key, live_server, conn_b)
        token = await issue_runtime_token(api_key, live_server, "signal")

        cfg = Settings(config_dir=tmp_path / "cfg", cli_bin="/usr/bin/signal-cli")
        connector = SignalConnector(cfg)
        connector._base_url = live_server
        connector._token = token

        harness.script_model(
            [
                assistant(
                    tool_calls=[
                        tool_call("signal_send", {"text": "hi-from-A"}, call_id="call_send")
                    ]
                ),
                assistant("done"),
            ]
        )
        await sess_svc.append_user_message(harness._pool, session_a.id, "ping")

        connector_task = asyncio.create_task(connector.run())
        rpc_call: AsyncMock = mocked_signal_daemon["rpc_call"]
        try:
            await harness.run_step(session_a.id)

            async def _signal_send_called() -> bool:
                return any(c.args and c.args[0] == "send" for c in rpc_call.call_args_list)

            deadline = asyncio.get_running_loop().time() + 10.0
            while not await _signal_send_called():
                if asyncio.get_running_loop().time() >= deadline:
                    raise AssertionError("signal_send never reached the daemon RPC")
                await asyncio.sleep(0.1)

            # Outbound demux: the ONLY send call's account must be phone A.
            send_calls = [c for c in rpc_call.call_args_list if c.args and c.args[0] == "send"]
            assert len(send_calls) == 1
            params = send_calls[0].args[1]
            assert params["account"] == PHONE_A
            assert params["account"] != PHONE_B
            assert params["message"] == "hi-from-A"

            # Drive the wrap-up step so harness state stays clean.
            await harness.run_step(session_a.id)
            events = await harness.events(session_a.id)
            assert last_assistant_content(events) == "done"
        finally:
            connector_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await connector_task
