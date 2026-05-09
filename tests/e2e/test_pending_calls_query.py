"""E2E tests for the connector-calls primitives (#301).

The SSE endpoint at ``GET /v1/connectors/calls`` is glue over two
primitives:

1. ``queries.list_pending_calls_for_connection`` — backfill query
2. ``listen_for_connector_calls(db_url, connection_id)`` — LISTEN on
   the per-connection NOTIFY channel, fired by ``append_event`` when an
   assistant tool-calls event lands on a bound session

This file pins both primitives directly.  End-to-end SSE exercise with
a real consumer lands in PR 4 with the reference echo-http connector,
where the consumer's reconnect/dedup logic naturally drives the stream.
"""

from __future__ import annotations

import asyncio
import json

from aios.config import get_settings
from aios.crypto.vault import CryptoBox
from aios.db.listen import listen_for_connector_calls
from tests.conftest import needs_docker
from tests.e2e.harness import Harness


@needs_docker
class TestListPendingCalls:
    async def test_returns_pending_call_for_attached_session(
        self, harness: Harness, crypto_box: CryptoBox
    ) -> None:
        from aios.db import queries
        from aios.models.agents import ToolSpec
        from aios.services import agents as agents_service
        from aios.services import connections as connections_service
        from aios.services import environments as env_svc
        from aios.services import sessions as sess_svc

        agent = await agents_service.create_agent(
            harness._pool,
            name=f"q-{id(self)}",
            model="fake/test",
            system="",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        env = await env_svc.create_environment(harness._pool, name=f"env-q-{id(self)}")
        session = await sess_svc.create_session(
            harness._pool, agent_id=agent.id, environment_id=env.id, title=None, metadata={}
        )
        conn = await connections_service.create_connection(
            harness._pool,
            connector="echo",
            account=f"acct-q-{id(self)}",
            metadata={},
            crypto_box=crypto_box,
            tools=[
                ToolSpec(
                    type="custom",
                    name="echo_send",
                    description="",
                    input_schema={"type": "object"},
                ),
            ],
        )
        await connections_service.attach_connection(harness._pool, conn.id, session_id=session.id)

        # Seed an assistant tool_call + park session in requires_action.
        await sess_svc.append_event(
            harness._pool,
            session.id,
            "message",
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_q1",
                        "type": "function",
                        "function": {"name": "echo_send", "arguments": json.dumps({"text": "hi"})},
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
                "event_ids": ["call_q1"],
                "custom_tools": ["call_q1"],
            },
        )

        async with harness._pool.acquire() as db_conn:
            calls = await queries.list_pending_calls_for_connection(db_conn, conn.id)

        assert len(calls) == 1
        assert calls[0]["tool_call_id"] == "call_q1"
        assert calls[0]["name"] == "echo_send"
        assert calls[0]["session_id"] == session.id
        assert json.loads(calls[0]["arguments"]) == {"text": "hi"}

    async def test_returns_empty_for_unrelated_connection(
        self, harness: Harness, crypto_box: CryptoBox
    ) -> None:
        from aios.db import queries
        from aios.models.agents import ToolSpec
        from aios.services import connections as connections_service

        # A connection with tools but not bound to anything.
        conn = await connections_service.create_connection(
            harness._pool,
            connector="echo",
            account=f"acct-u-{id(self)}",
            metadata={},
            crypto_box=crypto_box,
            tools=[
                ToolSpec(
                    type="custom",
                    name="echo_send",
                    description="",
                    input_schema={"type": "object"},
                ),
            ],
        )
        async with harness._pool.acquire() as db_conn:
            calls = await queries.list_pending_calls_for_connection(db_conn, conn.id)
        assert calls == []


@needs_docker
class TestConnectorCallsNotify:
    """Verify ``set_session_status`` fires ``connector_calls_<cid>`` NOTIFY
    when a session parks in ``requires_action`` with pending custom_tools,
    for every connection bound to that session.

    The NOTIFY fires here (not in :func:`append_event`) so the SSE
    consumer's lookup query can rely on ``stop_reason`` already being
    committed.  Otherwise there's a race: assistant event lands → NOTIFY
    fires → SSE consumer queries → stop_reason still null → empty result.
    """

    async def test_notify_fires_on_requires_action_park(
        self, harness: Harness, crypto_box: CryptoBox
    ) -> None:
        from aios.models.agents import ToolSpec
        from aios.services import agents as agents_service
        from aios.services import connections as connections_service
        from aios.services import environments as env_svc
        from aios.services import sessions as sess_svc

        agent = await agents_service.create_agent(
            harness._pool,
            name=f"n-{id(self)}",
            model="fake/test",
            system="",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        env = await env_svc.create_environment(harness._pool, name=f"env-n-{id(self)}")
        session = await sess_svc.create_session(
            harness._pool, agent_id=agent.id, environment_id=env.id, title=None, metadata={}
        )
        conn = await connections_service.create_connection(
            harness._pool,
            connector="echo",
            account=f"acct-n-{id(self)}",
            metadata={},
            crypto_box=crypto_box,
            tools=[
                ToolSpec(
                    type="custom",
                    name="echo_send",
                    description="",
                    input_schema={"type": "object"},
                ),
            ],
        )
        await connections_service.attach_connection(harness._pool, conn.id, session_id=session.id)

        settings = get_settings()
        async with listen_for_connector_calls(settings.db_url, conn.id) as queue:
            await sess_svc.set_session_status(
                harness._pool,
                session.id,
                "idle",
                stop_reason={
                    "type": "requires_action",
                    "event_ids": ["call_n1"],
                    "custom_tools": ["call_n1"],
                },
            )
            payload = await asyncio.wait_for(queue.get(), timeout=5.0)
            assert payload == session.id

    async def test_notify_silent_on_non_requires_action_park(
        self, harness: Harness, crypto_box: CryptoBox
    ) -> None:
        """``end_turn`` and other stop reasons MUST NOT trigger the
        connector-calls fan-out — only ``requires_action`` with pending
        ``custom_tools`` represents new work for the connector.
        """
        from aios.models.agents import ToolSpec
        from aios.services import agents as agents_service
        from aios.services import connections as connections_service
        from aios.services import environments as env_svc
        from aios.services import sessions as sess_svc

        agent = await agents_service.create_agent(
            harness._pool,
            name=f"nu-{id(self)}",
            model="fake/test",
            system="",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        env = await env_svc.create_environment(harness._pool, name=f"env-nu-{id(self)}")
        session = await sess_svc.create_session(
            harness._pool, agent_id=agent.id, environment_id=env.id, title=None, metadata={}
        )
        conn = await connections_service.create_connection(
            harness._pool,
            connector="echo",
            account=f"acct-nu-{id(self)}",
            metadata={},
            crypto_box=crypto_box,
            tools=[
                ToolSpec(
                    type="custom",
                    name="echo_send",
                    description="",
                    input_schema={"type": "object"},
                ),
            ],
        )
        await connections_service.attach_connection(harness._pool, conn.id, session_id=session.id)

        settings = get_settings()
        async with listen_for_connector_calls(settings.db_url, conn.id) as queue:
            await sess_svc.set_session_status(
                harness._pool,
                session.id,
                "idle",
                stop_reason={"type": "end_turn"},
            )
            try:
                payload = await asyncio.wait_for(queue.get(), timeout=0.5)
            except TimeoutError:
                payload = None
            assert payload is None, f"unexpected NOTIFY for end_turn park: {payload}"

    async def test_notify_silent_on_unrelated_connection(
        self, harness: Harness, crypto_box: CryptoBox
    ) -> None:
        """An assistant tool_calls event on session A must NOT NOTIFY
        a connection bound to session B.  Scope is the connection's
        bound sessions, not all sessions.
        """
        from aios.models.agents import ToolSpec
        from aios.services import agents as agents_service
        from aios.services import connections as connections_service
        from aios.services import environments as env_svc
        from aios.services import sessions as sess_svc

        agent = await agents_service.create_agent(
            harness._pool,
            name=f"nu-{id(self)}",
            model="fake/test",
            system="",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        env = await env_svc.create_environment(harness._pool, name=f"env-nu-{id(self)}")
        session_a = await sess_svc.create_session(
            harness._pool, agent_id=agent.id, environment_id=env.id, title=None, metadata={}
        )
        session_b = await sess_svc.create_session(
            harness._pool, agent_id=agent.id, environment_id=env.id, title=None, metadata={}
        )
        # conn_b is bound only to session_b; conn_a is bound to session_a.
        conn_a = await connections_service.create_connection(
            harness._pool,
            connector="echo",
            account=f"acct-nu-A-{id(self)}",
            metadata={},
            crypto_box=crypto_box,
            tools=[
                ToolSpec(type="custom", name="x", description="", input_schema={"type": "object"}),
            ],
        )
        conn_b = await connections_service.create_connection(
            harness._pool,
            connector="echo",
            account=f"acct-nu-B-{id(self)}",
            metadata={},
            crypto_box=crypto_box,
            tools=[
                ToolSpec(type="custom", name="x", description="", input_schema={"type": "object"}),
            ],
        )
        await connections_service.attach_connection(
            harness._pool, conn_a.id, session_id=session_a.id
        )
        await connections_service.attach_connection(
            harness._pool, conn_b.id, session_id=session_b.id
        )

        settings = get_settings()
        # Listen as conn_b but park session_a — conn_b shouldn't see it.
        async with listen_for_connector_calls(settings.db_url, conn_b.id) as queue:
            await sess_svc.set_session_status(
                harness._pool,
                session_a.id,
                "idle",
                stop_reason={
                    "type": "requires_action",
                    "event_ids": ["call_nu1"],
                    "custom_tools": ["call_nu1"],
                },
            )
            try:
                payload = await asyncio.wait_for(queue.get(), timeout=0.5)
            except TimeoutError:
                payload = None
            assert payload is None, f"unexpected NOTIFY across connections: {payload}"
