"""E2E tests for the connector-facing inbound + calls-stream endpoints (#301).

Pins the contract a connector container talks to:

* ``POST /v1/connectors/inbound`` appends a user-message event to the
  session bound to the caller's connection.  Idempotent on ``event_id``
  (replays are no-ops with ``deduped=True``).  Drops on detached /
  oversized / archived-template / attachment-failure surface as 4xx/5xx
  with a ``drop_reason`` body.

* ``GET /v1/connectors/calls`` (SSE) emits one event per pending custom
  tool call referencing a tool declared on the caller's connection.
  Backfills at subscribe time, then tails ``connector_calls_<cid>``.

Both routes use ``ConnectorAuthDep`` — the caller's bearer token
resolves to a single connection_id, never the global API key.
"""

from __future__ import annotations

import httpx

from aios.ids import EVENT, make_id, split_id
from tests.conftest import needs_docker
from tests.e2e.harness import Harness
from tests.helpers.connections import bearer


def _new_event_id() -> str:
    """A fresh 26-char ULID — the dedup-key shape connectors emit."""
    return split_id(make_id(EVENT))[1]


async def _create_connection(http_client: httpx.AsyncClient, account: str) -> str:
    r = await http_client.post(
        "/v1/connections",
        json={"connector": "echo", "account": account},
    )
    assert r.status_code == 201, r.text
    return str(r.json()["id"])


async def _attach(harness: Harness, connection_id: str, session_id: str) -> None:
    """Bypass the API attach (which calls into procrastinate for the
    snapshot drift check, unavailable in tests) — go straight to the service.
    """
    from aios.services import connections as connections_service

    await connections_service.attach_connection(harness._pool, connection_id, session_id=session_id)


async def _set_tools(
    http_client: httpx.AsyncClient, connection_id: str, tool_names: list[str]
) -> None:
    tools = [
        {
            "type": "custom",
            "name": name,
            "description": f"send via {name}",
            "input_schema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        }
        for name in tool_names
    ]
    r = await http_client.put(
        f"/v1/connections/{connection_id}/tools",
        json={"tools": tools},
    )
    assert r.status_code == 200, r.text


async def _issue_token(http_client: httpx.AsyncClient, connection_id: str) -> str:
    r = await http_client.post("/v1/connector-tokens", json={"connection_id": connection_id})
    assert r.status_code == 201, r.text
    return str(r.json()["plaintext"])


@needs_docker
class TestPostInbound:
    async def test_appends_user_event(
        self, http_client: httpx.AsyncClient, harness: Harness
    ) -> None:
        from aios.services import agents as agents_service
        from aios.services import environments as env_svc
        from aios.services import sessions as sess_svc

        agent = await agents_service.create_agent(
            harness._pool,
            name=f"inb-{id(self)}",
            model="fake/test",
            system="",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        env = await env_svc.create_environment(harness._pool, name=f"env-inb-{id(self)}")
        session = await sess_svc.create_session(
            harness._pool, agent_id=agent.id, environment_id=env.id, title=None, metadata={}
        )
        connection_id = await _create_connection(http_client, f"acct-{id(self)}")
        await _attach(harness, connection_id, session.id)
        token = await _issue_token(http_client, connection_id)

        r = await http_client.post(
            "/v1/connectors/inbound",
            headers=bearer(token),
            json={
                "event_id": _new_event_id(),
                "chat_id": "chat-1",
                "sender": {"display_name": "Alice"},
                "content": "hello",
            },
        )
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["session_id"] == session.id
        assert body["deduped"] is False

        events = await harness.events(session.id)
        user_events = [e for e in events if e.kind == "message" and e.data.get("role") == "user"]
        assert len(user_events) == 1
        assert user_events[0].data["content"] == "hello"
        assert user_events[0].data["metadata"]["sender"] == "Alice"

    async def test_dedup_returns_same_event(
        self, http_client: httpx.AsyncClient, harness: Harness
    ) -> None:
        from aios.services import agents as agents_service
        from aios.services import environments as env_svc
        from aios.services import sessions as sess_svc

        agent = await agents_service.create_agent(
            harness._pool,
            name=f"dedup-{id(self)}",
            model="fake/test",
            system="",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        env = await env_svc.create_environment(harness._pool, name=f"env-dedup-{id(self)}")
        session = await sess_svc.create_session(
            harness._pool, agent_id=agent.id, environment_id=env.id, title=None, metadata={}
        )
        connection_id = await _create_connection(http_client, f"acct-d-{id(self)}")
        await _attach(harness, connection_id, session.id)
        token = await _issue_token(http_client, connection_id)

        event_id = _new_event_id()
        body = {
            "event_id": event_id,
            "chat_id": "chat-1",
            "sender": {"display_name": "Alice"},
            "content": "hello",
        }

        r1 = await http_client.post(
            "/v1/connectors/inbound",
            headers=bearer(token),
            json=body,
        )
        assert r1.status_code == 201
        assert r1.json()["deduped"] is False

        r2 = await http_client.post(
            "/v1/connectors/inbound",
            headers=bearer(token),
            json=body,
        )
        assert r2.status_code == 201
        assert r2.json()["deduped"] is True
        assert r2.json()["session_id"] == session.id

        # No duplicate event in the log.
        events = await harness.events(session.id)
        user_events = [e for e in events if e.kind == "message" and e.data.get("role") == "user"]
        assert len(user_events) == 1

    async def test_detached_connection_drops_with_422(self, http_client: httpx.AsyncClient) -> None:
        connection_id = await _create_connection(http_client, f"acct-det-{id(self)}")
        token = await _issue_token(http_client, connection_id)

        r = await http_client.post(
            "/v1/connectors/inbound",
            headers=bearer(token),
            json={
                "event_id": _new_event_id(),
                "chat_id": "chat-1",
                "sender": {},
                "content": "hi",
            },
        )
        assert r.status_code == 422, r.text
        assert r.json()["error"]["detail"]["drop_reason"] == "detached"

    async def test_operator_key_rejected(
        self, http_client: httpx.AsyncClient, aios_env: dict[str, str]
    ) -> None:
        r = await http_client.post(
            "/v1/connectors/inbound",
            headers=bearer(aios_env["AIOS_API_KEY"]),
            json={
                "event_id": "01J00000000000000000000000",
                "chat_id": "chat-1",
                "sender": {},
                "content": "hi",
            },
        )
        assert r.status_code == 401, r.text
