"""E2E tests for connection-declared tools (#301).

Connections can carry a ``tools`` jsonb column (migration 0029).  When a
session has connections attached — single_session, per_chat origin, or
operator-bound chat — the connection's ``type="custom"`` tools are
merged into the model's tool list at step time.  The model calls them,
the session parks in ``requires_action``, an external client (a
connector container) executes the tool and POSTs the result back.

This file pins the contract end-to-end with the real harness:

* Tools are sourced via ``services.connections.list_tools_for_session``.
* Step prelude includes them alongside agent + MCP + connector-subprocess tools.
* The standard custom-tool flow (requires_action → tool-results → resume)
  works regardless of whether the tool came from an agent or a connection.
* Multimodal tool-result content (``list[dict]``) flows through ``POST
  /v1/sessions/:id/tool-results`` intact.
"""

from __future__ import annotations

import json

import httpx

from tests.conftest import needs_docker
from tests.e2e.harness import Harness, assistant, last_assistant_content, tool_call


@needs_docker
class TestConnectionToolsInPrelude:
    """A connection's tools become available to the model when the
    connection is attached to (or originates) the session.
    """

    async def test_attached_connection_tools_visible_to_model(self, harness: Harness) -> None:
        from aios.harness.step_context import compute_step_prelude
        from aios.models.agents import ToolSpec
        from aios.services import agents as agents_service
        from aios.services import connections as connections_service
        from aios.services import environments as env_svc
        from aios.services import sessions as sess_svc

        # Agent with NO tools — the only tool will come from the connection.
        agent = await agents_service.create_agent(
            harness._pool,
            name=f"conn-tool-prelude-{id(self)}",
            model="fake/test",
            system="You are a test assistant.",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        env = await env_svc.create_environment(harness._pool, name=f"env-{id(self)}")
        session = await sess_svc.create_session(
            harness._pool,
            agent_id=agent.id,
            environment_id=env.id,
            title="conn-tool-prelude",
            metadata={},
        )

        # Connection with a single custom tool, attached to the session.
        connection = await connections_service.create_connection(
            harness._pool,
            connector="echo",
            account="echo-1",
            metadata={},
            tools=[
                ToolSpec(
                    type="custom",
                    name="chat_send",
                    description="Send a chat message to the user",
                    input_schema={
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                ),
            ],
        )
        await connections_service.attach_connection(
            harness._pool, connection.id, session_id=session.id
        )

        prelude = await compute_step_prelude(
            pool=harness._pool,
            session_id=session.id,
            session=await sess_svc.get_session(harness._pool, session.id),
            agent=agent,
            channels=[],
            memory_store_echoes=[],
        )

        names = [t["function"]["name"] for t in prelude.tools]
        assert "chat_send" in names, (
            f"connection tool absent from prelude — model will never see schema. tools={names}"
        )
        # Schema flows through unchanged.
        chat_send = next(t for t in prelude.tools if t["function"]["name"] == "chat_send")
        assert chat_send["function"]["parameters"]["properties"]["text"]["type"] == "string"

    async def test_per_chat_origin_connection_tools_visible(self, harness: Harness) -> None:
        """A session spawned by a per_chat connection sees that connection's tools.

        This is the lineage path used in production: a Telegram bot in
        per_chat mode spawns a fresh session for each chat partner, and
        that session must see the bot's send/edit/react tools even though
        the connection isn't directly ``session_id``-attached.
        """
        from aios.harness.step_context import compute_step_prelude
        from aios.models.agents import ToolSpec
        from aios.services import agents as agents_service
        from aios.services import connections as connections_service
        from aios.services import environments as env_svc
        from aios.services import session_templates as templates_svc
        from aios.services import sessions as sess_svc

        agent = await agents_service.create_agent(
            harness._pool,
            name=f"per-chat-{id(self)}",
            model="fake/test",
            system="",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        env = await env_svc.create_environment(harness._pool, name=f"env-pc-{id(self)}")
        template = await templates_svc.create_session_template(
            harness._pool,
            name=f"tpl-{id(self)}",
            agent_id=agent.id,
            agent_version=None,
            environment_id=env.id,
            vault_ids=[],
            memory_store_ids=[],
            metadata={},
        )

        connection = await connections_service.create_connection(
            harness._pool,
            connector="echo",
            account="echo-pc",
            metadata={},
            tools=[
                ToolSpec(
                    type="custom",
                    name="bot_send",
                    description="Send via bot",
                    input_schema={"type": "object", "properties": {}},
                ),
            ],
        )
        await connections_service.configure_per_chat(
            harness._pool, connection.id, session_template_id=template.id
        )

        # Spawn a session as if from inbound, recording the connection lineage.
        session = await sess_svc.create_session(
            harness._pool,
            agent_id=agent.id,
            environment_id=env.id,
            title="spawned",
            metadata={},
            spawned_from_connection_id=connection.id,
        )

        prelude = await compute_step_prelude(
            pool=harness._pool,
            session_id=session.id,
            session=await sess_svc.get_session(harness._pool, session.id),
            agent=agent,
            channels=[],
            memory_store_echoes=[],
        )
        names = [t["function"]["name"] for t in prelude.tools]
        assert "bot_send" in names, f"per_chat origin tool absent from prelude. tools={names}"


@needs_docker
class TestConnectionToolDispatch:
    """Full requires_action round-trip with a connection-sourced tool."""

    async def test_model_calls_connection_tool_then_resumes_after_result(
        self, harness: Harness
    ) -> None:
        from aios.models.agents import ToolSpec
        from aios.services import agents as agents_service
        from aios.services import connections as connections_service
        from aios.services import environments as env_svc
        from aios.services import sessions as sess_svc

        agent = await agents_service.create_agent(
            harness._pool,
            name=f"dispatch-{id(self)}",
            model="fake/test",
            system="",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        env = await env_svc.create_environment(harness._pool, name=f"env-d-{id(self)}")
        session = await sess_svc.create_session(
            harness._pool,
            agent_id=agent.id,
            environment_id=env.id,
            title="dispatch",
            metadata={},
        )
        connection = await connections_service.create_connection(
            harness._pool,
            connector="echo",
            account="echo-d",
            metadata={},
            tools=[
                ToolSpec(
                    type="custom",
                    name="chat_send",
                    description="Send",
                    input_schema={
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                ),
            ],
        )
        await connections_service.attach_connection(
            harness._pool, connection.id, session_id=session.id
        )

        # Model calls the connection tool, then (after the result lands)
        # produces a final assistant message.
        harness.script_model(
            [
                assistant(tool_calls=[tool_call("chat_send", {"text": "hi"}, call_id="cs_1")]),
                assistant("Done."),
            ]
        )
        await sess_svc.append_user_message(harness._pool, session.id, "say hi")

        # Step 1: model calls chat_send → session parks in requires_action.
        await harness.run_step(session.id)
        s = await harness.session(session.id)
        assert s.status == "idle"
        assert s.stop_reason is not None
        assert s.stop_reason["type"] == "requires_action"
        assert "cs_1" in (s.stop_reason.get("custom_tools") or [])

        # External client (the connector container, in production) POSTs the result.
        await sess_svc.append_event(
            harness._pool,
            session.id,
            "message",
            {"role": "tool", "tool_call_id": "cs_1", "content": "ok", "name": "chat_send"},
        )

        # Step 2: session resumes, model produces the wrap-up.
        await harness.run_step(session.id)
        events = await harness.events(session.id)
        assert last_assistant_content(events) == "Done."
        s = await harness.session(session.id)
        assert s.stop_reason == {"type": "end_turn"}


@needs_docker
class TestMultimodalToolResults:
    """``POST /v1/sessions/:id/tool-results`` accepts ``content: list[dict]``."""

    async def test_list_content_round_trips_intact(
        self, http_client: httpx.AsyncClient, harness: Harness
    ) -> None:
        from aios.models.agents import ToolSpec
        from aios.services import agents as agents_service
        from aios.services import environments as env_svc
        from aios.services import sessions as sess_svc

        agent = await agents_service.create_agent(
            harness._pool,
            name=f"mm-{id(self)}",
            model="fake/test",
            system="",
            tools=[
                ToolSpec(
                    type="custom",
                    name="fetch_image",
                    description="",
                    input_schema={"type": "object"},
                ),
            ],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        env = await env_svc.create_environment(harness._pool, name=f"env-mm-{id(self)}")
        session = await sess_svc.create_session(
            harness._pool,
            agent_id=agent.id,
            environment_id=env.id,
            title="mm",
            metadata={},
        )
        # Seed an assistant tool_call so /tool-results has a parent to bind to.
        await sess_svc.append_user_message(harness._pool, session.id, "fetch")
        await sess_svc.append_event(
            harness._pool,
            session.id,
            "message",
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "mm_1",
                        "type": "function",
                        "function": {"name": "fetch_image", "arguments": "{}"},
                    }
                ],
            },
        )

        multimodal_content = [
            {"type": "text", "text": "Here's the image:"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/cat.png"},
            },
        ]
        r = await http_client.post(
            f"/v1/sessions/{session.id}/tool-results",
            json={"tool_call_id": "mm_1", "content": multimodal_content},
        )
        assert r.status_code == 201, r.text
        event = r.json()

        # Stored exactly as posted — events.data is jsonb.
        stored_content = event["data"]["content"]
        assert stored_content == multimodal_content
        # And the tool_name was promoted from the parent assistant event.
        assert event["data"]["name"] == "fetch_image"

    async def test_string_content_still_works(
        self, http_client: httpx.AsyncClient, harness: Harness
    ) -> None:
        """Backwards-compat: existing clients that POST a plain string keep working."""
        from aios.models.agents import ToolSpec
        from aios.services import agents as agents_service
        from aios.services import environments as env_svc
        from aios.services import sessions as sess_svc

        agent = await agents_service.create_agent(
            harness._pool,
            name=f"mm-str-{id(self)}",
            model="fake/test",
            system="",
            tools=[
                ToolSpec(
                    type="custom",
                    name="fetch_text",
                    description="",
                    input_schema={"type": "object"},
                ),
            ],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        env = await env_svc.create_environment(harness._pool, name=f"env-str-{id(self)}")
        session = await sess_svc.create_session(
            harness._pool,
            agent_id=agent.id,
            environment_id=env.id,
            title="str",
            metadata={},
        )
        await sess_svc.append_user_message(harness._pool, session.id, "fetch")
        await sess_svc.append_event(
            harness._pool,
            session.id,
            "message",
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "str_1",
                        "type": "function",
                        "function": {"name": "fetch_text", "arguments": "{}"},
                    }
                ],
            },
        )

        r = await http_client.post(
            f"/v1/sessions/{session.id}/tool-results",
            json={"tool_call_id": "str_1", "content": "plain string"},
        )
        assert r.status_code == 201, r.text
        assert r.json()["data"]["content"] == "plain string"


@needs_docker
class TestSetConnectionToolsEndpoint:
    """``PUT /v1/connections/{id}/tools`` replaces the connection's tools."""

    async def test_replaces_tools(self, http_client: httpx.AsyncClient, harness: Harness) -> None:
        from aios.services import connections as connections_service

        connection = await connections_service.create_connection(
            harness._pool,
            connector="echo",
            account=f"echo-set-{id(self)}",
            metadata={},
        )
        assert connection.tools == []

        r = await http_client.put(
            f"/v1/connections/{connection.id}/tools",
            json={
                "tools": [
                    {
                        "type": "custom",
                        "name": "x_send",
                        "description": "send",
                        "input_schema": {"type": "object"},
                    }
                ],
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["tools"]) == 1
        assert body["tools"][0]["name"] == "x_send"

    async def test_rejects_non_custom_tool_type(
        self, http_client: httpx.AsyncClient, harness: Harness
    ) -> None:
        from aios.services import connections as connections_service

        connection = await connections_service.create_connection(
            harness._pool,
            connector="echo",
            account=f"echo-rej-{id(self)}",
            metadata={},
        )

        r = await http_client.put(
            f"/v1/connections/{connection.id}/tools",
            json={"tools": [{"type": "bash"}]},
        )
        assert r.status_code == 422, r.text
        assert "custom" in json.dumps(r.json())
