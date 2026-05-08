"""Unit tests for connector-subprocess tool enumeration.

The ``ConnectorSubprocessRegistry`` owns long-lived stdio MCP sessions
for every connector instance.  For the model to actually call those
tools — instead of pattern-matching tool *names* from prior history —
the harness has to enumerate them at step-prelude time and merge them
into the ``tools[]`` payload sent to LiteLLM.

These tests pin two pieces of that contract:

* ``ConnectorSubprocessRegistry.list_tools()`` walks the running
  instances, calls each session's ``list_tools()``, and returns
  OpenAI-format tool dicts namespaced ``mcp__<connector>__<tool>``.

* ``compute_step_prelude`` calls into that registry and the resulting
  ``StepPrelude.tools`` includes the connector tools alongside builtin
  + HTTP MCP tools.

Both the SDK and the registry's per-instance state are mocked; we're
only exercising orchestration.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aios_connector import ConnectorSpec

from aios.config import Settings
from aios.harness.connector_supervisor import (
    ConnectorState,
    ConnectorSubprocessRegistry,
)


def _fake_tool(name: str, description: str = "", schema: dict[str, Any] | None = None) -> Any:
    """Minimal ``mcp.types.Tool`` stand-in: only the attributes the
    enumeration code reads (``name``, ``description``, ``inputSchema``).
    Using ``SimpleNamespace`` keeps us off the real pydantic model so
    schema-evolution upstream doesn't break these tests.
    """
    return SimpleNamespace(
        name=name,
        description=description,
        inputSchema=schema or {"type": "object", "properties": {}},
    )


def _registry_with_running_instance(
    *,
    connector: str,
    instance: str,
    tools: list[Any],
) -> ConnectorSubprocessRegistry:
    """Build a registry whose ``(connector, instance)`` state has a
    fake live session whose ``list_tools()`` returns ``tools``.
    """
    spec = ConnectorSpec(name=connector, command="/bin/true", args=[])
    state = ConnectorState(connector=connector, instance=instance, spec=spec)
    state.status = "running"
    fake_session = MagicMock()
    fake_session.list_tools = AsyncMock(return_value=SimpleNamespace(tools=tools))
    state.session = fake_session
    state.ready.set()

    registry = ConnectorSubprocessRegistry([], settings=Settings())
    # Inject the prepared state directly; we're skipping the supervisor
    # loop entirely (that's covered by other tests).
    registry._states[(connector, instance)] = state
    return registry


class TestRegistryListTools:
    async def test_returns_namespaced_openai_tools_for_running_instance(self) -> None:
        registry = _registry_with_running_instance(
            connector="telegram",
            instance="telegram",
            tools=[
                _fake_tool(
                    "telegram_send",
                    description="Send a message",
                    schema={
                        "type": "object",
                        "properties": {
                            "account": {"type": "string"},
                            "chat_id": {"type": "string"},
                            "text": {"type": "string"},
                        },
                        "required": ["account", "chat_id", "text"],
                    },
                )
            ],
        )

        tools = await registry.list_tools()

        assert len(tools) == 1
        tool = tools[0]
        assert tool["type"] == "function"
        fn = tool["function"]
        assert fn["name"] == "mcp__telegram__telegram_send"
        assert fn["description"] == "Send a message"
        # Schema flows through unchanged so the model sees parameter shapes.
        assert "account" in fn["parameters"]["properties"]
        assert "text" in fn["parameters"]["properties"]

    async def test_skips_instances_that_arent_running(self) -> None:
        spec = ConnectorSpec(name="signal", command="/bin/true", args=[])
        state = ConnectorState(connector="signal", instance="signal", spec=spec)
        # status defaults to "starting"; ready is not set; session is None.
        registry = ConnectorSubprocessRegistry([], settings=Settings())
        registry._states[("signal", "signal")] = state

        tools = await registry.list_tools()

        assert tools == []

    async def test_dedupes_tools_across_instances_of_same_connector(self) -> None:
        """Multi-instance connectors (telegram:bot1, telegram:bot2)
        publish the same toolset.  The model sees one ``mcp__telegram__*``
        namespace, so duplicates from sibling instances must be collapsed.
        """
        registry = _registry_with_running_instance(
            connector="telegram",
            instance="bot1",
            tools=[_fake_tool("telegram_send")],
        )
        # Add a second instance with the same tool.
        spec = ConnectorSpec(name="telegram", command="/bin/true", args=[])
        bot2_state = ConnectorState(connector="telegram", instance="bot2", spec=spec)
        bot2_state.status = "running"
        fake_session = MagicMock()
        fake_session.list_tools = AsyncMock(
            return_value=SimpleNamespace(tools=[_fake_tool("telegram_send")])
        )
        bot2_state.session = fake_session
        bot2_state.ready.set()
        registry._states[("telegram", "bot2")] = bot2_state

        tools = await registry.list_tools()

        names = [t["function"]["name"] for t in tools]
        assert names == ["mcp__telegram__telegram_send"]


class TestStepPreludeIncludesConnectorTools:
    """Integration over ``compute_step_prelude``: with a connector
    registry on ``runtime``, the connector tools must appear in
    ``StepPrelude.tools``.  This is the bug the parent investigation
    surfaced: a live worker request had 51 tools, none of which were
    connector tools, because tool assembly never consulted the registry.
    """

    @pytest.fixture(autouse=True)
    def _restore_registry(self) -> Any:
        """Clear ``runtime.connector_subprocess_registry`` after each test
        so a fixture isn't visible to unrelated tests in the same run.
        """
        from aios.harness import runtime

        before = runtime.connector_subprocess_registry
        try:
            yield
        finally:
            runtime.connector_subprocess_registry = before

    async def test_connector_tools_appear_in_prelude(self) -> None:
        from aios.harness import runtime
        from aios.harness.step_context import compute_step_prelude

        runtime.connector_subprocess_registry = _registry_with_running_instance(
            connector="signal",
            instance="signal",
            tools=[
                _fake_tool(
                    "signal_send",
                    description="Send a Signal message",
                    schema={
                        "type": "object",
                        "properties": {
                            "account": {"type": "string"},
                            "recipient": {"type": "string"},
                            "text": {"type": "string"},
                        },
                        "required": ["account", "recipient", "text"],
                    },
                )
            ],
        )

        agent = SimpleNamespace(
            tools=[],
            mcp_servers=[],
            skills=[],
            system="you are an agent",
        )
        session = SimpleNamespace(id="sess_x", focal_channel=None)

        with (
            patch(
                "aios.harness.skills.augment_system_prompt",
                side_effect=lambda system, _versions: system,
            ),
            patch(
                "aios.services.connections.list_tools_for_session",
                new=AsyncMock(return_value=[]),
            ),
        ):
            prelude = await compute_step_prelude(
                pool=AsyncMock(),
                session_id="sess_x",
                session=session,
                agent=agent,
                channels=[],
                memory_store_echoes=[],
            )

        names = [t["function"]["name"] for t in prelude.tools]
        assert "mcp__signal__signal_send" in names, (
            f"connector tool absent from prelude — model will never see schema. tools={names}"
        )
