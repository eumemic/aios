"""Guard agent ToolSpec literals against built-in registry drift."""

from __future__ import annotations

from typing import get_args

from aios.models.agents import AgentCreate, BuiltinToolType, ToolSpec


def test_agent_tool_type_literal_covers_agent_callable_registry_tools() -> None:
    """Every model-callable built-in must be grantable in an agent definition."""
    import aios.tools  # noqa: F401 - trigger built-in registration side effects
    from aios.tools.registry import registry

    grantable_tool_types = set(get_args(BuiltinToolType))
    harness_injected_tools = {"switch_channel", "return", "error", "cancel_goal"}
    agent_callable_names = {
        name for name in registry.names() if registry.get(name).transport in {"agent_tool", "both"}
    } - harness_injected_tools

    assert agent_callable_names <= grantable_tool_types


def test_agent_tool_type_literal_entries_resolve_to_registry_tools() -> None:
    """Every built-in grant type should have a backing runtime registration."""
    import aios.tools  # noqa: F401 - trigger built-in registration side effects
    from aios.tools.registry import registry

    harness_injected_tools = {"switch_channel", "return", "error", "cancel_goal"}
    registered_names = set(registry.names()) | harness_injected_tools

    assert set(get_args(BuiltinToolType)) <= registered_names


def test_agent_definition_can_grant_list_workflows() -> None:
    agent = AgentCreate.model_validate(
        {
            "name": "workflow-reader",
            "model": "openrouter/test",
            "tools": [{"type": "list_workflows"}],
        }
    )

    assert agent.tools == [ToolSpec(type="list_workflows")]
