"""Ingress-only curated toolset expansion."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aios.models.agents import AgentCreate, AgentUpdate, ToolSpec


def _create(tools: list[dict[str, object]]) -> AgentCreate:
    return AgentCreate.model_validate({"name": "agent", "model": "gpt-4", "tools": tools})


def test_goal_management_expands_to_persistable_tools() -> None:
    agent = _create([{"type": "toolset", "name": "goal_management", "version": 1}])

    assert agent.tools == [
        ToolSpec(type="create_goal"),
        ToolSpec(type="list_obligations"),
        ToolSpec(type="defer_obligations"),
    ]
    assert all(isinstance(tool, ToolSpec) for tool in agent.tools)
    assert all(item["type"] != "toolset" for item in agent.model_dump()["tools"])


@pytest.mark.parametrize("explicit_first", [False, True])
def test_explicit_member_wins_and_keeps_expansion_order(explicit_first: bool) -> None:
    toolset: dict[str, object] = {
        "type": "toolset",
        "name": "goal_management",
        "version": 1,
    }
    override: dict[str, object] = {
        "type": "create_goal",
        "permission": "always_ask",
        "transport": "agent_tool",
    }
    agent = _create([override, toolset] if explicit_first else [toolset, override])

    assert [tool.type for tool in agent.tools] == [
        "create_goal",
        "list_obligations",
        "defer_obligations",
    ]
    assert agent.tools[0].permission == "always_ask"
    assert agent.tools[0].transport == "agent_tool"


def test_multiple_bundles_deduplicate_shared_members() -> None:
    agent = _create(
        [
            {"type": "toolset", "name": "delegation", "version": 1},
            {"type": "toolset", "name": "workflow_management", "version": 1},
        ]
    )

    assert [tool.type for tool in agent.tools].count("call_workflow") == 1


@pytest.mark.parametrize(
    "toolset",
    [
        {"type": "toolset", "name": "unknown", "version": 1},
        {"type": "toolset", "name": "goal_management", "version": 2},
    ],
)
def test_unknown_or_retired_bundle_is_rejected(toolset: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="unknown toolset"):
        _create([toolset])


def test_update_expands_before_storage() -> None:
    update = AgentUpdate.model_validate(
        {
            "version": 4,
            "tools": [{"type": "toolset", "name": "trigger_management", "version": 1}],
        }
    )

    assert update.tools is not None
    assert [tool.type for tool in update.tools] == [
        "trigger_create",
        "trigger_remove",
        "trigger_update",
        "trigger_list",
        "list_account_triggers",
    ]


def test_incomplete_protocol_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    _create([{"type": "create_goal"}])

    assert "tool_protocol_incomplete" in caplog.text
    assert "list_obligations" in caplog.text


def test_openapi_input_schema_exposes_toolset_arm() -> None:
    schema = AgentCreate.model_json_schema()

    assert "ToolsetSpec" in schema["$defs"]
