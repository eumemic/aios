"""Unit tests for the typed authoring facade (``aios_sdk.authoring``).

Fully deterministic and offline: the facade never touches the network, the
event log, or the server. Each test asserts the facade is a faithful,
zero-validation sibling of the generated models — it builds exactly the legal
dict for each kind and round-trips it through ``from_dict`` / ``to_dict``.
"""

from __future__ import annotations

import pytest
from aios_sdk import (
    AgentBuilder,
    define_agent,
    define_tool_builtin,
    define_tool_custom,
    define_tool_mcp,
)
from aios_sdk._generated.models.agent_create import AgentCreate
from aios_sdk._generated.models.tool_spec import ToolSpec
from aios_sdk._generated.models.tool_spec_type_type_0 import ToolSpecTypeType0
from aios_sdk._generated.models.tool_spec_type_type_1 import ToolSpecTypeType1


def test_define_agent_builds_expected_dict() -> None:
    built = (
        define_agent("x", "anthropic/claude-opus-4-6")
        .system("hi")
        .tool(define_tool_builtin("bash"))
        .build()
    )
    assert isinstance(built, AgentCreate)
    assert built.to_dict() == {
        "name": "x",
        "model": "anthropic/claude-opus-4-6",
        "system": "hi",
        "tools": [{"type": "bash", "enabled": True}],
    }


def test_define_tool_custom_populates_required_fields() -> None:
    tool = define_tool_custom(name="t", description="d", input_schema={"type": "object"})
    assert isinstance(tool, ToolSpec)
    assert tool.type_ is ToolSpecTypeType1.CUSTOM
    d = tool.to_dict()
    assert d["type"] == "custom"
    assert d["name"] == "t"
    assert d["description"] == "d"
    assert d["input_schema"] == {"type": "object"}
    # No mcp_toolset field leaks into a custom tool.
    assert "mcp_server_name" not in d


def test_build_roundtrip_lossless() -> None:
    builder = (
        define_agent("agent", "anthropic/claude-opus-4-6")
        .system("be helpful")
        .description("a test agent")
        .window(10_000, 20_000)
        .litellm_extra({"temperature": 0.0})
        .metadata({"team": "platform"})
        .tools(
            define_tool_builtin("bash", permission="always_ask", transport="cli"),
            define_tool_custom(
                name="my_tool",
                description="does a thing",
                input_schema={"type": "object", "properties": {}},
                transport="agent_tool",
            ),
            define_tool_mcp(mcp_server_name="srv"),
        )
    )
    rebuilt = AgentCreate.from_dict(builder.build().to_dict())
    assert rebuilt.to_dict() == builder.build().to_dict()


def test_builtin_constructor_has_no_custom_fields() -> None:
    # Static-typing assertion (documented): mypy --strict rejects passing
    # ``input_schema`` / ``mcp_server_name`` to ``define_tool_builtin`` because
    # the signature has no such parameters. Such a call is unspellable without a
    # ``# type: ignore``; we therefore assert the property the *runtime* can see:
    # the produced dict carries none of the custom/mcp-only keys.
    d = define_tool_builtin("read").to_dict()
    assert d["type"] == "read"
    for forbidden in ("name", "description", "input_schema", "mcp_server_name"):
        assert forbidden not in d

    # The kind is selected by which constructor is called, not by a flag: the
    # built-in carries a ToolSpecTypeType0 member, never a ToolSpecTypeType1.
    assert isinstance(define_tool_builtin("read").type_, ToolSpecTypeType0)


def test_define_tool_mcp_only_carries_mcp_fields() -> None:
    tool = define_tool_mcp(mcp_server_name="srv")
    assert tool.type_ is ToolSpecTypeType1.MCP_TOOLSET
    d = tool.to_dict()
    assert d["type"] == "mcp_toolset"
    assert d["mcp_server_name"] == "srv"
    for forbidden in ("name", "description", "input_schema"):
        assert forbidden not in d


def test_mcp_prefix_name_still_rejected_by_server_model() -> None:
    # The facade cannot structurally prevent a reserved ``mcp__`` name on a
    # custom tool (the name is a plain str), so this combo is *representable*.
    # It must still reject when round-tripped through the SOURCE model, proving
    # the facade leaves the server as the single validation authority.
    pytest.importorskip("aios.models.agents")
    from aios.models.agents import ToolSpec as SourceToolSpec

    facade_dict = define_tool_custom(
        name="mcp__evil", description="d", input_schema={"type": "object"}
    ).to_dict()
    with pytest.raises(ValueError, match="mcp__"):
        SourceToolSpec.model_validate(facade_dict)


def test_define_agent_returns_builder() -> None:
    assert isinstance(define_agent("x", "m"), AgentBuilder)
