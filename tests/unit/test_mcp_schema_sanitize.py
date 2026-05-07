from __future__ import annotations

import copy
from typing import Any

from litellm import token_counter
from mcp.types import Tool

from aios.mcp.schema import make_function_tool, sanitize_mcp_schema

_MALFORMED_OPTIONAL_LIST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tools": {
            "anyOf": [
                {"items": {"type": "object"}, "type": "array"},
                {"type": "null"},
            ],
            "title": "tools",
            "type": "array",
        },
    },
}


class TestSanitizer:
    def test_strips_sibling_type_next_to_anyof(self) -> None:
        cleaned = sanitize_mcp_schema(_MALFORMED_OPTIONAL_LIST_SCHEMA)
        tools_field = cleaned["properties"]["tools"]
        assert "anyOf" in tools_field
        assert "type" not in tools_field
        assert tools_field["title"] == "tools"

    def test_strips_sibling_type_next_to_oneof(self) -> None:
        node = {
            "oneOf": [{"type": "string"}, {"type": "null"}],
            "type": "string",
        }
        cleaned = sanitize_mcp_schema(node)
        assert "oneOf" in cleaned
        assert "type" not in cleaned

    def test_preserves_clean_array_schema(self) -> None:
        node = {"type": "array", "items": {"type": "string"}, "title": "names"}
        assert sanitize_mcp_schema(node) == node

    def test_preserves_clean_object_schema(self) -> None:
        node = {
            "type": "object",
            "properties": {
                "x": {"type": "integer"},
                "y": {"type": "string"},
            },
        }
        assert sanitize_mcp_schema(node) == node

    def test_preserves_sibling_items_when_anyof_present(self) -> None:
        # Stripping only `type` — sibling `items` carries no payload that crashes
        # litellm and might legitimately be meaningful in some union shapes; leave it.
        node = {
            "anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}],
            "items": {"type": "integer"},
        }
        cleaned = sanitize_mcp_schema(node)
        assert cleaned["items"] == {"type": "integer"}

    def test_recurses_into_anyof_branches(self) -> None:
        node = {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "inner": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "type": "string",
                        }
                    },
                },
                {"type": "null"},
            ],
        }
        cleaned = sanitize_mcp_schema(node)
        inner = cleaned["anyOf"][0]["properties"]["inner"]
        assert "type" not in inner
        assert "anyOf" in inner

    def test_returns_non_dict_unchanged(self) -> None:
        assert sanitize_mcp_schema("string") == "string"
        assert sanitize_mcp_schema(42) == 42
        assert sanitize_mcp_schema(None) is None
        assert sanitize_mcp_schema([1, 2, 3]) == [1, 2, 3]

    def test_does_not_mutate_input(self) -> None:
        original = {
            "anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}],
            "type": "array",
        }
        before = copy.deepcopy(original)
        sanitize_mcp_schema(original)
        assert original == before


class TestMakeFunctionToolIntegration:
    def test_make_function_tool_sanitizes_and_survives_token_counter(self) -> None:
        tool = Tool(
            name="update_agent",
            description="Update an agent",
            inputSchema=_MALFORMED_OPTIONAL_LIST_SCHEMA,
        )
        envelope = make_function_tool("mcp__aios__update_agent", tool)

        assert envelope["type"] == "function"
        assert envelope["function"]["name"] == "mcp__aios__update_agent"
        assert envelope["function"]["description"] == "Update an agent"
        assert "type" not in envelope["function"]["parameters"]["properties"]["tools"]

        count = token_counter(messages=[{"role": "user", "content": "hi"}], tools=[envelope])
        assert count > 0

    def test_make_function_tool_handles_none_description(self) -> None:
        tool = Tool(name="t", description=None, inputSchema={"type": "object"})
        envelope = make_function_tool("mcp__aios__t", tool)
        assert envelope["function"]["description"] == ""
