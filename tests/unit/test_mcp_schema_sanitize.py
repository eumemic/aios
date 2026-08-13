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

    def test_preserves_property_literally_named_type(self) -> None:
        # `type`/`anyOf`/`oneOf` are JSON Schema KEYWORDS, but inside a `properties`
        # map they are property NAMES. A tool param literally named `type` must not
        # be stripped just because a sibling param is named `anyOf`/`oneOf`: the
        # union-strip targets the `type` KEYWORD (a str/list value), not a named
        # sub-schema (a dict). aios sanitizes untrusted third-party MCP schemas, so a
        # tool with a param named `type` (common) alongside one named `anyOf` would
        # otherwise have its `type` param silently dropped — the model then sees a
        # parameter with no schema and calls the tool wrong.
        node = {
            "type": "object",
            "properties": {
                "type": {"type": "string", "description": "the kind"},
                "anyOf": {"type": "integer"},
            },
        }
        cleaned = sanitize_mcp_schema(node)
        assert "type" in cleaned["properties"]
        assert cleaned["properties"]["type"] == {"type": "string", "description": "the kind"}
        assert "anyOf" in cleaned["properties"]

    def test_preserves_param_named_type_when_keyword_type_is_a_list(self) -> None:
        # The `type` keyword may be a list (`["string","null"]`); that form is still a
        # keyword and IS stripped next to a union, but a param NAMED type (dict value)
        # at the same map is preserved.
        node = {
            "properties": {
                "type": {"type": ["string", "null"]},
                "oneOf": {"type": "boolean"},
            },
        }
        cleaned = sanitize_mcp_schema(node)
        assert "type" in cleaned["properties"]

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
        assert envelope["function"]["strict"] is True
        assert "type" not in envelope["function"]["parameters"]["properties"]["tools"]

        count = token_counter(messages=[{"role": "user", "content": "hi"}], tools=[envelope])
        assert count > 0

    def test_make_function_tool_handles_none_description(self) -> None:
        tool = Tool(name="t", description=None, inputSchema={"type": "object"})
        envelope = make_function_tool("mcp__aios__t", tool)
        assert envelope["function"]["description"] == ""

    def test_make_function_tool_preserves_nested_required_constraints(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "draft": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "pieces": {
                            "type": "array",
                            "items": {"type": "object"},
                            "minItems": 1,
                            "maxItems": 20,
                        },
                    },
                    "required": ["name", "pieces"],
                    "additionalProperties": False,
                }
            },
            "required": ["draft"],
        }
        tool = Tool(name="propose", description="Propose", inputSchema=schema)

        function = make_function_tool("mcp__planner__propose", tool)["function"]
        draft = function["parameters"]["properties"]["draft"]

        assert function["strict"] is True
        assert function["parameters"]["required"] == ["draft"]
        assert draft["required"] == ["name", "pieces"]
        assert draft["properties"]["pieces"]["type"] == "array"
        assert draft["properties"]["pieces"]["minItems"] == 1
        assert draft["properties"]["pieces"]["maxItems"] == 20
        assert draft["additionalProperties"] is False

    def test_make_function_tool_preserves_object_and_nullable_union_properties(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "summary": {
                    "anyOf": [{"type": "string", "maxLength": 200}, {"type": "null"}],
                    "type": "string",
                },
                "draft": {
                    "type": "object",
                    "properties": {"name": {"type": "string", "format": "hostname"}},
                    "required": ["name"],
                },
            },
        }
        tool = Tool(name="propose", description="Propose", inputSchema=schema)

        parameters = make_function_tool("mcp__planner__propose", tool)["function"]["parameters"]

        assert set(parameters["properties"]) == {"summary", "draft"}
        assert parameters["properties"]["summary"]["anyOf"] == [
            {"type": "string", "maxLength": 200},
            {"type": "null"},
        ]
        assert "type" not in parameters["properties"]["summary"]
        assert parameters["properties"]["draft"]["required"] == ["name"]
