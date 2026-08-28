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

    # Arrays must always carry a dict ``items`` and properties must be dicts:
    # litellm's ``_format_type`` does ``props['items']`` unconditionally on
    # ``type == "array"`` (the #2294 KeyError incident class) and its
    # ``_format_object_parameters`` calls ``.get`` on every property value;
    # OpenAI additionally rejects array schemas without ``items``.

    def test_adds_items_to_bare_array(self) -> None:
        assert sanitize_mcp_schema({"type": "array"}) == {"type": "array", "items": {}}

    def test_adds_items_to_nested_bare_array(self) -> None:
        node = {"type": "object", "properties": {"xs": {"type": "array"}}}
        cleaned = sanitize_mcp_schema(node)
        assert cleaned["properties"]["xs"] == {"type": "array", "items": {}}

    def test_replaces_tuple_form_items(self) -> None:
        # Draft-4 tuple validation: ``items: [...]`` — litellm recurses with
        # ``.get`` and crashes on the list.
        node = {"type": "array", "items": [{"type": "string"}, {"type": "integer"}]}
        assert sanitize_mcp_schema(node)["items"] == {}

    def test_replaces_boolean_items(self) -> None:
        node = {"type": "array", "items": True}
        assert sanitize_mcp_schema(node)["items"] == {}

    def test_keeps_dict_items_intact(self) -> None:
        node = {"type": "array", "items": {"type": "string"}}
        assert sanitize_mcp_schema(node) == node

    def test_coerces_non_dict_property_value(self) -> None:
        # ``"foo": true`` is a valid boolean JSON Schema ("anything"), but
        # litellm's ``_format_object_parameters`` calls ``.get`` on it.
        node = {"type": "object", "properties": {"foo": True, "bar": {"type": "string"}}}
        cleaned = sanitize_mcp_schema(node)
        assert cleaned["properties"]["foo"] == {}
        assert cleaned["properties"]["bar"] == {"type": "string"}

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
        assert envelope["function"]["strict"] is False
        assert "type" not in envelope["function"]["parameters"]["properties"]["tools"]

        count = token_counter(messages=[{"role": "user", "content": "hi"}], tools=[envelope])
        assert count > 0

    def test_bare_array_schema_survives_token_counter(self) -> None:
        # Red on the pre-fix sanitizer: litellm's ``_format_type`` raises
        # ``KeyError: 'items'`` on a bare array (the #2294 incident class).
        schema = {"type": "object", "properties": {"xs": {"type": "array"}}}
        tool = Tool(name="t", description="d", inputSchema=schema)
        envelope = make_function_tool("mcp__s__t", tool)

        assert envelope["function"]["parameters"]["properties"]["xs"]["items"] == {}
        count = token_counter(messages=[{"role": "user", "content": "hi"}], tools=[envelope])
        assert count > 0

    def test_tuple_and_bool_items_survive_token_counter(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "pairs": {"type": "array", "items": [{"type": "string"}]},
                "anything": {"type": "array", "items": True},
            },
        }
        tool = Tool(name="t", description="d", inputSchema=schema)
        envelope = make_function_tool("mcp__s__t", tool)

        count = token_counter(messages=[{"role": "user", "content": "hi"}], tools=[envelope])
        assert count > 0

    def test_non_dict_property_value_survives_token_counter(self) -> None:
        schema = {"type": "object", "properties": {"foo": True}}
        tool = Tool(name="t", description="d", inputSchema=schema)
        envelope = make_function_tool("mcp__s__t", tool)

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

        assert function["strict"] is False
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

    def test_make_function_tool_never_promotes_optional_schema_to_strict(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "required_name": {"type": "string"},
                "optional_note": {"type": "string"},
            },
            "required": ["required_name"],
        }
        tool = Tool(name="annotate", description="Annotate", inputSchema=schema)

        function = make_function_tool("mcp__notes__annotate", tool)["function"]

        assert function["strict"] is False
        assert function["parameters"] == schema
