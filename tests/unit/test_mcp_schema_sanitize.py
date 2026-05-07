from __future__ import annotations

from typing import Any

from litellm import token_counter

from aios.mcp.schema import sanitize_mcp_schema

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
        assert "items" not in tools_field
        assert tools_field["title"] == "tools"

    def test_strips_sibling_type_next_to_oneof(self) -> None:
        node = {
            "oneOf": [{"type": "string"}, {"type": "null"}],
            "type": "string",
            "items": {"type": "string"},
        }
        cleaned = sanitize_mcp_schema(node)
        assert "oneOf" in cleaned
        assert "type" not in cleaned
        assert "items" not in cleaned

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
        before = {**original, "anyOf": [dict(b) for b in original["anyOf"]]}
        sanitize_mcp_schema(original)
        assert original == before


class TestLitellmTokenCounterIntegration:
    def test_sanitized_malformed_schema_does_not_crash_token_counter(self) -> None:
        sanitized = sanitize_mcp_schema(_MALFORMED_OPTIONAL_LIST_SCHEMA)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "mcp__aios__update_agent",
                    "description": "Update an agent",
                    "parameters": sanitized,
                },
            }
        ]
        count = token_counter(messages=[{"role": "user", "content": "hi"}], tools=tools)
        assert count > 0
