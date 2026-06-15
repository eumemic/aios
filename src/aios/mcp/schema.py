"""MCP tool schemas — sanitize inputSchemas, build OpenAI function-tool envelopes."""

from __future__ import annotations

from typing import Any

from mcp.types import Tool


def sanitize_mcp_schema(node: Any) -> Any:
    """Drop the ``type`` keyword next to ``anyOf``/``oneOf`` (the union carries the real shape).

    Only the JSON Schema ``type`` **keyword** — whose value is a type name
    (``"string"``) or a list of them (``["string", "null"]``) — is redundant beside
    a union. A property literally **named** ``type`` (its value is a sub-schema
    ``dict``) inside a ``properties`` map is a parameter, not a keyword, and must be
    preserved even when a sibling property is named ``anyOf``/``oneOf``; aios
    sanitizes untrusted third-party tool schemas, so dropping it would silently
    corrupt a valid tool and make the model call it wrong.
    """
    if isinstance(node, dict):
        has_union = "anyOf" in node or "oneOf" in node
        drop_type = has_union and isinstance(node.get("type"), (str, list))
        return {
            key: sanitize_mcp_schema(value)
            for key, value in node.items()
            if not (drop_type and key == "type")
        }
    if isinstance(node, list):
        return [sanitize_mcp_schema(item) for item in node]
    return node


def make_function_tool(qualified_name: str, tool: Tool) -> dict[str, Any]:
    """Build the envelope, applying :func:`sanitize_mcp_schema` to ``tool.inputSchema``."""
    return {
        "type": "function",
        "function": {
            "name": qualified_name,
            "description": tool.description or "",
            "parameters": sanitize_mcp_schema(tool.inputSchema),
        },
    }
