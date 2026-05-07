from __future__ import annotations

from typing import Any

from mcp.types import Tool


def sanitize_mcp_schema(node: Any) -> Any:
    """Drop ``type`` siblings of ``anyOf``/``oneOf`` (the union carries the real shape)."""
    if isinstance(node, dict):
        has_union = "anyOf" in node or "oneOf" in node
        return {
            key: sanitize_mcp_schema(value)
            for key, value in node.items()
            if not (has_union and key == "type")
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
