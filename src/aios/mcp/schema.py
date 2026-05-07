"""Build the OpenAI ``{"type":"function", ...}`` envelope for an MCP tool.

The single entry point :func:`make_function_tool` enforces the schema
sanitization step that downstream consumers (notably the token counter
in the harness) require.  Routing every MCP-discovered tool through
this helper means a future third call site can't forget the sanitizer.
"""

from __future__ import annotations

from typing import Any

from mcp.types import Tool

# Sibling JSON Schema keywords that must be dropped when an ``anyOf`` /
# ``oneOf`` envelope is present: the union already carries the real
# shape, and a stray sibling ``type: "array"`` without its companion
# ``items`` crashes JSON-Schema-walking consumers.
_HOISTED_SIBLINGS = frozenset({"type", "items"})


def sanitize_mcp_schema(node: Any) -> Any:
    """Recursively drop ``type``/``items`` siblings of ``anyOf``/``oneOf``."""
    if isinstance(node, dict):
        has_union = "anyOf" in node or "oneOf" in node
        return {
            key: sanitize_mcp_schema(value)
            for key, value in node.items()
            if not (has_union and key in _HOISTED_SIBLINGS)
        }
    if isinstance(node, list):
        return [sanitize_mcp_schema(item) for item in node]
    return node


def make_function_tool(qualified_name: str, tool: Tool) -> dict[str, Any]:
    """Build an OpenAI function-tool dict from an MCP :class:`Tool`."""
    return {
        "type": "function",
        "function": {
            "name": qualified_name,
            "description": tool.description or "",
            "parameters": sanitize_mcp_schema(tool.inputSchema),
        },
    }
