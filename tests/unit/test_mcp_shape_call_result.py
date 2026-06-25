"""Tests for :func:`aios.mcp.client.shape_call_result` and structured output.

Regression coverage for #1493: a spec-compliant MCP server (2025-06-18) may
declare an ``outputSchema`` and return its payload **only** in
``structuredContent`` with an empty ``content`` list (the backward-compat
serialized-JSON text block is a SHOULD, not a MUST). Before the fix,
``shape_call_result`` ignored ``structuredContent`` entirely and yielded an
empty-success envelope, silently dropping the whole payload.
"""

from __future__ import annotations

import json

from mcp.types import CallToolResult, TextContent, Tool

from aios.mcp.client import shape_call_result
from aios.mcp.schema import make_function_tool


class TestShapeCallResult:
    def test_text_content_passthrough(self) -> None:
        r = CallToolResult(
            content=[TextContent(type="text", text="hello")],
            isError=False,
        )
        assert shape_call_result(r) == {"content": "hello"}

    def test_structured_only_payload_is_serialized(self) -> None:
        # Repro from the issue: empty content, payload only in structuredContent.
        r = CallToolResult(
            content=[],
            structuredContent={"temperature": 21.5, "unit": "C"},
            isError=False,
        )
        out = shape_call_result(r)
        assert "error" not in out
        assert out["content"] != ""
        assert json.loads(out["content"]) == {"temperature": 21.5, "unit": "C"}

    def test_text_content_wins_over_structured(self) -> None:
        # When the backward-compat text block is present, it is authoritative;
        # we don't double-up by also appending the structured serialization.
        r = CallToolResult(
            content=[TextContent(type="text", text="hello")],
            structuredContent={"x": 1},
            isError=False,
        )
        assert shape_call_result(r) == {"content": "hello"}

    def test_structured_error_is_serialized(self) -> None:
        r = CallToolResult(
            content=[],
            structuredContent={"reason": "bad"},
            isError=True,
        )
        out = shape_call_result(r)
        assert out["code"] == "tool_error"
        assert json.loads(out["error"]) == {"reason": "bad"}

    def test_empty_result_stays_empty(self) -> None:
        r = CallToolResult(content=[], isError=False)
        assert shape_call_result(r) == {"content": ""}

    def test_falsy_structured_payload_is_serialized(self) -> None:
        # A structured payload that is falsy-but-present (e.g. an empty list)
        # is still real data and must not be dropped.
        r = CallToolResult(
            content=[],
            structuredContent={"items": []},
            isError=False,
        )
        out = shape_call_result(r)
        assert json.loads(out["content"]) == {"items": []}


class TestOutputSchemaPropagation:
    def test_output_schema_propagated_to_function_tool(self) -> None:
        tool = Tool(
            name="get_weather",
            description="weather",
            inputSchema={"type": "object", "properties": {}},
            outputSchema={
                "type": "object",
                "properties": {"temperature": {"type": "number"}},
            },
        )
        env = make_function_tool("mcp__srv__get_weather", tool)
        # The model must be told the tool produces structured output.
        assert "outputSchema" in env["function"]
        assert env["function"]["outputSchema"]["properties"]["temperature"] == {"type": "number"}

    def test_no_output_schema_omits_key(self) -> None:
        tool = Tool(
            name="noop",
            description="",
            inputSchema={"type": "object", "properties": {}},
        )
        env = make_function_tool("mcp__srv__noop", tool)
        assert "outputSchema" not in env["function"]
