"""Provider-independent MCP argument validation at the dispatch boundary."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from aios.mcp.client import call_mcp_tool

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "draft": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "pieces": {
                    "type": "array",
                    "items": {"type": "object"},
                    "minItems": 1,
                },
            },
            "required": ["name", "pieces"],
        }
    },
    "required": ["draft"],
}


def _successful_result() -> MagicMock:
    content = MagicMock(text="ok", type="text")
    return MagicMock(content=[content], isError=False)


async def test_missing_nested_required_argument_never_reaches_mcp_session() -> None:
    session = AsyncMock()
    session.call_tool = AsyncMock(return_value=_successful_result())

    with patch("aios.mcp.client._open_session", new_callable=AsyncMock) as open_session:
        result = await call_mcp_tool(
            "https://mcp.example/",
            None,
            {},
            "propose",
            {"draft": {"name": "Tuesday"}},
            input_schema=_INPUT_SCHEMA,
        )

    open_session.assert_not_awaited()
    session.call_tool.assert_not_awaited()
    assert result["code"] == "tool_error"
    assert "draft.pieces" in result["error"]


async def test_valid_arguments_reach_mcp_session_unchanged() -> None:
    arguments = {"draft": {"name": "Tuesday", "pieces": [{"exercise": "squat"}]}}
    session = AsyncMock()
    session.call_tool = AsyncMock(return_value=_successful_result())

    with patch(
        "aios.mcp.client._open_session",
        new_callable=AsyncMock,
        return_value=(session, MagicMock()),
    ):
        result = await call_mcp_tool(
            "https://mcp.example/",
            None,
            {},
            "propose",
            arguments,
            input_schema=_INPUT_SCHEMA,
        )

    session.call_tool.assert_awaited_once_with("propose", arguments, meta=None)
    assert result == {"content": "ok"}


async def test_invalid_third_party_schema_keeps_advertised_only_dispatch() -> None:
    session = AsyncMock()
    session.call_tool = AsyncMock(return_value=_successful_result())

    with patch(
        "aios.mcp.client._open_session",
        new_callable=AsyncMock,
        return_value=(session, MagicMock()),
    ):
        result = await call_mcp_tool(
            "https://mcp.example/",
            None,
            {},
            "legacy_tool",
            {},
            input_schema={"type": "not-a-json-schema-type"},
        )

    session.call_tool.assert_awaited_once_with("legacy_tool", {}, meta=None)
    assert result == {"content": "ok"}
