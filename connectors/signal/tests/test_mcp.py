"""Tests for mcp.py — Signal tools + bearer-auth middleware."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from mcp.server.fastmcp import Context
from mcp.shared.context import RequestContext
from mcp.types import RequestParams
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route

from aios_signal.mcp import (
    BearerAuthMiddleware,
    _extract_timestamp,
    build_mcp_app,
    build_mcp_server,
    build_react_params,
    build_send_params,
    focal_chat_id_from_meta,
    parse_bind,
)


def _ctx_with_focal(chat_id: str | None) -> Context[Any, Any, Any]:
    """Build a Context carrying an ``aios.focal_channel_path`` in ``_meta``.

    The ToolManager.call_tool path accepts an explicit Context; this
    lets us simulate what aios sends without going through a full
    ClientSession round-trip.
    """
    rc = MagicMock(spec=RequestContext)
    if chat_id is None:
        rc.meta = None
    else:
        rc.meta = RequestParams.Meta.model_validate({"aios.focal_channel_path": chat_id})
    return Context(request_context=rc)


class FakeRpc:
    """Records every ``call`` invocation for assertion."""

    def __init__(self, results: dict[str, Any] | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any] | None]] = []
        self._results = results or {}

    async def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        self.calls.append((method, params))
        return self._results.get(method, {})


async def _call_tool(
    rpc: FakeRpc,
    name: str,
    args: dict[str, Any],
    *,
    focal_chat_id: str | None = None,
) -> dict[str, Any]:
    """Invoke a Signal MCP tool with an aios-style focal-channel meta.

    Goes through ``FastMCP.tool_manager.call_tool`` so the tool handler's
    ``ctx: Context`` dependency is filled in with the constructed meta.
    """
    mcp = build_mcp_server(rpc=rpc)  # type: ignore[arg-type]
    result = await mcp._tool_manager.call_tool(
        name,
        args,
        _ctx_with_focal(focal_chat_id),
        convert_result=True,
    )
    # call_tool returns (content_blocks, structured_result) when the tool
    # has an output_schema (ours do — typed dict returns).
    assert isinstance(result, tuple)
    structured: Any = result[1]
    assert isinstance(structured, dict)
    return structured


# ─── unit: pure helpers ─────────────────────────────────────────────────────


class TestFocalChatIdFromMeta:
    def test_returns_path_when_present(self) -> None:
        meta = RequestParams.Meta.model_validate({"aios.focal_channel_path": "chat-abc"})
        assert focal_chat_id_from_meta(meta) == "chat-abc"

    def test_none_meta_raises(self) -> None:
        with pytest.raises(ValueError, match="focal channel"):
            focal_chat_id_from_meta(None)

    def test_missing_key_raises(self) -> None:
        meta = RequestParams.Meta.model_validate({"other_key": "x"})
        with pytest.raises(ValueError, match="focal channel"):
            focal_chat_id_from_meta(meta)

    def test_empty_string_raises(self) -> None:
        meta = RequestParams.Meta.model_validate({"aios.focal_channel_path": ""})
        with pytest.raises(ValueError, match="focal channel"):
            focal_chat_id_from_meta(meta)


class TestBuildSendParams:
    def test_dm_uuid(self) -> None:
        params = build_send_params("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "hi")
        assert params == {
            "message": "hi",
            "recipient": ["aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"],
        }

    def test_group_urlsafe_base64_decoded(self) -> None:
        params = build_send_params("abc-def_xyz==", "hi")
        assert params["groupId"] == "abc+def/xyz=="
        assert "recipient" not in params

    def test_markdown_produces_text_styles(self) -> None:
        params = build_send_params("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "**bold** now")
        assert params["message"] == "bold now"
        assert any("BOLD" in s for s in params["textStyles"])


class TestBuildReactParams:
    def test_dm_shape(self) -> None:
        params = build_react_params(
            "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "cccccccc-dddd-eeee-ffff-000000000000",
            987654,
            "👍",
        )
        assert params == {
            "emoji": "👍",
            "targetAuthor": "cccccccc-dddd-eeee-ffff-000000000000",
            "targetTimestamp": 987654,
            "recipient": ["aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"],
        }

    def test_group_shape(self) -> None:
        params = build_react_params(
            "abc-def_xyz==",
            "cccccccc-dddd-eeee-ffff-000000000000",
            987654,
            "👍",
        )
        assert params["groupId"] == "abc+def/xyz=="
        assert "recipient" not in params


# ─── integration: tool handlers consume _meta ──────────────────────────────


async def test_signal_send_reads_focal_from_meta() -> None:
    rpc = FakeRpc(results={"send": {"timestamp": 123456}})
    result = await _call_tool(
        rpc,
        "signal_send",
        {"text": "hello"},
        focal_chat_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    )
    assert result == {"sent_at_ms": 123456}
    method, params = rpc.calls[0]
    assert method == "send"
    assert params == {
        "message": "hello",
        "recipient": ["aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"],
    }


async def test_signal_send_errors_without_meta() -> None:
    rpc = FakeRpc()
    mcp = build_mcp_server(rpc=rpc)  # type: ignore[arg-type]
    # No focal — aios should have filtered this tool out, but defend.
    ctx_no_meta = _ctx_with_focal(None)
    with pytest.raises(Exception, match="focal channel"):
        await mcp._tool_manager.call_tool(
            "signal_send",
            {"text": "hi"},
            ctx_no_meta,
            convert_result=True,
        )


async def test_signal_react_reads_focal_from_meta() -> None:
    rpc = FakeRpc()
    await _call_tool(
        rpc,
        "signal_react",
        {
            "target_author_uuid": "cccccccc-dddd-eeee-ffff-000000000000",
            "target_timestamp_ms": 987654,
            "emoji": "👍",
        },
        focal_chat_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    )
    method, params = rpc.calls[0]
    assert method == "sendReaction"
    assert params == {
        "emoji": "👍",
        "targetAuthor": "cccccccc-dddd-eeee-ffff-000000000000",
        "targetTimestamp": 987654,
        "recipient": ["aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"],
    }


async def test_signal_send_group_via_meta() -> None:
    rpc = FakeRpc(results={"send": {"timestamp": 99}})
    await _call_tool(rpc, "signal_send", {"text": "hi"}, focal_chat_id="abc-def_xyz==")
    _method, params = rpc.calls[0]
    assert params is not None
    assert params["groupId"] == "abc+def/xyz=="
    assert "recipient" not in params


async def test_signal_send_markdown_via_meta() -> None:
    rpc = FakeRpc(results={"send": {"timestamp": 1}})
    await _call_tool(
        rpc,
        "signal_send",
        {"text": "**bold** now"},
        focal_chat_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    )
    _method, params = rpc.calls[0]
    assert params is not None
    assert params["message"] == "bold now"
    assert any("BOLD" in s for s in params["textStyles"])


async def test_signal_read_receipt() -> None:
    rpc = FakeRpc()
    await _call_tool(
        rpc,
        "signal_read_receipt",
        {
            "sender_uuid": "eeeeeeee-ffff-0000-1111-222222222222",
            "timestamp_ms_list": [100, 200, 300],
        },
    )
    method, params = rpc.calls[0]
    assert method == "sendReceipt"
    assert params == {
        "recipient": ["eeeeeeee-ffff-0000-1111-222222222222"],
        "type": "read",
        "targetTimestamp": [100, 200, 300],
    }


def test_extract_timestamp_happy_path() -> None:
    assert _extract_timestamp({"timestamp": 42}) == 42


def test_extract_timestamp_rejects_junk() -> None:
    with pytest.raises(ValueError):
        _extract_timestamp({"no_timestamp": True})
    with pytest.raises(ValueError):
        _extract_timestamp({"timestamp": "42"})  # strings rejected — int-only
    with pytest.raises(ValueError):
        _extract_timestamp("not-a-dict")


def test_parse_bind() -> None:
    assert parse_bind("127.0.0.1:9100") == ("127.0.0.1", 9100)
    assert parse_bind("0.0.0.0:80") == ("0.0.0.0", 80)


def test_parse_bind_ipv6() -> None:
    assert parse_bind("[::1]:9100") == ("::1", 9100)


def test_parse_bind_rejects_malformed() -> None:
    with pytest.raises(ValueError):
        parse_bind("9100")
    with pytest.raises(ValueError):
        parse_bind(":9100")


# ─── Bearer-auth middleware tests ────────────────────────────────────────────


def _build_dummy_app(token: str) -> Starlette:
    async def ok(_r: Any) -> PlainTextResponse:
        return PlainTextResponse("ok")

    app = Starlette(routes=[Route("/mcp", ok)])
    app.add_middleware(BearerAuthMiddleware, expected_token=token)
    return app


async def test_bearer_auth_rejects_missing_header() -> None:
    app = _build_dummy_app("secret")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/mcp")
    assert r.status_code == 401


async def test_bearer_auth_rejects_wrong_token() -> None:
    app = _build_dummy_app("secret")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/mcp", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401


async def test_bearer_auth_accepts_correct_token() -> None:
    app = _build_dummy_app("secret")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/mcp", headers={"Authorization": "Bearer secret"})
    assert r.status_code == 200


async def test_bearer_auth_rejects_non_bearer_scheme() -> None:
    app = _build_dummy_app("secret")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/mcp", headers={"Authorization": "Basic secret"})
    assert r.status_code == 401


def test_build_mcp_app_returns_starlette() -> None:
    rpc = FakeRpc()
    mcp = build_mcp_server(rpc=rpc)  # type: ignore[arg-type]
    app = build_mcp_app(mcp, token="t")
    assert isinstance(app, Starlette)


def test_build_mcp_server_passes_signal_instructions() -> None:
    """The MCP server's ``instructions`` field is the transport for
    Signal's per-connector affordance prose; aios reads it from the
    ``InitializeResult`` returned by ``session.initialize()`` and
    composes it into the agent's system prompt.
    """
    from aios_signal.prompts import SIGNAL_SERVER_INSTRUCTIONS

    rpc = FakeRpc()
    mcp = build_mcp_server(rpc=rpc)  # type: ignore[arg-type]
    assert mcp.instructions == SIGNAL_SERVER_INSTRUCTIONS
    # Sanity-check the prose covers the v1 toolset.
    assert "signal_send" in mcp.instructions
    assert "signal_react" in mcp.instructions
    assert "signal_read_receipt" in mcp.instructions
