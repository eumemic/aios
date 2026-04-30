"""Unit tests for MCP inbound protocol helpers."""

from __future__ import annotations

import asyncio

import anyio
import pytest
from mcp.shared.message import SessionMessage
from mcp.types import (
    ErrorData,
    Implementation,
    InitializeResult,
    JSONRPCError,
    JSONRPCMessage,
    JSONRPCNotification,
    JSONRPCResponse,
    ServerCapabilities,
)

from aios.inbound.protocol import (
    NOTIFICATION_MESSAGE,
    JsonRpcError,
    RawInboundMcpClient,
    find_hidden_subscribe_tool,
    has_aios_inbound_capability,
)


def test_detects_aios_inbound_capability() -> None:
    result = InitializeResult(
        protocolVersion="2025-06-18",
        serverInfo=Implementation(name="signal", version="1"),
        capabilities=ServerCapabilities(experimental={"aiosInbound": {"version": 1}}),
    )
    assert has_aios_inbound_capability(result) is True


def test_missing_aios_inbound_capability_is_false() -> None:
    result = InitializeResult(
        protocolVersion="2025-06-18",
        serverInfo=Implementation(name="other", version="1"),
        capabilities=ServerCapabilities(),
    )
    assert has_aios_inbound_capability(result) is False


def test_find_hidden_subscribe_tool_requires_internal_marker() -> None:
    tools = [
        {"name": "aios_inbound_subscribe", "_meta": {}},
        {"name": "aios_inbound_subscribe", "_meta": {"aios.internal": True}},
    ]
    assert find_hidden_subscribe_tool(tools) == tools[1]


async def test_raw_client_routes_response_and_custom_notification() -> None:
    read_send, read_recv = anyio.create_memory_object_stream[SessionMessage | Exception](10)
    write_send, write_recv = anyio.create_memory_object_stream[SessionMessage](10)
    client = RawInboundMcpClient("http://mcp.test", {})
    client._write_stream = write_send
    reader = asyncio.create_task(client._read_loop(read_recv))
    try:
        request_task = asyncio.create_task(client.request("tools/list", None))
        outbound = await write_recv.receive()
        request_id = outbound.message.root.id

        await read_send.send(
            SessionMessage(
                message=JSONRPCMessage(
                    JSONRPCResponse(jsonrpc="2.0", id=request_id, result={"tools": []})
                )
            )
        )
        assert await request_task == {"tools": []}

        await read_send.send(
            SessionMessage(
                message=JSONRPCMessage(
                    JSONRPCNotification(
                        jsonrpc="2.0",
                        method=NOTIFICATION_MESSAGE,
                        params={"event_id": "e1"},
                    )
                )
            )
        )
        notification = await asyncio.wait_for(client._notifications.get(), timeout=1)
        assert notification.method == NOTIFICATION_MESSAGE
        assert notification.params == {"event_id": "e1"}
    finally:
        reader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reader


async def test_raw_client_routes_jsonrpc_errors() -> None:
    read_send, read_recv = anyio.create_memory_object_stream[SessionMessage | Exception](10)
    write_send, write_recv = anyio.create_memory_object_stream[SessionMessage](10)
    client = RawInboundMcpClient("http://mcp.test", {})
    client._write_stream = write_send
    reader = asyncio.create_task(client._read_loop(read_recv))
    try:
        request_task = asyncio.create_task(client.request("tools/call", {"name": "x"}))
        outbound = await write_recv.receive()
        request_id = outbound.message.root.id

        await read_send.send(
            SessionMessage(
                message=JSONRPCMessage(
                    JSONRPCError(
                        jsonrpc="2.0",
                        id=request_id,
                        error=ErrorData(code=-32000, message="boom"),
                    )
                )
            )
        )
        with pytest.raises(JsonRpcError, match="boom"):
            await request_task
    finally:
        reader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reader
