"""Raw MCP Streamable HTTP client for aios inbound notifications."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, suppress
from typing import Any

import httpx
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.message import SessionMessage
from mcp.types import (
    LATEST_PROTOCOL_VERSION,
    ClientCapabilities,
    ErrorData,
    Implementation,
    InitializeResult,
    JSONRPCError,
    JSONRPCMessage,
    JSONRPCNotification,
    JSONRPCRequest,
    JSONRPCResponse,
)

INBOUND_CAPABILITY = "aiosInbound"
INBOUND_SUBSCRIBE_TOOL = "aios_inbound_subscribe"
NOTIFICATION_MESSAGE = "notifications/aios/inbound/message"
NOTIFICATION_CHANNELS_SNAPSHOT = "notifications/aios/inbound/channels_snapshot"
NOTIFICATION_CHANNELS_DELTA = "notifications/aios/inbound/channels_delta"
NOTIFICATION_REPLAY_LOST = "notifications/aios/inbound/replay_lost"

_MCP_INBOUND_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0)


class JsonRpcError(RuntimeError):
    def __init__(self, error: ErrorData) -> None:
        super().__init__(error.message)
        self.error = error


class RawInboundMcpClient:
    """Small raw JSON-RPC client that accepts aios custom notifications."""

    def __init__(
        self,
        url: str,
        headers: dict[str, str],
        *,
        client_name: str = "aios-inbound",
        client_version: str = "0",
    ) -> None:
        self.url = url
        self.headers = headers
        self.client_name = client_name
        self.client_version = client_version
        self._stack: AsyncExitStack | None = None
        self._write_stream: Any | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._next_id = 1
        self._pending: dict[str | int, asyncio.Future[dict[str, Any]]] = {}
        self._notifications: asyncio.Queue[JSONRPCNotification] = asyncio.Queue()

    async def __aenter__(self) -> RawInboundMcpClient:
        stack = AsyncExitStack()
        await stack.__aenter__()
        try:
            http_client = await stack.enter_async_context(
                httpx.AsyncClient(headers=self.headers, timeout=_MCP_INBOUND_TIMEOUT)
            )
            read_stream, write_stream, _ = await stack.enter_async_context(
                streamable_http_client(self.url, http_client=http_client)
            )
            self._write_stream = write_stream
            self._reader_task = asyncio.create_task(
                self._read_loop(read_stream),
                name="mcp-inbound-read-loop",
            )
            self._stack = stack
        except BaseException:
            await stack.aclose()
            raise
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._reader_task is not None:
            self._reader_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._reader_task
            self._reader_task = None
        for fut in self._pending.values():
            if not fut.done():
                fut.cancel()
        self._pending.clear()
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
        self._write_stream = None

    async def initialize(self) -> InitializeResult:
        result = await self.request(
            "initialize",
            {
                "protocolVersion": LATEST_PROTOCOL_VERSION,
                "capabilities": ClientCapabilities().model_dump(
                    by_alias=True, mode="json", exclude_none=True
                ),
                "clientInfo": Implementation(
                    name=self.client_name,
                    version=self.client_version,
                ).model_dump(by_alias=True, mode="json", exclude_none=True),
            },
        )
        await self.notify("notifications/initialized")
        return InitializeResult.model_validate(result)

    async def list_tools(self) -> list[dict[str, Any]]:
        result = await self.request("tools/list", None)
        tools = result.get("tools")
        return tools if isinstance(tools, list) else []

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        return await self.request(
            "tools/call",
            {
                "name": name,
                "arguments": arguments or {},
            },
        )

    async def request(
        self,
        method: str,
        params: dict[str, Any] | None,
        *,
        request_timeout: float = 30.0,
    ) -> dict[str, Any]:
        if self._write_stream is None:
            raise RuntimeError("MCP client is not connected")
        request_id = self._next_id
        self._next_id += 1
        fut: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = fut
        message = JSONRPCRequest(jsonrpc="2.0", id=request_id, method=method, params=params)
        await self._send(JSONRPCMessage(message))
        try:
            return await asyncio.wait_for(fut, timeout=request_timeout)
        finally:
            self._pending.pop(request_id, None)

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        await self._send(
            JSONRPCMessage(JSONRPCNotification(jsonrpc="2.0", method=method, params=params))
        )

    async def notifications(self) -> AsyncIterator[JSONRPCNotification]:
        while True:
            yield await self._notifications.get()

    async def _send(self, message: JSONRPCMessage) -> None:
        if self._write_stream is None:
            raise RuntimeError("MCP client is not connected")
        await self._write_stream.send(SessionMessage(message=message))

    async def _read_loop(self, read_stream: Any) -> None:
        try:
            async for incoming in read_stream:
                if isinstance(incoming, Exception):
                    self._fail_pending(incoming)
                    continue
                root = incoming.message.root
                if isinstance(root, JSONRPCResponse):
                    fut = self._pending.get(root.id)
                    if fut is not None and not fut.done():
                        fut.set_result(root.result)
                elif isinstance(root, JSONRPCError):
                    fut = self._pending.get(root.id)
                    if fut is not None and not fut.done():
                        fut.set_exception(JsonRpcError(root.error))
                elif isinstance(root, JSONRPCNotification):
                    await self._notifications.put(root)
                elif isinstance(root, JSONRPCRequest):
                    await self._respond_method_not_found(root)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fail_pending(exc)

    def _fail_pending(self, exc: BaseException) -> None:
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(exc)

    async def _respond_method_not_found(self, request: JSONRPCRequest) -> None:
        error = JSONRPCError(
            jsonrpc="2.0",
            id=request.id,
            error=ErrorData(code=-32601, message="Method not found"),
        )
        await self._send(JSONRPCMessage(error))


def has_aios_inbound_capability(init_result: InitializeResult) -> bool:
    experimental = init_result.capabilities.experimental
    return isinstance(experimental, dict) and INBOUND_CAPABILITY in experimental


def find_hidden_subscribe_tool(tools: list[dict[str, Any]]) -> dict[str, Any] | None:
    for tool in tools:
        if tool.get("name") != INBOUND_SUBSCRIBE_TOOL:
            continue
        meta = tool.get("_meta") or tool.get("meta")
        if isinstance(meta, dict) and meta.get("aios.internal") is True:
            return tool
    return None
