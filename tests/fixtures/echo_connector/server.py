"""Echo connector — minimal MCP server used by PR2 supervisor e2e tests.

Two tools (``ping``, ``echo``), one ``notifications/aios/accounts``
emission shortly after initialize, and the
``experimental.aios/connector`` capability the supervisor requires.

The accounts notification bypasses :meth:`BaseSession.send_notification`
because the SDK's ``ServerNotification`` discriminated union rejects
custom methods.  Pushing :class:`JSONRPCNotification` directly onto the
write stream is the same approach the reference SDK in PR3 will take,
just inlined here so the test fixture has zero external deps beyond
the SDK we already use.
"""

from __future__ import annotations

import anyio
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from mcp.shared.message import SessionMessage
from mcp.types import (
    JSONRPCMessage,
    JSONRPCNotification,
    TextContent,
    Tool,
)

from aios.mcp.stdio_transport import AIOS_NOTIFICATION_PREFIX

ACCOUNTS_PAYLOAD = {
    "accounts": [
        {"id": "echo-1", "display_name": "Echo Account One"},
    ]
}


def _build_server() -> Server[None, None]:
    server: Server[None, None] = Server(
        "echo",
        version="0.0.0-test",
        instructions="Test echo connector. Use `ping` to verify wiring.",
    )

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="ping",
                description="Returns 'pong' to confirm the connector is responsive.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            ),
            Tool(
                name="echo",
                description="Echoes the supplied text.",
                inputSchema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "additionalProperties": False,
                },
            ),
            Tool(
                name="crash",
                description="Exits the connector process — used to verify supervisor restart.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, object]) -> list[TextContent]:
        if name == "ping":
            return [TextContent(type="text", text="pong")]
        if name == "echo":
            text = str(arguments.get("text", ""))
            return [TextContent(type="text", text=text)]
        if name == "crash":
            import os

            os._exit(1)
        raise ValueError(f"unknown tool: {name}")

    return server


async def _emit_accounts_after_init(
    write_stream: anyio.abc.ObjectSendStream[SessionMessage],
) -> None:
    """Push one ``notifications/aios/accounts`` after a brief delay.

    Run as a sibling task so the supervisor sees the snapshot land
    soon after ``initialize()`` completes.  100 ms is more than enough
    for the local-pipe round trip; tests poll the snapshot with a
    longer ceiling.
    """
    await anyio.sleep(0.1)
    notification = JSONRPCMessage(
        JSONRPCNotification(
            jsonrpc="2.0",
            method=f"{AIOS_NOTIFICATION_PREFIX}accounts",
            params=dict(ACCOUNTS_PAYLOAD),
        )
    )
    await write_stream.send(SessionMessage(message=notification))


async def run() -> None:
    server = _build_server()
    init_opts = server.create_initialization_options(
        notification_options=NotificationOptions(),
        experimental_capabilities={"aios/connector": {}},
    )
    async with (
        stdio_server() as (read_stream, write_stream),
        anyio.create_task_group() as tg,
    ):
        tg.start_soon(_emit_accounts_after_init, write_stream)
        await server.run(read_stream, write_stream, init_opts)
        tg.cancel_scope.cancel()
