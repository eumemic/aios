"""MCP server exposing Signal connector tools.

Tools (invoked by the aios worker over streamable HTTP):

- ``signal_send`` → signal-cli ``send`` RPC
- ``signal_react`` → signal-cli ``sendReaction`` RPC
- ``signal_read_receipt`` → signal-cli ``sendReceipt`` with ``type="read"``

We don't use FastMCP's built-in auth because it requires AuthSettings with
an OAuth-shaped ``issuer_url`` that doesn't fit a static-token deployment.
A thin Starlette middleware wrapping the ASGI app gives us a uniform
``Authorization: Bearer`` contract across loopback and remote deployments.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import urllib.parse
from typing import Any

import structlog
import uvicorn
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .addressing import decode_chat_id
from .markdown import convert_markdown_to_signal_styles
from .rpc import RpcClient

log = structlog.get_logger(__name__)


class BearerAuthMiddleware:
    def __init__(self, app: ASGIApp, *, expected_token: str) -> None:
        self._app = app
        self._expected = expected_token.encode("utf-8")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        headers: list[tuple[bytes, bytes]] = scope.get("headers") or []
        token: bytes | None = None
        for name, value in headers:
            if name.lower() == b"authorization":
                if value.startswith(b"Bearer "):
                    token = value[len(b"Bearer ") :]
                break
        if token is None or not hmac.compare_digest(token, self._expected):
            response = JSONResponse(
                {"error": {"type": "unauthorized", "message": "invalid bearer token"}},
                status_code=401,
            )
            await response(scope, receive, send)
            return
        await self._app(scope, receive, send)


def build_mcp_server(*, rpc: RpcClient) -> FastMCP:
    mcp = FastMCP("aios-signal", stateless_http=True)

    @mcp.tool()
    async def signal_send(chat_id: str, text: str) -> dict[str, Any]:
        """Send a text message to a Signal chat.

        Args:
            chat_id: URL-safe chat ID (DM UUID or URL-safe-base64 group ID).
            text: Message body. Markdown is converted to Signal text styles.
        """
        chat_type, raw_id = decode_chat_id(chat_id)
        stripped, styles = convert_markdown_to_signal_styles(text)
        params: dict[str, Any] = {"message": stripped}
        if styles:
            params["textStyles"] = styles
        if chat_type == "group":
            params["groupId"] = raw_id
        else:
            params["recipient"] = [raw_id]
        result = await rpc.call("send", params)
        return {"sent_at_ms": _extract_timestamp(result)}

    @mcp.tool()
    async def signal_react(
        chat_id: str,
        target_author_uuid: str,
        target_timestamp_ms: int,
        emoji: str,
    ) -> dict[str, Any]:
        """React to a Signal message with an emoji.

        Args:
            chat_id: URL-safe chat ID where the target message lives.
            target_author_uuid: ACI UUID of the message's author.
            target_timestamp_ms: Timestamp of the target message (from inbound metadata).
            emoji: The reaction emoji.
        """
        chat_type, raw_id = decode_chat_id(chat_id)
        params: dict[str, Any] = {
            "emoji": emoji,
            "targetAuthor": target_author_uuid,
            "targetTimestamp": target_timestamp_ms,
        }
        if chat_type == "group":
            params["groupId"] = raw_id
        else:
            params["recipient"] = [raw_id]
        await rpc.call("sendReaction", params)
        return {"status": "ok"}

    @mcp.tool()
    async def signal_read_receipt(
        sender_uuid: str,
        timestamp_ms_list: list[int],
    ) -> dict[str, Any]:
        """Mark one or more messages from ``sender_uuid`` as read.

        Args:
            sender_uuid: ACI UUID of the original sender.
            timestamp_ms_list: Message timestamps (from inbound metadata) to ack.
        """
        # Wrap in list for consistency with send/sendReaction and to match
        # signal-cli's accepted shape on recent versions.
        params: dict[str, Any] = {
            "recipient": [sender_uuid],
            "type": "read",
            "targetTimestamp": list(timestamp_ms_list),
        }
        await rpc.call("sendReceipt", params)
        return {"status": "ok"}

    return mcp


def build_mcp_app(mcp: FastMCP, *, token: str) -> Starlette:
    inner = mcp.streamable_http_app()
    app = Starlette(routes=inner.routes, lifespan=inner.router.lifespan_context)
    app.add_middleware(BearerAuthMiddleware, expected_token=token)
    return app


def _extract_timestamp(rpc_result: Any) -> int:
    if not isinstance(rpc_result, dict):
        raise ValueError(f"signal-cli send returned unexpected shape: {rpc_result!r}")
    ts = rpc_result.get("timestamp")
    if not isinstance(ts, int):
        raise ValueError(f"signal-cli send timestamp not an int: {ts!r}")
    return ts


async def serve_mcp(app: Starlette, *, host: str, port: int) -> None:
    # Uvicorn's Server.serve doesn't react to CancelledError; we flip
    # should_exit and await shutdown ourselves.
    config = uvicorn.Config(app, host=host, port=port, log_config=None, lifespan="on")
    server = uvicorn.Server(config)
    log.info("signal.mcp.serving", host=host, port=port)
    try:
        await server.serve()
    except asyncio.CancelledError:
        server.should_exit = True
        with contextlib.suppress(Exception):
            await asyncio.wait_for(server.shutdown(), timeout=5.0)
        raise


def parse_bind(bind: str) -> tuple[str, int]:
    # urlsplit handles both "host:port" and "[::1]:port" (IPv6) correctly.
    parts = urllib.parse.urlsplit(f"//{bind}")
    if not parts.hostname or parts.port is None:
        raise ValueError(f"invalid bind {bind!r} — expected host:port")
    return parts.hostname, parts.port
