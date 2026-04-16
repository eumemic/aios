"""MCP server exposing Signal connector tools.

Tools (invoked by the aios worker over streamable HTTP):

- ``signal_send(chat_id, text)`` → signal-cli ``send`` RPC
- ``signal_react(chat_id, target_author_uuid, target_timestamp_ms, emoji)`` →
  signal-cli ``sendReaction`` RPC
- ``signal_read_receipt(sender_uuid, timestamp_ms_list)`` → signal-cli
  ``sendReceipt`` RPC with ``type="read"``

Authentication is a single static bearer token, presented as
``Authorization: Bearer <token>``. This matches the ``static_bearer`` vault
credential shape aios already understands.

We don't use FastMCP's built-in auth because it requires AuthSettings with an
issuer URL (OAuth resource-server shape) that doesn't fit a static-token
deployment. A thin Starlette middleware wrapping the ASGI app is simpler and
keeps the contract uniform across loopback and remote deployments.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
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
    """Starlette-style ASGI middleware enforcing ``Authorization: Bearer``."""

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


def build_mcp_server(
    *,
    rpc: RpcClient,
    bot_account_uuid: str,  # accepted for future use (self-reference in metadata, logs)
) -> FastMCP:
    """Build the FastMCP instance with the three Signal tools registered."""
    _ = bot_account_uuid  # kept for future use
    mcp = FastMCP("aios-signal", stateless_http=True)

    @mcp.tool()
    async def signal_send(chat_id: str, text: str) -> dict[str, Any]:
        """Send a text message to a Signal chat.

        Args:
            chat_id: URL-safe chat ID (DM UUID or URL-safe-base64 group ID).
            text: Message body. Markdown formatting is converted to Signal text styles.

        Returns:
            ``{"sent_at_ms": <signal-cli timestamp>}``.
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
        return {"sent_at_ms": int(_extract_timestamp(result))}

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
            target_timestamp_ms: Timestamp of the target message (from its inbound metadata).
            emoji: The reaction emoji. Pass ``""`` is invalid; use signal-cli remove semantics.
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
            timestamp_ms_list: List of message timestamps (from inbound metadata) to ack.
        """
        params: dict[str, Any] = {
            "recipient": sender_uuid,
            "type": "read",
            "targetTimestamp": list(timestamp_ms_list),
        }
        await rpc.call("sendReceipt", params)
        return {"status": "ok"}

    return mcp


def build_mcp_app(mcp: FastMCP, *, token: str) -> Starlette:
    """Wrap the FastMCP streamable-http app with bearer auth."""
    inner = mcp.streamable_http_app()
    app = Starlette(
        routes=inner.routes,
        lifespan=inner.router.lifespan_context,
    )
    app.add_middleware(BearerAuthMiddleware, expected_token=token)
    return app


def _extract_timestamp(rpc_result: Any) -> int:
    """Pull a timestamp out of signal-cli's ``send`` response.

    signal-cli's ``send`` returns ``{"timestamp": <ms>, "results": [...]}``.
    We keep the parse defensive: the RPC shape has varied across versions.
    """
    if isinstance(rpc_result, dict):
        ts = rpc_result.get("timestamp")
        if isinstance(ts, int):
            return ts
        if isinstance(ts, str) and ts.isdigit():
            return int(ts)
    raise ValueError(f"signal-cli send returned unexpected shape: {rpc_result!r}")


async def serve_mcp(app: Starlette, *, host: str, port: int) -> None:
    """Run the MCP server on uvicorn, exiting cleanly on task cancellation.

    Uvicorn's ``Server.serve`` does not react to :exc:`asyncio.CancelledError`
    on its own — we have to flip ``should_exit`` and await shutdown ourselves.
    """
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
    """Split ``host:port`` into (host, port) with type-checked port."""
    host, _, port_str = bind.rpartition(":")
    if not host or not port_str:
        raise ValueError(f"invalid bind {bind!r} — expected host:port")
    return host, int(port_str)
