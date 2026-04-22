"""MCP server exposing Telegram connector tools.

Tools (invoked by the aios worker over streamable HTTP):

- ``telegram_send`` → python-telegram-bot ``Bot.send_message``

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
from mcp.server.fastmcp import Context, FastMCP
from mcp.types import RequestParams
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send
from telegram import Bot

from .prompts import build_instructions

log = structlog.get_logger(__name__)


# Key under a JSON-RPC tool-call request's ``_meta`` populated by the
# aios worker when dispatching a connection-provided tool.  Its value
# is the focal-channel path suffix — for Telegram's 3-segment address
# ``telegram/<bot_id>/<chat_id>``, the suffix is just ``<chat_id>`` and
# can be parsed as a signed integer.
_FOCAL_CHANNEL_META_KEY = "aios.focal_channel_path"


def focal_chat_id_from_meta(meta: RequestParams.Meta | None) -> int:
    """Extract the Telegram ``chat_id`` from an MCP request's ``_meta``.

    aios injects ``aios.focal_channel_path`` for connection-provided tool
    calls; its value is the full focal-channel suffix (stripped of
    ``<connector>/<account>``), which for a 3-segment Telegram address
    is the decimal chat id. Missing / malformed meta raises — the agent
    shouldn't be able to reach these tools without a focal channel set
    (aios filters them out of the tool list when focal is NULL), so any
    absence here is a real error to surface.
    """
    path: Any = None
    if meta is not None:
        extra = getattr(meta, "model_extra", None) or {}
        path = extra.get(_FOCAL_CHANNEL_META_KEY)
    if not isinstance(path, str) or not path:
        raise ValueError(
            "telegram tools require a focal channel — aios should inject "
            f"{_FOCAL_CHANNEL_META_KEY!r} in _meta when focal is set"
        )
    try:
        return int(path)
    except ValueError as e:
        raise ValueError(f"telegram chat_id must be an integer; got {path!r}") from e


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


def build_mcp_server(
    *,
    bot: Bot,
    bot_id: int,
    first_name: str,
    username: str | None = None,
) -> FastMCP:
    mcp = FastMCP(
        "aios-telegram",
        instructions=build_instructions(
            bot_id=bot_id,
            first_name=first_name,
            username=username,
        ),
        stateless_http=True,
    )

    @mcp.tool()
    async def telegram_send(text: str, ctx: Context[Any, Any, Any]) -> dict[str, Any]:
        """Send a text message to your focal Telegram chat.

        The chat id is taken implicitly from your focal channel —
        aios injects it via the JSON-RPC ``_meta`` field on each call.
        Set focal with the built-in ``switch_channel`` tool.

        Args:
            text: Message body. Plain text only — markdown is not rendered.
        """
        chat_id = focal_chat_id_from_meta(ctx.request_context.meta)
        sent = await bot.send_message(chat_id=chat_id, text=text)
        return {"message_id": sent.message_id}

    return mcp


def build_mcp_app(mcp: FastMCP, *, token: str) -> Starlette:
    inner = mcp.streamable_http_app()
    app = Starlette(routes=inner.routes, lifespan=inner.router.lifespan_context)
    app.add_middleware(BearerAuthMiddleware, expected_token=token)
    return app


async def serve_mcp(app: Starlette, *, host: str, port: int) -> None:
    # Uvicorn's Server.serve doesn't react to CancelledError; we flip
    # should_exit and await shutdown ourselves.
    config = uvicorn.Config(app, host=host, port=port, log_config=None, lifespan="on")
    server = uvicorn.Server(config)
    log.info("telegram.mcp.serving", host=host, port=port)
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
