"""MCP server exposing Signal connector tools.

Tools (invoked by the aios worker over streamable HTTP):

- ``signal_send`` → signal-cli ``send`` RPC
- ``signal_react`` → signal-cli ``sendReaction`` RPC

Read receipts are NOT an agent-facing tool. The semantically-correct
moment to mark a message as read is when the agent's ``reacting_to``
watermark advances past it — the model chose to reason about it — not
when the model happens to call a tool. That's a connector-side
automation, not a tool. Punted to a follow-up (see SMOKE_TEST_NOTES.md).

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
from types import MethodType
from typing import Any

import structlog
import uvicorn
from mcp.server.fastmcp import Context, FastMCP
from mcp.types import RequestParams
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .addressing import decode_chat_id
from .inbound import SignalInboundBroker
from .markdown import convert_markdown_to_signal_styles
from .prompts import build_instructions
from .rpc import RpcClient

log = structlog.get_logger(__name__)


# Key under a JSON-RPC tool-call request's ``_meta`` populated by the aios
# worker when dispatching an MCP tool call while a focal channel is set. Its
# value is the account-relative focal channel: ``<bot_uuid>/<chat_id>``.
# See ``aios.harness.channels`` for the client-side contract.
_FOCAL_CHANNEL_META_KEY = "aios.focal_channel"


def focal_chat_id_from_meta(
    meta: RequestParams.Meta | None,
    *,
    account_id: str | None = None,
) -> str:
    """Extract the Signal ``chat_id`` from an MCP request's ``_meta``.

    aios injects ``aios.focal_channel`` for MCP tool calls while focal is set.
    Its value is account-relative: ``<bot_uuid>/<chat_id>``. Missing,
    malformed, or wrong-account meta raises so the agent sees that it must
    focus a channel on this account before using Signal response tools.
    """
    focal_channel: Any = None
    if meta is not None:
        extra = getattr(meta, "model_extra", None) or {}
        focal_channel = extra.get(_FOCAL_CHANNEL_META_KEY)
    if not isinstance(focal_channel, str) or not focal_channel:
        raise ValueError(
            "signal tools require a focal channel — aios should inject "
            f"{_FOCAL_CHANNEL_META_KEY!r} in _meta when focal is set"
        )
    account, sep, chat_id = focal_channel.partition("/")
    if not sep or not account or not chat_id:
        raise ValueError(
            "signal focal channel must be account-relative "
            f"'<account>/<chat_id>'; got {focal_channel!r}"
        )
    if account_id is not None and account != account_id:
        raise ValueError(
            "signal focal channel belongs to a different account "
            f"({account!r}, expected {account_id!r})"
        )
    return chat_id


def build_send_params(chat_id: str, text: str) -> dict[str, Any]:
    """Translate ``(chat_id, text)`` into signal-cli ``send`` params.

    Pure function: decodes the URL-safe chat_id, applies Signal's
    markdown-to-textStyles transformation, and attaches either
    ``recipient`` (DM) or ``groupId`` (group) per the decoded kind.
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
    return params


def build_react_params(
    chat_id: str,
    target_author_uuid: str,
    target_timestamp_ms: int,
    emoji: str,
) -> dict[str, Any]:
    """Translate a react request into signal-cli ``sendReaction`` params."""
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
    return params


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
    rpc: RpcClient,
    bot_uuid: str,
    phone: str,
    groups: list[Any] | None = None,
    contact_names: dict[str, str] | None = None,
    inbound_broker: SignalInboundBroker | None = None,
) -> FastMCP:
    # If `listContacts` included the bot's own entry, use its display
    # name as the profile_name for the identity block.  Absent when the
    # bot hasn't set a profile or isn't listed in its own contacts —
    # `build_instructions` renders nothing for that case.
    profile_name = (contact_names or {}).get(bot_uuid)
    mcp = FastMCP(
        "aios-signal",
        instructions=build_instructions(
            bot_uuid=bot_uuid,
            phone=phone,
            profile_name=profile_name,
            groups=groups,
            contact_names=contact_names,
        ),
        stateless_http=False,
    )

    if inbound_broker is not None:
        _install_aios_inbound_capability(mcp)

        @mcp.tool(
            name="aios_inbound_subscribe",
            description="Internal aios inbound subscription hook.",
            meta={"aios.internal": True},
        )
        async def aios_inbound_subscribe(
            account_id: str,
            ctx: Context[Any, Any, Any],
            since_event_id: str | None = None,
        ) -> dict[str, Any]:
            return await inbound_broker.subscribe(
                account_id=account_id,
                since_event_id=since_event_id,
                session=ctx.session,
            )

    @mcp.tool()
    async def signal_send(text: str, ctx: Context[Any, Any, Any]) -> dict[str, Any]:
        """Send a text message to your focal Signal chat.

        The chat id is taken implicitly from your focal channel —
        aios injects it via the JSON-RPC ``_meta`` field on each call.
        Set focal with the built-in ``switch_channel`` tool.

        Args:
            text: Message body. Markdown is converted to Signal text styles.
        """
        chat_id = focal_chat_id_from_meta(ctx.request_context.meta, account_id=bot_uuid)
        params = build_send_params(chat_id, text)
        result = await rpc.call("send", params)
        ts = _extract_timestamp(result)
        return {"sent_at_ms": ts} if ts is not None else {"status": "ok"}

    @mcp.tool()
    async def signal_react(
        target_author_uuid: str,
        target_timestamp_ms: int,
        emoji: str,
        ctx: Context[Any, Any, Any],
    ) -> dict[str, Any]:
        """React to a message in your focal Signal chat with an emoji.

        The chat id is taken implicitly from your focal channel —
        aios injects it via the JSON-RPC ``_meta`` field on each call.

        The target message is identified by ``(target_author_uuid, target_timestamp_ms)``.
        Every inbound Signal message in your conversation starts with a header line like
        ``[channel=... · from=... · sender_uuid=<uuid> · timestamp_ms=<ms> (<iso>)]``.
        Copy ``sender_uuid`` and the raw ``timestamp_ms`` integer from that header; do
        not construct them yourself.

        Args:
            target_author_uuid: The ``sender_uuid`` from the header of the message
                you're reacting to.
            target_timestamp_ms: The ``timestamp_ms`` integer from the header of the
                message you're reacting to.
            emoji: The reaction emoji.
        """
        chat_id = focal_chat_id_from_meta(ctx.request_context.meta, account_id=bot_uuid)
        params = build_react_params(chat_id, target_author_uuid, target_timestamp_ms, emoji)
        await rpc.call("sendReaction", params)
        return {"status": "ok"}

    return mcp


def _install_aios_inbound_capability(mcp: FastMCP) -> None:
    """Advertise the aios inbound extension in MCP initialize."""
    server = mcp._mcp_server  # FastMCP exposes no public hook for this yet.
    base = server.create_initialization_options

    def create_initialization_options(
        self: Any,
        notification_options: Any | None = None,
        experimental_capabilities: dict[str, dict[str, Any]] | None = None,
    ) -> Any:
        experimental = dict(experimental_capabilities or {})
        experimental["aiosInbound"] = {
            "version": 1,
            "connectorId": "signal",
        }
        return base(notification_options, experimental)

    server.create_initialization_options = MethodType(create_initialization_options, server)  # type: ignore[method-assign]


def build_mcp_app(mcp: FastMCP, *, token: str) -> Starlette:
    inner = mcp.streamable_http_app()
    app = Starlette(routes=inner.routes, lifespan=inner.router.lifespan_context)
    app.add_middleware(BearerAuthMiddleware, expected_token=token)
    return app


def _extract_timestamp(rpc_result: Any) -> int | None:
    """Pull the send timestamp from signal-cli's ``send`` result, or ``None``.

    signal-cli delivers DM sends with ``{"timestamp": <ms>, ...}``, but group
    sends in 0.14.x return a bare ``null`` even on success — the RPC doesn't
    carry a timestamp. RPC-level delivery failures raise ``RpcError`` in the
    transport layer, so if we reach this function the send *did* happen; a
    missing timestamp just means we don't have an ID to hand back. Return
    ``None`` and let the caller convey "sent, no ID" to the model.
    """
    if not isinstance(rpc_result, dict):
        return None
    ts = rpc_result.get("timestamp")
    return ts if isinstance(ts, int) else None


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
