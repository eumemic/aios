"""Tests for mcp.py — telegram_send tool + bearer-auth middleware."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from mcp.server.fastmcp import Context
from mcp.shared.context import RequestContext
from mcp.types import RequestParams
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from telegram import Bot, Message

from aios_telegram.mcp import (
    BearerAuthMiddleware,
    build_mcp_server,
    focal_chat_id_from_meta,
    parse_bind,
)


def _ctx_with_focal(chat_id: str | None) -> Context[Any, Any, Any]:
    rc = MagicMock(spec=RequestContext)
    if chat_id is None:
        rc.meta = None
    else:
        rc.meta = RequestParams.Meta.model_validate({"aios.focal_channel_path": chat_id})
    return Context(request_context=rc)


# ─── focal_chat_id_from_meta ────────────────────────────────────────────────


def test_focal_meta_dm_id() -> None:
    meta = RequestParams.Meta.model_validate({"aios.focal_channel_path": "123456789"})
    assert focal_chat_id_from_meta(meta) == 123456789


def test_focal_meta_group_negative_id() -> None:
    meta = RequestParams.Meta.model_validate({"aios.focal_channel_path": "-1001234567890"})
    assert focal_chat_id_from_meta(meta) == -1001234567890


def test_focal_meta_missing_raises() -> None:
    with pytest.raises(ValueError, match="focal channel"):
        focal_chat_id_from_meta(None)


def test_focal_meta_not_integer_raises() -> None:
    meta = RequestParams.Meta.model_validate({"aios.focal_channel_path": "abc"})
    with pytest.raises(ValueError, match="integer"):
        focal_chat_id_from_meta(meta)


# ─── telegram_send tool ─────────────────────────────────────────────────────


async def _call_send(bot: Bot, text: str, *, focal: str | None) -> dict[str, Any]:
    mcp = build_mcp_server(bot=bot)
    result = await mcp._tool_manager.call_tool(
        "telegram_send",
        {"text": text},
        _ctx_with_focal(focal),
        convert_result=True,
    )
    assert isinstance(result, tuple)
    structured: Any = result[1]
    assert isinstance(structured, dict)
    return structured


async def test_telegram_send_happy_path() -> None:
    sent = MagicMock(spec=Message)
    sent.message_id = 42
    bot = MagicMock(spec=Bot)
    bot.send_message = AsyncMock(return_value=sent)

    out = await _call_send(bot, "hello", focal="123456789")

    assert out == {"message_id": 42}
    bot.send_message.assert_awaited_once_with(chat_id=123456789, text="hello")


async def test_telegram_send_group_negative_chat_id() -> None:
    sent = MagicMock(spec=Message)
    sent.message_id = 7
    bot = MagicMock(spec=Bot)
    bot.send_message = AsyncMock(return_value=sent)

    out = await _call_send(bot, "yo", focal="-1001234567890")

    assert out == {"message_id": 7}
    bot.send_message.assert_awaited_once_with(chat_id=-1001234567890, text="yo")


# ─── bearer auth middleware ─────────────────────────────────────────────────


def _auth_app(token: str) -> Starlette:
    async def ok(_request: Any) -> PlainTextResponse:
        return PlainTextResponse("ok")

    app = Starlette(routes=[Route("/", ok)])
    app.add_middleware(BearerAuthMiddleware, expected_token=token)
    return app


async def test_bearer_auth_rejects_missing_header() -> None:
    app = _auth_app("secret")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t"
    ) as client:
        resp = await client.get("/")
    assert resp.status_code == 401


async def test_bearer_auth_rejects_wrong_token() -> None:
    app = _auth_app("secret")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t"
    ) as client:
        resp = await client.get("/", headers={"Authorization": "Bearer nope"})
    assert resp.status_code == 401


async def test_bearer_auth_accepts_correct_token() -> None:
    app = _auth_app("secret")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t"
    ) as client:
        resp = await client.get("/", headers={"Authorization": "Bearer secret"})
    assert resp.status_code == 200
    assert resp.text == "ok"


# ─── parse_bind ─────────────────────────────────────────────────────────────


def test_parse_bind_ipv4() -> None:
    assert parse_bind("127.0.0.1:9200") == ("127.0.0.1", 9200)


def test_parse_bind_ipv6() -> None:
    assert parse_bind("[::1]:9200") == ("::1", 9200)


def test_parse_bind_invalid() -> None:
    with pytest.raises(ValueError):
        parse_bind("not-a-bind")
