"""Unit tests for the shared Tavily client."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from aios.tools.invoke import ToolBail
from aios.tools.tavily import tavily_request


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ConnectError("connection refused"),
        httpx.TimeoutException("timed out"),
    ],
)
async def test_transport_faults_are_tool_bails(exc: httpx.HTTPError) -> None:
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.post.side_effect = exc
    settings = MagicMock(tavily_api_key="key")
    with (
        patch("aios.tools.tavily.get_settings", return_value=settings),
        patch("aios.tools.tavily.httpx.AsyncClient", return_value=client),
        pytest.raises(ToolBail) as raised,
    ):
        await tavily_request("search", {"query": "x"})
    assert "search" in raised.value.message


async def test_http_status_is_a_tool_bail() -> None:
    request = httpx.Request("POST", "https://api.tavily.com/extract")
    response = httpx.Response(503, text="upstream down", request=request)
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.post.return_value = response
    settings = MagicMock(tavily_api_key="key")
    with (
        patch("aios.tools.tavily.get_settings", return_value=settings),
        patch("aios.tools.tavily.httpx.AsyncClient", return_value=client),
        pytest.raises(ToolBail) as raised,
    ):
        await tavily_request("extract", {"urls": ["https://example.com"]})
    assert "503" in raised.value.message


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(b"", id="empty"),
        pytest.param(b"<!doctype html><html>captive portal</html>", id="html"),
        pytest.param(b'{"results": [', id="truncated-json"),
        pytest.param(b"not json", id="plain-text"),
    ],
)
async def test_non_json_200_response_is_a_tool_bail(body: bytes) -> None:
    """A 200 with an empty/non-JSON body makes ``resp.json()`` raise
    ``json.JSONDecodeError`` — a ``ValueError``, NOT an ``httpx.HTTPError`` —
    so the broadened ``except httpx.HTTPError`` arm never sees it. Without an
    explicit conversion it escapes to ``_classify_tool_error``'s catch-all arm
    (``evict=True``), recycling the session's sandbox on a benign network blip
    (aios#1697) — precisely the eviction the docstrings guarantee never happens.
    Convert it to a legible ``ToolBail`` at the shared seam covering both
    web_search and web_fetch."""
    request = httpx.Request("POST", "https://api.tavily.com/search")
    response = httpx.Response(200, content=body, request=request)
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.post.return_value = response
    settings = MagicMock(tavily_api_key="key")
    with (
        patch("aios.tools.tavily.get_settings", return_value=settings),
        patch("aios.tools.tavily.httpx.AsyncClient", return_value=client),
        pytest.raises(ToolBail) as raised,
    ):
        await tavily_request("search", {"query": "x"})
    assert "non-JSON" in raised.value.message
    assert "search" in raised.value.message
    assert isinstance(raised.value.__cause__, json.JSONDecodeError)


async def test_valid_json_200_response_is_returned() -> None:
    """The new ``except json.JSONDecodeError`` must not swallow a well-formed
    JSON 200 — happy-path regression for the parsing guard."""
    request = httpx.Request("POST", "https://api.tavily.com/search")
    response = httpx.Response(
        200,
        json={"results": [{"title": "t", "url": "u", "content": "c"}]},
        request=request,
    )
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.post.return_value = response
    settings = MagicMock(tavily_api_key="key")
    with (
        patch("aios.tools.tavily.get_settings", return_value=settings),
        patch("aios.tools.tavily.httpx.AsyncClient", return_value=client),
    ):
        result = await tavily_request("search", {"query": "x"})
    assert result == {"results": [{"title": "t", "url": "u", "content": "c"}]}
