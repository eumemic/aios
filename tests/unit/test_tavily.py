"""Unit tests for the shared Tavily client."""

from __future__ import annotations

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
