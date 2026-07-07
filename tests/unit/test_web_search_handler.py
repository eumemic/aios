"""Unit tests for the web_search tool handler."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from aios.tools.invoke import ToolBail
from aios.tools.tavily import WebToolError
from aios.tools.web_search import WebSearchArgumentError, web_search_handler


@pytest.fixture
def mock_tavily() -> Any:
    """Patch tavily_request for web_search."""
    with patch("aios.tools.web_search.tavily_request", new_callable=AsyncMock) as m:
        yield m


_CANNED_RESPONSE: dict[str, Any] = {
    "results": [
        {"title": "Result 1", "url": "https://example.com/1", "content": "Description 1"},
        {"title": "Result 2", "url": "https://example.com/2", "content": "Description 2"},
    ]
}


class TestWebSearchHandler:
    async def test_valid_query_returns_results(self, mock_tavily: AsyncMock) -> None:
        mock_tavily.return_value = _CANNED_RESPONSE
        result = await web_search_handler("sess_01TEST", {"query": "python testing"})
        assert len(result["results"]) == 2
        assert result["results"][0]["title"] == "Result 1"
        assert result["results"][0]["url"] == "https://example.com/1"
        assert result["results"][0]["description"] == "Description 1"

    async def test_missing_query_raises(self) -> None:
        with pytest.raises(WebSearchArgumentError):
            await web_search_handler("sess_01TEST", {})

    async def test_empty_query_raises(self) -> None:
        with pytest.raises(WebSearchArgumentError):
            await web_search_handler("sess_01TEST", {"query": ""})

    async def test_default_limit_is_5(self, mock_tavily: AsyncMock) -> None:
        mock_tavily.return_value = _CANNED_RESPONSE
        await web_search_handler("sess_01TEST", {"query": "test"})
        call_payload = mock_tavily.call_args[0][1]
        assert call_payload["max_results"] == 5

    async def test_custom_limit(self, mock_tavily: AsyncMock) -> None:
        mock_tavily.return_value = _CANNED_RESPONSE
        await web_search_handler("sess_01TEST", {"query": "test", "limit": 10})
        call_payload = mock_tavily.call_args[0][1]
        assert call_payload["max_results"] == 10

    async def test_limit_capped_at_20(self, mock_tavily: AsyncMock) -> None:
        mock_tavily.return_value = _CANNED_RESPONSE
        await web_search_handler("sess_01TEST", {"query": "test", "limit": 50})
        call_payload = mock_tavily.call_args[0][1]
        assert call_payload["max_results"] == 20

    async def test_tavily_error_raises(self, mock_tavily: AsyncMock) -> None:
        mock_tavily.side_effect = WebToolError("TAVILY_API_KEY not set")
        with pytest.raises(WebToolError):
            await web_search_handler("sess_01TEST", {"query": "test"})

    async def test_partial_result_missing_title_url_defaults(self, mock_tavily: AsyncMock) -> None:
        """Tavily /search guarantees no per-entry fields. A result lacking
        'title'/'url' must default to '' (like web_fetch's defensive .get), not
        raise a KeyError that escapes the handler's except and — via
        _classify_tool_error — needlessly evicts the session's sandbox while the
        model sees a cryptic 'KeyError: title' instead of a legible result."""
        mock_tavily.return_value = {"results": [{"content": "desc only, no title or url"}]}
        result = await web_search_handler("sess_01TEST", {"query": "x"})
        assert result["results"][0]["title"] == ""
        assert result["results"][0]["url"] == ""
        assert result["results"][0]["description"] == "desc only, no title or url"

    async def test_malformed_response_without_results_returns_error(
        self, mock_tavily: AsyncMock
    ) -> None:
        """A 200 response missing the 'results' key surfaces as a legible tool
        error (mirroring web_fetch), not a KeyError that escapes and evicts the
        sandbox."""
        # Post-#1680: raises ``ToolBail`` (one typed failure channel) rather than
        # returning a bare ``{"error": ...}`` dict — still no KeyError escapes.
        mock_tavily.return_value = {}
        with pytest.raises(ToolBail):
            await web_search_handler("sess_01TEST", {"query": "x"})

    @pytest.mark.parametrize(
        "exc",
        [
            httpx.ConnectError("connection refused"),
            httpx.ReadError("read failed"),
            httpx.RemoteProtocolError("protocol error"),
            httpx.PoolTimeout("pool exhausted"),
            httpx.TimeoutException("timed out"),
        ],
    )
    async def test_transport_faults_raise_toolbail_not_raw_httpx(
        self, mock_tavily: AsyncMock, exc: httpx.HTTPError
    ) -> None:
        """A benign upstream transport blip (ConnectError/ReadError/… — the whole
        ``httpx.TransportError`` family, plus timeouts) must convert to ``ToolBail``,
        NOT propagate raw. A raw httpx error reaches ``_classify_tool_error`` →
        ``evict=True`` → the sandbox is torn down on a transient network failure
        (aios#1697). Mirrors ``http_request``'s broad ``httpx.HTTPError`` catch."""
        mock_tavily.side_effect = exc
        with pytest.raises(ToolBail):
            await web_search_handler("sess_01TEST", {"query": "x"})

    async def test_http_status_error_raises_toolbail(self, mock_tavily: AsyncMock) -> None:
        """A non-2xx from Tavily/upstream is an expected failure → ``ToolBail``
        carrying the status, never the raw httpx error (which would evict)."""
        request = httpx.Request("GET", "https://example.com")
        response = httpx.Response(503, text="upstream down", request=request)
        mock_tavily.side_effect = httpx.HTTPStatusError(
            "server error", request=request, response=response
        )
        with pytest.raises(ToolBail) as excinfo:
            await web_search_handler("sess_01TEST", {"query": "x"})
        assert "503" in excinfo.value.message
