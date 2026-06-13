"""Unit tests for the web_search tool handler."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

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
        mock_tavily.return_value = {}
        result = await web_search_handler("sess_01TEST", {"query": "x"})
        assert "error" in result
