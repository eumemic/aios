"""The web_search tool — search the web using Tavily.

Uses Tavily's /search endpoint. Returns a list of results with
title, URL, and description.

Return shape: {"results": [{"title": "...", "url": "...", "description": "..."}]}
On an expected failure (non-2xx, timeout, empty response) raises
:class:`~aios.tools.invoke.ToolBail`; a TAVILY-config failure raises the client-class
:class:`~aios.tools.tavily.WebToolError` (#1680). On the workflow-run path these raises
are translated back to value-shaped ``{"error": ...}`` dicts at the ``invoke_run_tool``
seam, preserving the runs errors-as-values contract.
"""

from __future__ import annotations

from typing import Any

import httpx

from aios.errors import AiosError
from aios.tools.invoke import ToolBail
from aios.tools.registry import registry
from aios.tools.tavily import WebToolError, tavily_request

_DEFAULT_LIMIT = 5
_MAX_LIMIT = 20


class WebSearchArgumentError(AiosError):
    """Raised when the web_search tool is called with malformed arguments."""

    error_type = "web_search_argument_error"
    status_code = 400


WEB_SEARCH_DESCRIPTION = (
    "Search the web using Tavily. Returns up to `limit` results "
    "(default 5, max 20) each with a title, URL, and description."
)

WEB_SEARCH_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "The search query."},
        "limit": {
            "type": "integer",
            "description": "Maximum number of results (default 5, max 20).",
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}


async def web_search_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Handler for the web_search tool. See module docstring for the return shape."""
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        raise WebSearchArgumentError("web_search tool requires a non-empty 'query' string")

    limit = arguments.get("limit", _DEFAULT_LIMIT)
    if not isinstance(limit, int) or limit < 1:
        limit = _DEFAULT_LIMIT

    try:
        response = await tavily_request(
            "search",
            {"query": query, "max_results": min(limit, _MAX_LIMIT), "include_images": False},
        )
        # Tavily /search guarantees no per-entry fields, so default each one
        # (like web_fetch). A bracket read would raise KeyError that escapes
        # this except and — via _classify_tool_error — evicts the session's
        # sandbox while the model sees a cryptic Python error, not a tool error.
        normalized = [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "description": r.get("content", ""),
            }
            for r in response["results"]
        ]
        return {"results": normalized}
    except WebToolError:
        # Already a client-class AiosError (status 400): re-raise so the single
        # writer classifies it as a clean refusal, not a sandbox-evicting failure.
        raise
    except httpx.HTTPStatusError as exc:
        # An EXPECTED non-2xx — raise ToolBail (benign refusal), NOT the raw httpx
        # error, which _classify_tool_error would treat as internal + evict the sandbox.
        raise ToolBail(
            f"HTTP {exc.response.status_code}: {exc.response.text[:200]}"
        ) from exc
    except httpx.TimeoutException as exc:
        raise ToolBail("Search request timed out") from exc
    except (KeyError, IndexError) as exc:
        # A 200 response missing the 'results' key — surface a legible tool
        # error instead of letting the KeyError escape and evict the sandbox.
        raise ToolBail("No results returned for query") from exc


def _register() -> None:
    registry.register(
        name="web_search",
        description=WEB_SEARCH_DESCRIPTION,
        parameters_schema=WEB_SEARCH_PARAMETERS_SCHEMA,
        handler=web_search_handler,
        transport="both",
    )


_register()
