"""The web_fetch tool — fetch a URL and return its content as markdown.

Uses Tavily's /extract endpoint for HTML-to-markdown conversion.
SSRF protection blocks private/internal URLs.

Return shape: {"url": "...", "title": "...", "content": "markdown..."}
The ``title`` is derived from the markdown's first heading: Tavily's /extract
response carries only ``url`` + ``raw_content`` (no title field — that belongs
to /search), so reading a ``title`` key returned ``""`` on every real call.
On error: {"error": "..."}
"""

from __future__ import annotations

import re
from typing import Any

import httpx

from aios.errors import AiosError
from aios.tools.registry import registry
from aios.tools.tavily import WebToolError, tavily_request
from aios.tools.url_safety import is_safe_url


class WebFetchArgumentError(AiosError):
    """Raised when the web_fetch tool is called with malformed arguments."""

    error_type = "web_fetch_argument_error"
    status_code = 400


_MAX_CONTENT_CHARS = 100_000

# First markdown ATX heading (``# `` … ``###### ``) — Tavily's /extract gives no
# title, so we derive one from the converted markdown's first heading.
_HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)


def _title_from_markdown(content: str) -> str:
    """Best-effort page title: the text of the markdown's first heading, or ``""``."""
    match = _HEADING_RE.search(content)
    return match.group(1) if match else ""


WEB_FETCH_DESCRIPTION = (
    "Fetch a URL and return its content as markdown. Uses Tavily's "
    "extract endpoint for HTML-to-markdown conversion. Content is "
    "truncated to 100k characters. Private/internal URLs are blocked."
)

WEB_FETCH_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "url": {"type": "string", "description": "The URL to fetch content from."},
    },
    "required": ["url"],
    "additionalProperties": False,
}


async def web_fetch_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Handler for the web_fetch tool. See module docstring for the return shape."""
    url = arguments.get("url")
    if not isinstance(url, str) or not url.strip():
        raise WebFetchArgumentError("web_fetch tool requires a non-empty 'url' string")

    if not is_safe_url(url):
        return {"error": "Blocked: URL targets a private/internal address"}

    try:
        response = await tavily_request("extract", {"urls": [url]})
        result = response["results"][0]
        # /extract returns only ``raw_content`` (markdown); there is no ``title``
        # or ``content`` field to read (those are /search fields), so derive the
        # title from the markdown's first heading.
        content = (result.get("raw_content") or "")[:_MAX_CONTENT_CHARS]
        return {"url": url, "title": _title_from_markdown(content), "content": content}
    except WebToolError:
        raise
    except httpx.HTTPStatusError as exc:
        return {"error": f"HTTP {exc.response.status_code}: {exc.response.text[:200]}"}
    except httpx.TimeoutException:
        return {"error": "Request timed out fetching URL"}
    except (KeyError, IndexError):
        return {"error": "No content returned for URL"}


def _register() -> None:
    registry.register(
        name="web_fetch",
        description=WEB_FETCH_DESCRIPTION,
        parameters_schema=WEB_FETCH_PARAMETERS_SCHEMA,
        handler=web_fetch_handler,
        transport="both",
    )


_register()
