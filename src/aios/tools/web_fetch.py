"""The web_fetch tool — fetch a URL and return its content as markdown.

Uses Tavily's /extract endpoint for HTML-to-markdown conversion.
SSRF protection blocks private/internal URLs.

Return shape: {"url": "...", "title": "...", "content": "markdown..."}
The ``title`` is derived from the markdown's first heading: Tavily's /extract
response carries only ``url`` + ``raw_content`` (no title field — that belongs
to /search), so reading a ``title`` key returned ``""`` on every real call.
When the content is cut at the char cap the result also carries ``"truncated":
True`` (present only when truncated, mirroring ``http_request``) so the model
never mistakes a cut page for a complete one.
On an expected failure (SSRF block, non-2xx, timeout, empty response) raises
:class:`~aios.tools.invoke.ToolBail`; a TAVILY-config failure raises the client-class
:class:`~aios.tools.tavily.WebToolError`. The single event writer stamps ``is_error``
(session path) — #1680. On the workflow-run path these raises are translated back to
value-shaped ``{"error": ...}`` dicts at the ``invoke_run_tool`` seam, preserving the
runs errors-as-values contract.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

import httpx

from aios.errors import AiosError
from aios.tools.invoke import ToolBail
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
    "truncated to 100k characters; when content is cut the result carries "
    '"truncated": true so you never mistake a truncated page for a complete '
    "one. Private/internal URLs are blocked."
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

    # is_safe_url does a blocking getaddrinfo; offload it so the SSRF pre-flight
    # never stalls the event loop (matches services/vault_oauth._guard_url).
    if not await asyncio.to_thread(is_safe_url, url):
        raise ToolBail("Blocked: URL targets a private/internal address")

    try:
        response = await tavily_request("extract", {"urls": [url]})
        result = response["results"][0]
        # /extract returns only ``raw_content`` (markdown); there is no ``title``
        # or ``content`` field to read (those are /search fields), so derive the
        # title from the markdown's first heading.
        raw = result.get("raw_content") or ""
        content = raw[:_MAX_CONTENT_CHARS]
        out: dict[str, Any] = {
            "url": url,
            "title": _title_from_markdown(content),
            "content": content,
        }
        # Signal a cut page explicitly, opt-in like ``http_request`` (aios#1294):
        # the flag is present ONLY when truncated, so the model can branch on its
        # mere presence. ``>`` strictly — exactly-100k is complete, not cut.
        if len(raw) > _MAX_CONTENT_CHARS:
            out["truncated"] = True
        return out
    except WebToolError:
        # Already a client-class AiosError (status 400): re-raise so the single
        # writer classifies it as a clean refusal, not a sandbox-evicting failure.
        raise
    except httpx.HTTPStatusError as exc:
        # A non-2xx from Tavily/upstream is an EXPECTED failure the model reads and
        # retries — raise ToolBail (a benign refusal), NOT the raw httpx error, which
        # _classify_tool_error would treat as an internal failure and evict the sandbox.
        raise ToolBail(f"HTTP {exc.response.status_code}: {exc.response.text[:200]}") from exc
    except httpx.TimeoutException as exc:
        raise ToolBail("Request timed out fetching URL") from exc
    except (KeyError, IndexError) as exc:
        raise ToolBail("No content returned for URL") from exc


def _register() -> None:
    registry.register(
        name="web_fetch",
        description=WEB_FETCH_DESCRIPTION,
        parameters_schema=WEB_FETCH_PARAMETERS_SCHEMA,
        handler=web_fetch_handler,
        transport="both",
    )


_register()
