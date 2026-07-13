"""Tavily API client for web_fetch and web_search tools."""

from __future__ import annotations

from typing import Any

import httpx

from aios.config import get_settings
from aios.errors import AiosError
from aios.tools.invoke import ToolBail


class WebToolError(AiosError):
    """Raised when a web tool operation fails."""

    error_type = "web_tool_error"
    status_code = 400


async def tavily_request(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST to Tavily API. Raises WebToolError if no API key configured."""
    api_key = get_settings().tavily_api_key
    if not api_key:
        raise WebToolError("TAVILY_API_KEY not set. Get a free key at https://app.tavily.com")
    payload["api_key"] = api_key
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"https://api.tavily.com/{endpoint.lstrip('/')}",
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            result: dict[str, Any] = resp.json()
            return result
    except httpx.HTTPStatusError as exc:
        raise ToolBail(f"HTTP {exc.response.status_code}: {exc.response.text[:200]}") from exc
    except httpx.TimeoutException as exc:
        raise ToolBail(f"Tavily {endpoint} request timed out") from exc
    except httpx.HTTPError as exc:
        raise ToolBail(
            f"HTTP transport error during Tavily {endpoint}: {type(exc).__name__}: {exc}"
        ) from exc
