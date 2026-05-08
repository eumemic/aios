"""Thin HTTP client for the aios connector-facing endpoints (#301).

A connector container talks to aios via three endpoints:

* ``POST /v1/connectors/inbound`` — submit an inbound user message.
* ``GET /v1/connectors/calls`` (SSE) — stream pending custom tool calls.
* ``POST /v1/sessions/:id/tool-results`` — submit a tool result.

The bearer token resolves server-side to a single ``connection_id``;
clients don't pick which connection their requests act on.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx


class AiosClient:
    """Minimal aios HTTP client for connector containers."""

    def __init__(self, base_url: str, token: str, *, timeout: float = 30.0) -> None:
        self._base_url = base_url
        self._headers = {"Authorization": f"Bearer {token}"}
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> AiosClient:
        self._client = httpx.AsyncClient(
            base_url=self._base_url, headers=self._headers, timeout=self._timeout
        )
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("AiosClient must be used as an async context manager")
        return self._client

    async def post_inbound(
        self,
        *,
        event_id: str,
        chat_id: str,
        sender: dict[str, Any],
        content: str,
        attachments: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        """POST a user message; idempotent on ``event_id``.  Raises on 4xx/5xx."""
        body: dict[str, Any] = {
            "event_id": event_id,
            "chat_id": chat_id,
            "sender": sender,
            "content": content,
        }
        if attachments is not None:
            body["attachments"] = attachments
        if metadata is not None:
            body["metadata"] = metadata
        if timestamp is not None:
            body["timestamp"] = timestamp
        r = await self._http.post("/v1/connectors/inbound", json=body)
        r.raise_for_status()
        return dict(r.json())

    async def post_tool_result(
        self,
        *,
        session_id: str,
        tool_call_id: str,
        content: str | list[dict[str, Any]],
        is_error: bool = False,
    ) -> None:
        """POST a custom tool result.  Resumes the session.

        Uses the connector-scoped ``/v1/connectors/tool-results``
        endpoint (not the operator-scoped session route) so the
        connector's bearer token authorizes — the route validates that
        the session is bound to the caller's connection.
        """
        body: dict[str, Any] = {
            "session_id": session_id,
            "tool_call_id": tool_call_id,
            "content": content,
            "is_error": is_error,
        }
        r = await self._http.post("/v1/connectors/tool-results", json=body)
        r.raise_for_status()

    async def whoami(self) -> str:
        """Resolve the bearer token to its ``connection_id``."""
        r = await self._http.get("/v1/connector-tokens/whoami")
        r.raise_for_status()
        connection_id: str = r.json()["connection_id"]
        return connection_id

    async def get_secrets(self) -> dict[str, str]:
        """Fetch decrypted platform secrets for the caller's connection.

        Operators set these via ``POST /v1/connections`` or
        ``PUT /v1/connections/{id}/secrets``; this is the only path that
        returns the decrypted values, scoped to the bearer token's
        connection.  Returns an empty dict when no secrets are
        configured — most connector authors should fail loudly in that
        case (they're missing the bot_token / phone / etc. they need to
        do their job).
        """
        r = await self._http.get("/v1/connectors/secrets")
        r.raise_for_status()
        secrets: dict[str, str] = r.json()["secrets"]
        return secrets

    async def set_connection_tools(self, tools: list[dict[str, Any]]) -> None:
        """Replace the connection's tool schemas wholesale.

        The connector container is the source of truth for what tools
        it serves; the SDK derives JSON Schemas from ``@tool``-decorated
        methods and publishes them at startup so the model's tool list
        always matches what the connector is actually willing to
        execute.  Operators don't hand-write ``tools.json``.
        """
        r = await self._http.put("/v1/connectors/tools", json={"tools": tools})
        r.raise_for_status()

    async def stream_calls(self) -> AsyncIterator[dict[str, Any]]:
        """Yield each ``event: call`` payload from the SSE calls stream.

        The stream is long-lived; the iterator runs until the underlying
        HTTP connection drops.  Callers wrap it in retry-with-backoff
        logic so transient network blips don't kill the connector.
        """
        async with self._http.stream("GET", "/v1/connectors/calls") as r:
            r.raise_for_status()
            buf = ""
            async for chunk in r.aiter_bytes():
                # Normalise CRLF → LF so frame parsing works the same
                # whether we're talking to a real HTTP server (CRLF, the
                # SSE spec form) or an ASGI test transport (LF).
                buf += chunk.decode("utf-8").replace("\r\n", "\n")
                while "\n\n" in buf:
                    frame, buf = buf.split("\n\n", 1)
                    parsed = _parse_sse_frame(frame)
                    if parsed is not None and parsed[0] == "call":
                        yield parsed[1]


def _parse_sse_frame(frame: str) -> tuple[str, dict[str, Any]] | None:
    """Parse one SSE frame; return ``(event_name, data_dict)`` or ``None``.

    Skips comment-only frames (sse-starlette pings) and frames missing
    an ``event:`` line.
    """
    event_name: str | None = None
    data_lines: list[str] = []
    for line in frame.splitlines():
        if line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if event_name is None or not data_lines:
        return None
    try:
        data = json.loads("\n".join(data_lines))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return event_name, data
