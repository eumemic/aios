"""Pure-ASGI request-logging middleware.

Emits exactly one ``api.request`` structlog line per http request with the
method, path, response status, and wall-clock duration. Implemented as a raw
ASGI middleware (not ``@app.middleware("http")``) so it runs in the same task
context as the route handler — an ``account_id`` bound by the auth dependency
via ``bind_contextvars`` is therefore visible on the request line through
``merge_contextvars`` (already in the processor chain). It is registered as the
OUTERMOST middleware so its duration covers the whole stack, including the
exception handlers.
"""

from __future__ import annotations

import time

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from aios.api._log_redaction import redact_sensitive_path
from aios.logging import get_logger

log = get_logger(__name__)


class RequestLoggingMiddleware:
    """One ``api.request`` line per http request (method, path, status, duration_ms)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # Only http requests are logged; websocket/lifespan pass straight through.
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope["method"]
        path = scope["path"]
        start = time.perf_counter()
        status: int | None = None

        async def send_wrapper(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            # Always emit the line, even if the downstream raised before any
            # response-start was sent — fall back to 500 when status is unknown.
            log.info(
                "api.request",
                method=method,
                path=redact_sensitive_path(path),
                status=status if status is not None else 500,
                duration_ms=round((time.perf_counter() - start) * 1000, 2),
            )
