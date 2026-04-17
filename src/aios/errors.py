"""Aios error hierarchy and FastAPI exception handlers.

Errors are returned to clients in the shape::

    {"error": {"type": "<error_type>", "message": "<human readable>", "detail": {...}}}

The ``type`` is a stable machine-readable string clients can branch on. The
``detail`` is optional and can contain field-level info, ids, etc.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class AiosError(Exception):
    """Base class for all aios-specific errors.

    Subclasses set a class-level ``error_type`` and ``status_code``; instances
    carry a human-readable message and an optional structured detail dict.
    """

    error_type: str = "internal_error"
    status_code: int = 500

    def __init__(
        self,
        message: str,
        *,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or {}

    def to_body(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "error": {
                "type": self.error_type,
                "message": self.message,
            }
        }
        if self.detail:
            body["error"]["detail"] = self.detail
        return body


class NotFoundError(AiosError):
    error_type = "not_found"
    status_code = 404


class ValidationError(AiosError):
    error_type = "validation_error"
    status_code = 422


class ConflictError(AiosError):
    error_type = "conflict"
    status_code = 409


class UnauthorizedError(AiosError):
    error_type = "unauthorized"
    status_code = 401


class ForbiddenError(AiosError):
    error_type = "forbidden"
    status_code = 403


class NoRouteError(AiosError):
    """Raised when a channel address has no matching binding or routing rule.

    Translates to 404 with envelope ``type="no_route"`` so callers (the
    inbound-message endpoint, connectors) can branch on the type.
    """

    error_type = "no_route"
    status_code = 404


class CryptoDecryptError(AiosError):
    """Raised when the CryptoBox cannot decrypt a stored ciphertext.

    Almost always indicates the master key has rotated without re-encrypting
    existing rows, or the row is corrupt.
    """

    error_type = "crypto_decrypt_error"
    status_code = 500


class OAuthRefreshError(AiosError):
    """Raised when refreshing an MCP OAuth access token fails.

    Causes include: the token endpoint returned a non-2xx response, the
    response was missing ``access_token``, the network call timed out, or
    the stored credential is missing required fields (``refresh_token``,
    ``token_endpoint``, ``client_id``).

    The error bubbles up from ``resolve_auth_headers`` so the model sees
    the resulting MCP failure in its next tool result envelope and can
    react. There is deliberately no silent fallback to the stale
    ``access_token`` — that would mask a recoverable failure as a
    permanent 401.
    """

    error_type = "oauth_refresh_error"
    status_code = 502


# ─── FastAPI integration ─────────────────────────────────────────────────────


async def aios_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Render an :class:`AiosError` as a JSON response."""
    assert isinstance(exc, AiosError)
    return JSONResponse(status_code=exc.status_code, content=exc.to_body())


async def http_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Render Starlette/FastAPI HTTPExceptions in our error envelope."""
    assert isinstance(exc, StarletteHTTPException)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": "http_error",
                "message": exc.detail if isinstance(exc.detail, str) else "http error",
            }
        },
    )


async def validation_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Render pydantic/FastAPI request validation errors."""
    assert isinstance(exc, RequestValidationError)
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "type": "validation_error",
                "message": "request body failed validation",
                "detail": {"errors": exc.errors()},
            }
        },
    )


def install_exception_handlers(app: FastAPI) -> None:
    """Wire all aios exception handlers into a FastAPI app."""
    app.add_exception_handler(AiosError, aios_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
