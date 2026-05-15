"""Common pydantic models reused across resource schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ListResponse[T](BaseModel):
    """Standard envelope for paginated list endpoints."""

    data: list[T]
    has_more: bool = False
    next_after: str | None = None


class ErrorBody(BaseModel):
    """The body of every aios error response."""

    type: str
    message: str
    detail: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    """Top-level error envelope: `{"error": {...}}`."""

    error: ErrorBody
