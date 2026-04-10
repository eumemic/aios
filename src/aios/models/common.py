"""Common pydantic models reused across resource schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Pagination(BaseModel):
    """Standard cursor-paginated query parameters.

    `after` is the id of the last item from the previous page (for keyset
    pagination); `limit` caps page size.
    """

    model_config = ConfigDict(extra="forbid")

    limit: int = Field(default=50, ge=1, le=200)
    after: str | None = None


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
