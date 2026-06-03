"""Common pydantic models reused across resource schemas."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel


class ListResponse[T](BaseModel):
    """Standard envelope for paginated list endpoints."""

    data: list[T]
    has_more: bool = False
    next_after: str | None = None

    @classmethod
    def paginate(
        cls,
        items: list[T],
        limit: int,
        *,
        cursor: Callable[[T], str],
    ) -> ListResponse[T]:
        """Wrap a windowed query result in the pagination envelope.

        Callers must fetch ``limit + 1`` rows and pass them here.  The extra
        row is used to detect whether more pages exist without a separate COUNT
        query; it is stripped before the response is built.

        Order-agnostic: the sentinel is always the last row and the cursor is
        taken from the last *kept* row, so the same logic serves forward pages
        (ASC — cursor is the largest seq → next ``?after=``) and backward pages
        (DESC — cursor is the smallest seq → next ``?before=``) alike.
        """
        has_more = len(items) > limit
        data = items[:limit]
        return cls(
            data=data,
            has_more=has_more,
            next_after=cursor(data[-1]) if data else None,
        )


class ErrorBody(BaseModel):
    """The body of every aios error response."""

    type: str
    message: str
    detail: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    """Top-level error envelope: `{"error": {...}}`."""

    error: ErrorBody
