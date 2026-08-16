"""Common pydantic models reused across resource schemas."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from aios.models.pagination import CursorState, Direction, encode_cursor


class ListResponse[T](BaseModel):
    """Standard envelope for paginated list endpoints."""

    data: list[T]
    has_more: bool = False
    # Opaque cursor for the next page, or None when there are no more pages.
    # Resend verbatim as ``?cursor=`` — it encodes position, direction, filters,
    # and page size, so the next request needs no other params.
    next_cursor: str | None = None

    @classmethod
    def paginate(
        cls,
        items: list[T],
        limit: int,
        *,
        cursor: Callable[[T], str | int],
        direction: Direction = "backward",
        filters: dict[str, Any] | None = None,
    ) -> ListResponse[T]:
        """Wrap a windowed query result in the pagination envelope.

        Callers must fetch ``limit + 1`` rows and pass them here. The extra row
        detects whether more pages exist without a separate COUNT query; it is
        stripped before the response is built.

        ``next_cursor`` is minted from the last *kept* row's keyset value
        (``cursor(...)``) combined with ``direction``, ``filters``, and
        ``limit``, so the next ``?cursor=`` request is self-contained. It is
        ``None`` when ``has_more`` is False — there is no further page to point
        at. Order-agnostic: a forward (ASC) page anchors on its largest key, a
        backward (DESC) page on its smallest, both as the last kept row.
        """
        has_more = len(items) > limit
        data = items[:limit]
        next_cursor: str | None = None
        if has_more and data:
            next_cursor = encode_cursor(
                CursorState(
                    cursor=cursor(data[-1]),
                    direction=direction,
                    limit=limit,
                    filters=filters or {},
                )
            )
        return cls(data=data, has_more=has_more, next_cursor=next_cursor)
