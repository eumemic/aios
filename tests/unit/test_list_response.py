"""Tests for ``ListResponse.paginate`` — the cursor-pagination envelope helper."""

from __future__ import annotations

from pydantic import BaseModel

from aios.models.common import ListResponse
from aios.models.pagination import decode_cursor


class _Item(BaseModel):
    id: str
    seq: int


def test_paginate_empty() -> None:
    """Empty items: no more pages, so ``next_cursor`` is None."""
    resp = ListResponse[_Item].paginate([], limit=10, cursor=lambda x: x.id)
    assert resp.data == []
    assert resp.has_more is False
    assert resp.next_cursor is None


def test_paginate_partial_page_has_no_cursor() -> None:
    """Result smaller than ``limit``: last page, so ``next_cursor`` is None."""
    items = [_Item(id="a", seq=1), _Item(id="b", seq=2)]
    resp = ListResponse[_Item].paginate(items, limit=10, cursor=lambda x: x.id)
    assert resp.data == items
    assert resp.has_more is False
    assert resp.next_cursor is None


def test_paginate_exact_boundary_has_no_cursor() -> None:
    """Exactly ``limit`` items (caller fetched limit+1, got limit): no next page."""
    items = [_Item(id="a", seq=1), _Item(id="b", seq=2)]
    resp = ListResponse[_Item].paginate(items, limit=2, cursor=lambda x: x.id)
    assert resp.data == items
    assert resp.has_more is False
    assert resp.next_cursor is None


def test_paginate_has_more_mints_cursor() -> None:
    """Caller fetched ``limit+1`` and got it back: ``has_more`` True, data trimmed,
    and ``next_cursor`` is an opaque token anchored on the last kept row."""
    items = [_Item(id="a", seq=1), _Item(id="b", seq=2), _Item(id="c", seq=3)]
    resp = ListResponse[_Item].paginate(
        items, limit=2, cursor=lambda x: x.id, direction="backward", filters={"name": "x"}
    )
    assert resp.data == [_Item(id="a", seq=1), _Item(id="b", seq=2)]
    assert resp.has_more is True
    assert isinstance(resp.next_cursor, str) and resp.next_cursor

    state = decode_cursor(resp.next_cursor)
    assert state.cursor == "b"  # last kept row
    assert state.direction == "backward"
    assert state.limit == 2
    assert state.filters == {"name": "x"}


def test_paginate_preserves_int_cursor_in_token() -> None:
    """An int keyset (version/seq) round-trips through the token as an int."""
    items = [_Item(id="a", seq=10), _Item(id="b", seq=20), _Item(id="c", seq=30)]
    resp = ListResponse[_Item].paginate(items, limit=2, cursor=lambda x: x.seq, direction="forward")
    assert resp.next_cursor is not None
    state = decode_cursor(resp.next_cursor)
    assert state.cursor == 20 and isinstance(state.cursor, int)
    assert state.direction == "forward"
