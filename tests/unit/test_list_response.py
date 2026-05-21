"""Tests for ``ListResponse.paginate`` — the cursor-pagination envelope helper."""

from __future__ import annotations

from pydantic import BaseModel

from aios.models.common import ListResponse


class _Item(BaseModel):
    id: str
    seq: int


def test_paginate_empty() -> None:
    """Empty items: ``next_after`` is None and ``has_more`` is False."""
    resp = ListResponse[_Item].paginate([], limit=10, cursor=lambda x: x.id)
    assert resp.data == []
    assert resp.has_more is False
    assert resp.next_after is None


def test_paginate_partial_page() -> None:
    """Result smaller than ``limit``: ``has_more`` is False, cursor on last item."""
    items = [_Item(id="a", seq=1), _Item(id="b", seq=2)]
    resp = ListResponse[_Item].paginate(items, limit=10, cursor=lambda x: x.id)
    assert resp.data == items
    assert resp.has_more is False
    assert resp.next_after == "b"


def test_paginate_full_page_no_more() -> None:
    """Exactly ``limit`` items returned (caller fetched limit+1, got limit): ``has_more`` False."""
    items = [_Item(id="a", seq=1), _Item(id="b", seq=2)]
    resp = ListResponse[_Item].paginate(items, limit=2, cursor=lambda x: x.id)
    assert resp.data == items
    assert resp.has_more is False
    assert resp.next_after == "b"


def test_paginate_has_more() -> None:
    """Caller fetched ``limit+1`` rows and got ``limit+1`` back: ``has_more`` True, data trimmed."""
    items = [_Item(id="a", seq=1), _Item(id="b", seq=2), _Item(id="c", seq=3)]
    resp = ListResponse[_Item].paginate(items, limit=2, cursor=lambda x: x.id)
    assert resp.data == [_Item(id="a", seq=1), _Item(id="b", seq=2)]
    assert resp.has_more is True
    assert resp.next_after == "b"
