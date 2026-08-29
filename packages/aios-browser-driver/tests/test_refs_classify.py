"""Ref resolution's driver-side decisions — the ones made without a round trip.

Bad format and above-watermark both resolve to their wire code without ever
touching the page; the live/stale distinction needs a real page and is covered
by the browser lane (test_walker_browser.py)."""

from __future__ import annotations

from typing import Any

import pytest
from aios_browser_driver.errors import ActionError
from aios_browser_driver.host import PageEntry
from aios_browser_driver.snapshot.refs import resolve_ref


class _ExplodingPage:
    """Any page round trip is a test failure — these paths must decide
    driver-side."""

    async def evaluate_handle(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("resolve_ref must not touch the page for this ref")


def _entry(issued: int) -> PageEntry:
    entry = PageEntry(session_id="sess", pages=[])
    entry.issued = issued
    entry.generation = 3
    return entry


@pytest.mark.parametrize("ref", ["", "e0", "e01", "abc", "12", "e", "e-1"])
async def test_malformed_refs_are_no_such_ref(ref: str) -> None:
    with pytest.raises(ActionError) as info:
        await resolve_ref(_ExplodingPage(), _entry(50), ref)  # type: ignore[arg-type]
    assert info.value.code == "no_such_ref"


async def test_ref_above_the_watermark_is_no_such_ref_without_a_round_trip() -> None:
    with pytest.raises(ActionError) as info:
        await resolve_ref(_ExplodingPage(), _entry(issued=10), "e11")  # type: ignore[arg-type]
    assert info.value.code == "no_such_ref"
    assert "never issued" in info.value.message
