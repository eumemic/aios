"""Unit tests for the connections service layer (issue #814).

Focus: :func:`aios.services.connections.iter_all_connections`, the
correct-by-construction "enumerate every connection on an account"
helper that internally keyset-paginates so callers never re-implement
the loop (and never silently truncate at the bounded ``list_connections``
default of 50).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aios.models.connections import Connection
from aios.services import connections as connections_service


def _mk_connection(cid: str, external: str = "ext") -> Connection:
    """Build a minimal :class:`Connection` with just the fields the helper
    reads (``id``, threaded through; ``external_account_id`` for parity)."""
    now = datetime(2024, 1, 1, tzinfo=UTC)
    return Connection(
        id=cid,
        connector="telegram",
        external_account_id=external,
        metadata={},
        secrets_set=False,
        created_at=now,
        updated_at=now,
    )


def _mk_pool() -> MagicMock:
    """Mock pool whose ``async with pool.acquire()`` yields a fresh MagicMock conn.

    Records every ``acquire()`` invocation so tests can assert one acquire
    per page (per-page release contract).
    """
    conn = MagicMock()
    acquire_cm = MagicMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=None)
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=acquire_cm)
    return pool


def _patch_list_connections(
    monkeypatch: pytest.MonkeyPatch, pages: list[list[Connection]]
) -> AsyncMock:
    """Patch ``aios.services.connections.queries.list_connections`` to return
    ``pages`` in order. Returns the AsyncMock for call-arg assertions."""
    mock = AsyncMock(side_effect=pages)
    monkeypatch.setattr("aios.services.connections.queries.list_connections", mock)
    return mock


async def _collect(pool: MagicMock, **kwargs: Any) -> list[Connection]:
    return [c async for c in connections_service.iter_all_connections(pool, **kwargs)]


async def test_yields_across_page_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    """A full page followed by a partial page yields every row and advances
    the cursor to ``page[-1].id`` between fetches."""
    monkeypatch.setattr(connections_service, "_CONNECTION_PAGE_SIZE", 2)
    page1 = [_mk_connection("c3"), _mk_connection("c2")]
    page2 = [_mk_connection("c1")]
    mock = _patch_list_connections(monkeypatch, [page1, page2])

    pool = _mk_pool()
    result = await _collect(pool, account_id="acct_X")

    assert [c.id for c in result] == ["c3", "c2", "c1"]
    assert mock.await_count == 2
    # First fetch: no cursor. Second fetch: cursor == last id of page1.
    first_kwargs = mock.await_args_list[0].kwargs
    second_kwargs = mock.await_args_list[1].kwargs
    assert first_kwargs["after"] is None
    assert second_kwargs["after"] == "c2"
    assert first_kwargs["limit"] == 2
    assert first_kwargs["account_id"] == "acct_X"


async def test_exact_boundary_issues_second_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the total equals the page size, a second (empty) fetch is issued
    and the iteration stops on the empty page."""
    monkeypatch.setattr(connections_service, "_CONNECTION_PAGE_SIZE", 2)
    page1 = [_mk_connection("c2"), _mk_connection("c1")]
    page2: list[Connection] = []
    mock = _patch_list_connections(monkeypatch, [page1, page2])

    pool = _mk_pool()
    result = await _collect(pool, account_id="acct_X")

    assert [c.id for c in result] == ["c2", "c1"]
    assert mock.await_count == 2


async def test_single_short_page_one_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    """A page shorter than the page size does exactly one fetch."""
    monkeypatch.setattr(connections_service, "_CONNECTION_PAGE_SIZE", 5)
    page1 = [_mk_connection("c2"), _mk_connection("c1")]
    mock = _patch_list_connections(monkeypatch, [page1])

    pool = _mk_pool()
    result = await _collect(pool, account_id="acct_X")

    assert [c.id for c in result] == ["c2", "c1"]
    assert mock.await_count == 1


async def test_empty_account_one_fetch_yields_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(connections_service, "_CONNECTION_PAGE_SIZE", 5)
    mock = _patch_list_connections(monkeypatch, [[]])

    pool = _mk_pool()
    result = await _collect(pool, account_id="acct_X")

    assert result == []
    assert mock.await_count == 1


async def test_connector_filter_threaded_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(connections_service, "_CONNECTION_PAGE_SIZE", 5)
    mock = _patch_list_connections(monkeypatch, [[]])

    pool = _mk_pool()
    await _collect(pool, account_id="acct_X", connector="telegram")

    assert mock.await_args_list[0].kwargs["connector"] == "telegram"


async def test_one_acquire_per_page(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each page is fetched under its own ``pool.acquire()``, released before
    the next — assert acquire is called once per page, not once total."""
    monkeypatch.setattr(connections_service, "_CONNECTION_PAGE_SIZE", 2)
    page1 = [_mk_connection("c4"), _mk_connection("c3")]
    page2 = [_mk_connection("c2"), _mk_connection("c1")]
    page3: list[Connection] = []
    _patch_list_connections(monkeypatch, [page1, page2, page3])

    pool = _mk_pool()
    result = await _collect(pool, account_id="acct_X")

    assert [c.id for c in result] == ["c4", "c3", "c2", "c1"]
    # Three pages fetched → three acquires.
    assert pool.acquire.call_count == 3


async def test_no_limit_parameter_exposed() -> None:
    """The helper exposes no ``limit`` parameter — correct-by-construction."""
    import inspect

    sig = inspect.signature(connections_service.iter_all_connections)
    assert "limit" not in sig.parameters
