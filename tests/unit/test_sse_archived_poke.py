"""Unit tests: the SSE ``/stream`` generator ends cleanly on an archive poke.

``archive_session`` fires a content-free ``EVENTS_ARCHIVED_NOTIFY`` sentinel on
``events_<session_id>`` after commit (issue #906). The sentinel is neither an
``evt_`` id nor a ``{"delta": …}`` payload, so the SSE tail must recognize it
and terminate the stream (an archived session is permanently gone — equivalent
to the ``lifecycle: terminated`` terminal) rather than log a spurious
``sse.event_not_found`` warning while trying to fetch a non-existent row.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from sse_starlette import ServerSentEvent

from aios.api.sse import sse_event_stream
from aios.db.listen import EVENTS_ARCHIVED_NOTIFY, ListenSubscription


def _mk_subscription(queue: asyncio.Queue[str]) -> ListenSubscription:
    return ListenSubscription(queue=queue, _conn=MagicMock())


def _mk_pool_empty_backfill() -> MagicMock:
    """Pool whose backfill ``conn.fetch`` returns no rows and whose
    per-notify ``conn.fetchrow`` returns None (the archived session has no
    new event row — only the sentinel poke)."""
    conn = MagicMock()
    conn.fetch = AsyncMock(return_value=[])
    conn.fetchrow = AsyncMock(return_value=None)
    acquire_cm = MagicMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=None)
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=acquire_cm)
    return pool


async def _drain(gen: Any) -> list[ServerSentEvent]:
    return [item async for item in gen]


async def test_sse_terminates_on_archive_poke() -> None:
    queue: asyncio.Queue[str] = asyncio.Queue()
    queue.put_nowait(EVENTS_ARCHIVED_NOTIFY)
    pool = _mk_pool_empty_backfill()
    sub = _mk_subscription(queue)

    out = await _drain(sse_event_stream(sub, pool, "sess_x", after_seq=0))

    # Exactly one item: the terminal ``done`` SSE event.
    assert len(out) == 1
    assert out[0].event == "done"
    # The archive poke must NOT have triggered a per-notify row fetch
    # (which would have logged sse.event_not_found on the None result).
    pool.acquire.return_value.__aenter__.return_value.fetchrow.assert_not_called()


async def test_sse_archive_poke_is_distinct_from_delta() -> None:
    """A real delta payload (``{"delta": …}``) still streams as a delta and
    does NOT terminate; only the archive sentinel terminates."""
    queue: asyncio.Queue[str] = asyncio.Queue()
    queue.put_nowait('{"delta": "hi"}')
    queue.put_nowait(EVENTS_ARCHIVED_NOTIFY)
    pool = _mk_pool_empty_backfill()
    sub = _mk_subscription(queue)

    out = await _drain(sse_event_stream(sub, pool, "sess_x", after_seq=0))

    assert [item.event for item in out] == ["delta", "done"]
