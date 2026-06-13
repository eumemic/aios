"""Unit tests for ``connection_discovery_stream`` backfill (issue #814).

After migrating the hand-rolled keyset loop onto
``connections_service.iter_all_connections``, the backfill must still:

* emit a byte-identical ``{event:'added', connection_id, external_account_id}``
  ``ServerSentEvent`` payload per active connection (the downstream runtime
  discovery loop parses ``external_account_id``);
* honour the ``connection_ids`` allowlist (out-of-scope ids skipped);
* dedup so the live tail doesn't re-emit an already-backfilled ``added``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from sse_starlette import ServerSentEvent

from aios.api.sse import connection_discovery_stream
from aios.db.listen import ListenSubscription
from aios.models.connections import Connection


def _mk_subscription() -> tuple[ListenSubscription, asyncio.Queue[str]]:
    conn = MagicMock()
    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=8)
    return ListenSubscription(queue=queue, _conn=conn), queue


def _mk_pool() -> MagicMock:
    conn = MagicMock()
    acquire_cm = MagicMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=None)
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=acquire_cm)
    return pool


def _mk_connection(cid: str, external: str) -> Connection:
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


def _payload(evt: ServerSentEvent) -> dict[str, str]:
    """Parse a ``ServerSentEvent``'s JSON ``data`` (typed ``Any | None``)."""
    assert isinstance(evt.data, str)
    parsed: dict[str, str] = json.loads(evt.data)
    return parsed


async def _drain_backfill(
    gen: AsyncIterator[ServerSentEvent], expected_count: int
) -> list[ServerSentEvent]:
    """Pull exactly ``expected_count`` backfill events, then stop (the live
    tail would block on an empty queue)."""
    events: list[ServerSentEvent] = []
    for _ in range(expected_count):
        events.append(await gen.__anext__())
    await gen.aclose()  # type: ignore[attr-defined]
    return events


async def test_backfill_payload_is_byte_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "aios.services.connections.queries.list_connections",
        AsyncMock(return_value=[_mk_connection("con_1", "ext_1")]),
    )
    subscription, _ = _mk_subscription()
    pool = _mk_pool()

    gen = connection_discovery_stream(subscription, pool, "telegram", account_id="acct_X")
    events = await _drain_backfill(gen, 1)

    assert len(events) == 1
    evt = events[0]
    assert evt.event == "connection"
    assert _payload(evt) == {
        "event": "added",
        "connection_id": "con_1",
        "external_account_id": "ext_1",
    }


async def test_backfill_honours_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "aios.services.connections.queries.list_connections",
        AsyncMock(
            return_value=[
                _mk_connection("con_1", "ext_1"),
                _mk_connection("con_2", "ext_2"),
                _mk_connection("con_3", "ext_3"),
            ]
        ),
    )
    subscription, _ = _mk_subscription()
    pool = _mk_pool()

    gen = connection_discovery_stream(
        subscription,
        pool,
        "telegram",
        account_id="acct_X",
        connection_ids=["con_1", "con_3"],
    )
    events = await _drain_backfill(gen, 2)

    ids = [_payload(e)["connection_id"] for e in events]
    assert ids == ["con_1", "con_3"]


async def test_live_tail_dedups_already_backfilled_added(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "aios.services.connections.queries.list_connections",
        AsyncMock(return_value=[_mk_connection("con_1", "ext_1")]),
    )
    subscription, queue = _mk_subscription()
    pool = _mk_pool()

    gen = connection_discovery_stream(subscription, pool, "telegram", account_id="acct_X")

    # First event is the backfilled "added".
    first = await gen.__anext__()
    assert _payload(first)["connection_id"] == "con_1"

    # A duplicate "added" arriving on the live tail must be suppressed; only the
    # subsequent distinct event is emitted.
    await queue.put("added|con_1|acct_X|ext_1")
    await queue.put("added|con_2|acct_X|ext_2")
    nxt = await gen.__anext__()
    assert _payload(nxt)["connection_id"] == "con_2"

    await gen.aclose()  # type: ignore[attr-defined]
