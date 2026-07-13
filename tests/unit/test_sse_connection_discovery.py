"""Durable connection discovery ledger stream tests."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from sse_starlette import ServerSentEvent

from aios.api.sse import connection_discovery_stream
from aios.db.listen import ListenSubscription
from aios.models.connections import Connection


def _sub() -> ListenSubscription:
    return ListenSubscription(queue=asyncio.Queue(), _conn=MagicMock())


def _pool() -> MagicMock:
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=MagicMock())
    cm.__aexit__ = AsyncMock(return_value=None)
    pool = MagicMock()
    pool.acquire.return_value = cm
    return pool


def _connection(cid: str) -> Connection:
    now = datetime(2024, 1, 1, tzinfo=UTC)
    return Connection(
        id=cid,
        connector="matrix",
        external_account_id=f"ghost_{cid}",
        metadata={},
        secrets_set=False,
        created_at=now,
        updated_at=now,
    )


def _data(event: ServerSentEvent) -> dict[str, object]:
    assert isinstance(event.data, str)
    return cast(dict[str, object], json.loads(event.data))


async def test_fresh_orders_cursor_snapshot_sentinel_then_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def snapshot(*args: object, **kwargs: object) -> AsyncGenerator[Connection]:
        yield _connection("con_1")

    monkeypatch.setattr(
        "aios.api.sse.queries.get_connection_change_floor", AsyncMock(return_value=1)
    )
    monkeypatch.setattr(
        "aios.api.sse.queries.get_connection_change_high_water", AsyncMock(return_value=7)
    )
    monkeypatch.setattr("aios.api.sse.connections_service.iter_all_connections", snapshot)
    changes = AsyncMock(
        side_effect=[
            [
                {
                    "seq": 8,
                    "kind": "removed",
                    "connection_id": "con_1",
                    "external_account_id": "ghost_con_1",
                }
            ],
            [],
        ]
    )
    monkeypatch.setattr("aios.api.sse.queries.list_connection_changes", changes)
    gen = connection_discovery_stream(_sub(), _pool(), "matrix", account_id="acct", arm="fresh")
    events = [_data(await gen.__anext__()) for _ in range(4)]
    await gen.aclose()  # type: ignore[attr-defined]
    assert [event["event"] for event in events] == [
        "cursor",
        "added",
        "snapshot_complete",
        "removed",
    ]
    assert events[0]["change_seq"] == 7
    assert events[-1]["change_seq"] == 8


async def test_tail_replays_delta_without_snapshot_and_honours_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = MagicMock()
    monkeypatch.setattr("aios.api.sse.connections_service.iter_all_connections", snapshot)
    monkeypatch.setattr(
        "aios.api.sse.queries.get_connection_change_floor", AsyncMock(return_value=1)
    )
    monkeypatch.setattr(
        "aios.api.sse.queries.get_connection_change_high_water", AsyncMock(return_value=1000)
    )
    monkeypatch.setattr(
        "aios.api.sse.queries.list_connection_changes",
        AsyncMock(
            return_value=[
                {
                    "seq": 1001,
                    "kind": "added",
                    "connection_id": "hidden",
                    "external_account_id": "x",
                },
                {
                    "seq": 1002,
                    "kind": "added",
                    "connection_id": "allowed",
                    "external_account_id": "y",
                },
            ]
        ),
    )
    gen = connection_discovery_stream(
        _sub(),
        _pool(),
        "matrix",
        account_id="acct",
        connection_ids=["allowed"],
        arm="tail",
        after_change_seq=1000,
    )
    assert _data(await gen.__anext__()) == {
        "event": "added",
        "change_seq": 1002,
        "connection_id": "allowed",
        "external_account_id": "y",
    }
    await gen.aclose()  # type: ignore[attr-defined]
    snapshot.assert_not_called()


async def test_tail_below_retention_floor_resets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "aios.api.sse.queries.get_connection_change_floor", AsyncMock(return_value=50)
    )
    monkeypatch.setattr(
        "aios.api.sse.queries.get_connection_change_high_water", AsyncMock(return_value=60)
    )
    gen = connection_discovery_stream(
        _sub(), _pool(), "matrix", account_id="acct", arm="tail", after_change_seq=10
    )
    assert _data(await gen.__anext__()) == {"event": "reset"}
    with pytest.raises(StopAsyncIteration):
        await gen.__anext__()


async def test_v1_emits_snapshot_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    async def empty(*args: object, **kwargs: object) -> AsyncGenerator[Connection]:
        return
        yield

    monkeypatch.setattr(
        "aios.api.sse.queries.get_connection_change_floor", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        "aios.api.sse.queries.get_connection_change_high_water", AsyncMock(return_value=0)
    )
    monkeypatch.setattr("aios.api.sse.connections_service.iter_all_connections", empty)
    gen = connection_discovery_stream(_sub(), _pool(), "matrix", account_id="acct")
    assert _data(await gen.__anext__()) == {"event": "snapshot_complete"}
    await gen.aclose()  # type: ignore[attr-defined]
