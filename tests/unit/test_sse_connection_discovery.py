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
from aios.db.queries.connections import notify_connection_change
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


def _notify(event: str, connection_id: str, account_id: str, external_account_id: str) -> str:
    """Build a NOTIFY payload exactly as ``notify_connection_change`` emits it."""
    return json.dumps(
        {
            "event": event,
            "connection_id": connection_id,
            "account_id": account_id,
            "external_account_id": external_account_id,
        }
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
    await queue.put(_notify("added", "con_1", "acct_X", "ext_1"))
    await queue.put(_notify("added", "con_2", "acct_X", "ext_2"))
    nxt = await gen.__anext__()
    assert _payload(nxt)["connection_id"] == "con_2"

    await gen.aclose()  # type: ignore[attr-defined]


async def test_live_tail_drops_cross_tenant_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tenant gate holds against a crafted ``external_account_id`` that, under
    the old positional ``split("|", 3)`` decode, would have displaced a
    sibling tenant's id into the slot the gate compares.

    The subscriber is scoped to ``acc_VICTIM``; the producer emits an event for
    ``acc_ATTACKER`` with ``external_account_id="acc_VICTIM|x"``. The old
    pipe-encoded string was
    ``"added|con_9|acc_ATTACKER|acc_VICTIM|x"`` — a naive re-split could
    surface ``acc_VICTIM`` in a displaced position. The JSON decode keys the
    gate on the named ``account_id`` (``acc_ATTACKER``), so the event is
    dropped and never leaks to the victim subscriber.
    """
    monkeypatch.setattr(
        "aios.services.connections.queries.list_connections",
        AsyncMock(return_value=[]),
    )
    subscription, queue = _mk_subscription()
    pool = _mk_pool()

    gen = connection_discovery_stream(subscription, pool, "telegram", account_id="acc_VICTIM")

    # Cross-tenant event with a crafted external_account_id; must be dropped.
    await queue.put(_notify("added", "con_9", "acc_ATTACKER", "acc_VICTIM|x"))
    # A legitimate event for the victim follows; only it should surface.
    await queue.put(_notify("added", "con_ok", "acc_VICTIM", "ext_ok"))

    nxt = await gen.__anext__()
    assert _payload(nxt) == {
        "event": "added",
        "connection_id": "con_ok",
        "external_account_id": "ext_ok",
    }

    await gen.aclose()  # type: ignore[attr-defined]


async def test_live_tail_skips_malformed_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A payload that isn't valid JSON, or is missing a required key, is a
    producer-side defect: log-and-skip, then continue serving the tail."""
    monkeypatch.setattr(
        "aios.services.connections.queries.list_connections",
        AsyncMock(return_value=[]),
    )
    subscription, queue = _mk_subscription()
    pool = _mk_pool()

    gen = connection_discovery_stream(subscription, pool, "telegram", account_id="acct_X")

    await queue.put("not json at all")
    await queue.put(json.dumps({"event": "added"}))  # missing keys
    await queue.put(_notify("added", "con_2", "acct_X", "ext_2"))

    nxt = await gen.__anext__()
    assert _payload(nxt)["connection_id"] == "con_2"

    await gen.aclose()  # type: ignore[attr-defined]


def test_notify_payload_round_trips_json() -> None:
    """``notify_connection_change`` emits a JSON object that round-trips back to
    the exact dict, including an ``external_account_id`` with JSON-significant
    and ``|`` characters."""
    captured: dict[str, str] = {}

    class _FakeConn:
        async def execute(self, _sql: str, channel: str, payload: str) -> None:
            captured["channel"] = channel
            captured["payload"] = payload

    tricky = 'acc_VICTIM|x"\\{}'
    asyncio.run(
        notify_connection_change(
            _FakeConn(),
            account_id="acc_ATTACKER",
            connector="telegram",
            connection_id="con_9",
            external_account_id=tricky,
            event="added",
        )
    )

    assert captured["channel"] == "connections_telegram"
    assert json.loads(captured["payload"]) == {
        "event": "added",
        "connection_id": "con_9",
        "account_id": "acc_ATTACKER",
        "external_account_id": tricky,
    }
