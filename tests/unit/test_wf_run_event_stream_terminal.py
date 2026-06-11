"""Unit test for ``wf_run_event_stream``'s terminal-status catch-up.

The hazard: a run can go terminal in the window between the stream's backfill
SELECT (whose MVCC snapshot predates the completion) and the subsequent status
read. The ``run_completed`` event then committed AND queued its NOTIFY, but the
backfill never saw it. The generator must NOT emit a bare ``done`` and return —
it would drop the terminal frame (carrying ``output``/``error``) that is sitting
unread in the queue. Because a terminal run's journal is frozen, a final
catch-up read past the cursor recovers it deterministically.

This forces that exact interleaving with a mock connection (an integration test
can't, without injecting a delay between the two real snapshots — which is why
the live-DB stream test exercises only the live-tail terminal path).
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

from aios.api.sse import wf_run_event_stream
from aios.db.listen import ListenSubscription


def _mk_subscription() -> ListenSubscription:
    conn = MagicMock()
    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=8)
    return ListenSubscription(queue=queue, _conn=conn)


def _mk_pool(conn: MagicMock) -> MagicMock:
    """Pool whose ``async with pool.acquire()`` always yields ``conn``."""
    acquire_cm = MagicMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=None)
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=acquire_cm)
    return pool


async def test_terminal_status_drains_catch_up_tail_then_done() -> None:
    """Empty backfill + status='completed' + a run_completed in the catch-up read
    ⇒ the stream emits the run_completed event frame, THEN done (not bare done)."""
    run_completed_row = {
        "id": "wfe_1",
        "run_id": "wfr_X",
        "seq": 5,
        "type": "run_completed",
        "call_key": None,
        "payload": json.dumps({"output": 2, "is_error": False}),
        "created_at": datetime(2024, 1, 1, tzinfo=UTC),
    }
    conn = MagicMock()
    # 1st fetch = backfill (race: snapshot predates completion → empty);
    # 2nd fetch = catch-up after the terminal status read → the run_completed.
    conn.fetch = AsyncMock(side_effect=[[], [run_completed_row]])
    conn.fetchval = AsyncMock(return_value="completed")
    pool = _mk_pool(conn)
    subscription = _mk_subscription()

    collected = [
        (msg.event, json.loads(str(msg.data)).get("type") if msg.event == "event" else None)
        async for msg in wf_run_event_stream(subscription, pool, "wfr_X", after_seq=0)
    ]

    assert collected == [("event", "run_completed"), ("done", None)]
    # The catch-up read was issued (not skipped) and the subscription cleaned up.
    assert conn.fetch.await_count == 2
    subscription._conn.terminate.assert_called_once()


async def test_annotation_event_surfaces_in_the_stream() -> None:
    """A ``log()``/``phase()`` annotation row streams as a normal event frame — the
    serializer is type-agnostic — in seq order ahead of the ``run_completed`` terminal.
    This is the SSE half of the journaled-progress contract."""
    annotation_row = {
        "id": "wfe_a",
        "run_id": "wfr_X",
        "seq": 1,
        "type": "annotation",
        "call_key": "sha:ann#0",
        "payload": json.dumps({"kind": "phase", "text": "build"}),
        "created_at": datetime(2024, 1, 1, tzinfo=UTC),
    }
    run_completed_row = {
        "id": "wfe_z",
        "run_id": "wfr_X",
        "seq": 2,
        "type": "run_completed",
        "call_key": None,
        "payload": json.dumps({"output": None, "is_error": False}),
        "created_at": datetime(2024, 1, 1, tzinfo=UTC),
    }
    conn = MagicMock()
    # Backfill carries both rows; the loop yields the annotation, then closes on the
    # run_completed (no catch-up fetch needed).
    conn.fetch = AsyncMock(side_effect=[[annotation_row, run_completed_row]])
    pool = _mk_pool(conn)
    subscription = _mk_subscription()

    messages = [msg async for msg in wf_run_event_stream(subscription, pool, "wfr_X", after_seq=0)]
    events = [json.loads(str(msg.data)) for msg in messages if msg.event == "event"]

    assert [msg.event for msg in messages] == ["event", "event", "done"]
    assert events[0]["type"] == "annotation"
    assert events[0]["payload"] == {"kind": "phase", "text": "build"}
    assert events[1]["type"] == "run_completed"
    subscription._conn.terminate.assert_called_once()


async def test_terminal_status_empty_catch_up_falls_through_to_done() -> None:
    """The genuine reconnect-past-terminal case: empty backfill + status terminal +
    nothing past the cursor ⇒ a single done frame (no hang, no spurious event)."""
    conn = MagicMock()
    conn.fetch = AsyncMock(side_effect=[[], []])  # backfill empty, catch-up empty
    conn.fetchval = AsyncMock(return_value="errored")
    pool = _mk_pool(conn)
    subscription = _mk_subscription()

    collected = [
        (msg.event, msg.data)
        async for msg in wf_run_event_stream(subscription, pool, "wfr_X", after_seq=99)
    ]

    assert collected == [("done", "{}")]
    subscription._conn.terminate.assert_called_once()
