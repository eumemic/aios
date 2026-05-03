"""``emit_inbound`` flow: spool persistence + write-stream push.

Stubs the write stream to capture the JSON-RPC notification, and uses
a real SQLite spool in a tmpdir.  End-to-end with a live ``ClientSession``
is covered by the e2e suite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anyio
import pytest
from aios_connector import Connector


class _StubConnector(Connector):
    name = "stub"


class _StubStream:
    """Captures sent ``SessionMessage``s in a list for assertion."""

    def __init__(self) -> None:
        self.sent: list[Any] = []

    async def send(self, message: Any) -> None:
        self.sent.append(message)


@pytest.fixture
def connector(tmp_path: Path) -> _StubConnector:
    return _StubConnector(spool_dir=tmp_path)


async def test_emit_inbound_persists_then_pushes(connector: _StubConnector) -> None:
    stream = _StubStream()
    connector._write_stream = stream  # type: ignore[assignment]
    connector._client_initialized.set()

    event_id = await connector.emit_inbound(
        account="echo-1",
        chat_id="chat-42",
        sender={"display_name": "Alice"},
        content="hello",
    )

    assert connector._spool.unacked() == [(event_id, _payload_with_id(connector, event_id))]
    assert len(stream.sent) == 1
    notification = stream.sent[0].message.root
    assert notification.method == "notifications/aios/inbound"
    assert notification.params["event_id"] == event_id
    assert notification.params["account"] == "echo-1"
    assert notification.params["chat_id"] == "chat-42"
    assert notification.params["content"] == "hello"


async def test_emit_inbound_before_run_raises(connector: _StubConnector) -> None:
    """Calling ``emit_inbound`` before :meth:`run` started is a programmer error.

    The check is correct-by-construction: connector authors who try to
    emit from ``__init__`` see the failure immediately rather than
    dropping the message silently.
    """
    with pytest.raises(RuntimeError, match="emit_inbound called before"):
        await connector.emit_inbound(
            account="x",
            chat_id="y",
            sender={"display_name": "z"},
            content="content",
        )


async def test_unacked_replays_after_init(connector: _StubConnector) -> None:
    """Pre-init emission writes spool but doesn't push — replay does the push.

    This is the worker-restart-mid-spool path: the connector starts,
    has unacked entries from a prior run, the SDK replays them on the
    new process AFTER ``notifications/initialized`` arrives.
    """
    # Simulate a prior run that wrote two entries before crash.
    stream_pre = _StubStream()
    connector._write_stream = stream_pre  # type: ignore[assignment]
    connector._client_initialized.set()

    eid_a = await connector.emit_inbound(
        account="echo-1", chat_id="c1", sender={"display_name": "A"}, content="m1"
    )
    eid_b = await connector.emit_inbound(
        account="echo-1", chat_id="c1", sender={"display_name": "A"}, content="m2"
    )

    # Now reset client_initialized (simulating new subprocess) and call replay.
    stream_post = _StubStream()
    connector._write_stream = stream_post  # type: ignore[assignment]
    connector._client_initialized = anyio.Event()

    # Trigger _emit_initial_state — it would normally wait on the event,
    # so we set it on a sibling task and let the coroutine drain.
    async with anyio.create_task_group() as tg:
        tg.start_soon(connector._emit_initial_state)
        connector._client_initialized.set()

    # First message is the accounts snapshot, then two replayed inbounds.
    methods = [m.message.root.method for m in stream_post.sent]
    assert methods == [
        "notifications/aios/accounts",
        "notifications/aios/inbound",
        "notifications/aios/inbound",
    ]
    replayed_ids = [m.message.root.params["event_id"] for m in stream_post.sent[1:]]
    assert replayed_ids == [eid_a, eid_b]


def _payload_with_id(connector: Connector, event_id: str) -> bytes:
    """Reconstruct the spool payload bytes for a given event id (test helper)."""
    for stored_id, payload in connector._spool.unacked():
        if stored_id == event_id:
            return payload
    raise AssertionError(f"no spool entry for {event_id}")
