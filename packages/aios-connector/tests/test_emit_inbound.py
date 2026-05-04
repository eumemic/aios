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
    connector._initial_state_done.set()

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
    # Timestamp omitted by default (not all platforms supply one).
    assert "timestamp" not in notification.params


async def test_emit_inbound_passes_timestamp_through(connector: _StubConnector) -> None:
    """Optional ``timestamp=`` becomes a top-level param on the notification.

    Per design §3.3: the supervisor stamps it as
    ``metadata.platform_timestamp`` on the appended event so operators
    can compare against the server-side ``created_at``.
    """
    stream = _StubStream()
    connector._write_stream = stream  # type: ignore[assignment]
    connector._client_initialized.set()
    connector._initial_state_done.set()

    iso = "2026-04-30T17:01:23.456+00:00"
    await connector.emit_inbound(
        account="echo-1",
        chat_id="chat-42",
        sender={"display_name": "Alice"},
        content="ping",
        timestamp=iso,
    )

    notification = stream.sent[0].message.root
    assert notification.params["timestamp"] == iso


async def test_emit_inbound_blocks_until_initial_state_done(
    connector: _StubConnector,
) -> None:
    """Live ``emit_inbound`` waits for the accounts snapshot + spool replay.

    Without this gate, a connector whose ``serve()`` fires fast inbounds
    during startup could ship a fresh notification past a half-flushed
    ``_emit_initial_state`` — out of order with the very entries the
    spool replay is supposed to deliver exactly once.
    """
    stream = _StubStream()
    connector._write_stream = stream  # type: ignore[assignment]
    connector._client_initialized.set()
    # _initial_state_done deliberately NOT set — emit_inbound must block.

    completed = anyio.Event()

    async def emit_and_signal() -> None:
        await connector.emit_inbound(
            account="echo-1",
            chat_id="c",
            sender={"display_name": "A"},
            content="m",
        )
        completed.set()

    async with anyio.create_task_group() as tg:
        tg.start_soon(emit_and_signal)
        # Give the task a beat to reach the await.
        await anyio.sleep(0.05)
        assert not completed.is_set(), "emit_inbound should be blocked"
        assert stream.sent == [], "no notification should have been pushed yet"

        # Releasing the gate unblocks the emit.
        connector._initial_state_done.set()
        with anyio.fail_after(1.0):
            await completed.wait()

    assert len(stream.sent) == 1


async def test_run_calls_teardown_when_setup_raises(tmp_path: Path) -> None:
    """A failing ``setup`` still triggers ``teardown`` so resources released.

    Pre-fix, ``setup()`` ran outside the ``try/finally``, so a partial
    setup (e.g., signal-cli's ``__aenter__`` succeeded then
    ``discover_bot_uuid`` raised) leaked the daemon subprocess.
    """
    teardown_called: list[bool] = []

    class _BoomConnector(Connector):
        name = "boom"

        async def setup(self) -> None:
            raise RuntimeError("simulated setup failure")

        async def teardown(self) -> None:
            teardown_called.append(True)

    connector = _BoomConnector(spool_dir=tmp_path)
    with pytest.raises(RuntimeError, match="simulated setup failure"):
        await connector.run()
    assert teardown_called == [True]


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
    connector._initial_state_done.set()

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
