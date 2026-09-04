"""Regression coverage for the graceful-shutdown inbound drain.

``TelegramConnector.serve_connection`` races a PTB polling loop with a
``_drain_queue`` task inside a ``TaskGroup``.  On graceful cancellation
(SIGTERM/SIGINT from a deploy/config-reload/supervisor-restart) the
``finally`` block must drain ``inbound_queue`` to completion *before*
the connector exits — otherwise updates already confirmed to (deleted
by) Telegram but still queued locally are silently lost on restart:
Telegram does not redeliver them and the ``connector_inbound_acks``
ledger has no row to dedup (it only covers the *opposite*,
emit-before-confirm, race).

These tests pin the load-bearing pieces of that shutdown sequence:

* :meth:`_shutdown_application` stops polling, processes every fetched
  PTB update, then drains ``inbound_queue`` to completion — in that
  order — before the final application shutdown.
* A per-item emit failure during the drain is logged and skipped, so
  one bad item cannot abandon the rest of the queue.
* :meth:`_drain_queue` re-enqueues the in-flight item on cancellation so
  the shutdown drain can still emit it (the aios dedup ledger absorbs a
  double-emit if the original POST already committed server-side).
* :meth:`_drain_queue` cancelled while idle (no item dequeued) does not
  spuriously re-enqueue.
* ``serve_connection`` end-to-end: a graceful cancellation drains every
  buffered item before the connector exits — the headline regression.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram import Bot, Message

from aios_telegram.connector import TelegramConnector, _TelegramConnectionState
from aios_telegram.parse import parse_message
from tests.conftest import BOT_ID, CONNECTION_ID

# ── helpers ─────────────────────────────────────────────────────────────────


def _msg(ptb_bot: Bot, *, message_id: int) -> Message:
    """A DM text message with a distinct ``message_id`` and a non-bot
    sender so ``parse_message`` keeps it."""
    data: dict[str, Any] = {
        "message_id": message_id,
        "date": 1700000030,
        "chat": {"id": 123456789, "type": "private"},
        "from": {"id": 123456789, "is_bot": False, "first_name": "Alice"},
        "text": f"hello {message_id}",
    }
    parsed = Message.de_json(data, ptb_bot)
    assert parsed is not None
    return parsed


def _inbound(ptb_bot: Bot, *, message_id: int) -> Any:
    msg = parse_message(_msg(ptb_bot, message_id=message_id), bot_id=BOT_ID)
    assert msg is not None
    return msg


def _event_id(message_id: int) -> str:
    return f"telegram-123456789-{message_id}"


def _make_state() -> tuple[_TelegramConnectionState, MagicMock, MagicMock]:
    """Build a ``_TelegramConnectionState`` whose PTB ``Application`` is a
    fully-mocked no-network double: ``initialize``/``start``/``stop``/
    ``shutdown`` and ``updater.start_polling``/``updater.stop`` are
    :class:`AsyncMock`\\s so ``_run_polling`` and the shutdown sequence run
    without touching Telegram.  Mirrors the e2e telegram harness's
    ``mocked_telegram_application`` shape."""
    application = MagicMock()
    application.bot = MagicMock()
    application.initialize = AsyncMock()
    application.start = AsyncMock()
    application.stop = AsyncMock()
    application.shutdown = AsyncMock()
    application.add_handler = MagicMock()
    application.add_error_handler = MagicMock()
    updater = MagicMock()
    updater.start_polling = AsyncMock()
    updater.stop = AsyncMock()
    application.updater = updater

    state = _TelegramConnectionState(
        application=application,
        bot_id=BOT_ID,
        first_name="TestBot",
        username="testbot",
        inbound_queue=asyncio.Queue(),
    )
    return state, application, updater


def _make_connector() -> TelegramConnector:
    c = TelegramConnector()
    return c


def _patch_emit(connector: TelegramConnector) -> list[Any]:
    """Replace ``emit_inbound`` with a recorder that captures ``event_id``."""
    emitted: list[Any] = []

    async def _fake_emit(**kwargs: Any) -> None:
        emitted.append(kwargs.get("event_id"))

    connector.emit_inbound = _fake_emit  # type: ignore[method-assign]
    return emitted


# ── _shutdown_application: drain ordering + completeness ────────────────────


async def test_shutdown_drains_queue_after_stopping_ptb_in_order(
    ptb_bot: Bot,
) -> None:
    """``_shutdown_application`` must, in order: stop the updater, stop the
    application (so every fetched update's handler has put into
    ``inbound_queue``), drain ``inbound_queue`` to completion, then shut
    the application down.  Pre-fix the drain was absent, so every item
    sitting in ``inbound_queue`` at shutdown was abandoned and then
    confirmed/deleted by Telegram — silent loss."""
    state, application, updater = _make_state()
    connector = _make_connector()
    order: list[str] = []

    async def _stop_updater() -> None:
        order.append("updater.stop")

    async def _stop_app() -> None:
        order.append("application.stop")

    async def _shutdown_app() -> None:
        order.append("application.shutdown")

    updater.stop = AsyncMock(side_effect=_stop_updater)
    application.stop = AsyncMock(side_effect=_stop_app)
    application.shutdown = AsyncMock(side_effect=_shutdown_app)
    emitted = _patch_emit(connector)

    # Recoded-emit side effect that stamps order at drain time.
    real_emit = connector.emit_inbound

    async def _ordered_emit(**kwargs: Any) -> None:
        order.append(f"emit:{kwargs.get('event_id')}")
        await real_emit(**kwargs)

    connector.emit_inbound = _ordered_emit  # type: ignore[method-assign]

    for mid in (101, 102, 103):
        await state.inbound_queue.put(_inbound(ptb_bot, message_id=mid))

    await connector._shutdown_application(CONNECTION_ID, state)

    assert state.inbound_queue.empty(), "inbound_queue must be fully drained"
    assert set(emitted) == {_event_id(101), _event_id(102), _event_id(103)}
    assert order == [
        "updater.stop",
        "application.stop",
        "emit:telegram-123456789-101",
        "emit:telegram-123456789-102",
        "emit:telegram-123456789-103",
        "application.shutdown",
    ], (
        f"shutdown ordering is load-bearing: updater.stop → application.stop "
        f"(process fetched PTB updates) → drain inbound_queue → "
        f"application.shutdown.  Got {order!r}"
    )


async def test_shutdown_drain_runs_even_if_updater_stop_raises(ptb_bot: Bot) -> None:
    """If ``updater.stop()`` raises (e.g. PTB state edge case), the drain
    must still run — a best-effort failure cannot skip emitting the
    queued inbound, or that failure turns a recoverable PTB hiccup into
    silent message loss."""
    state, _application, updater = _make_state()
    connector = _make_connector()
    emitted = _patch_emit(connector)
    updater.stop = AsyncMock(side_effect=RuntimeError("updater boom"))

    await state.inbound_queue.put(_inbound(ptb_bot, message_id=101))

    await connector._shutdown_application(CONNECTION_ID, state)

    assert set(emitted) == {_event_id(101)}, "drain must run despite updater.stop failing"
    assert state.inbound_queue.empty()


async def test_shutdown_drain_runs_even_if_application_stop_raises(ptb_bot: Bot) -> None:
    """Symmetric: a failure in ``application.stop()`` cannot skip the
    drain."""
    state, application, _updater = _make_state()
    connector = _make_connector()
    emitted = _patch_emit(connector)
    application.stop = AsyncMock(side_effect=RuntimeError("app stop boom"))

    await state.inbound_queue.put(_inbound(ptb_bot, message_id=101))

    await connector._shutdown_application(CONNECTION_ID, state)

    assert set(emitted) == {_event_id(101)}
    assert state.inbound_queue.empty()


async def test_shutdown_drain_empty_queue_is_a_noop(ptb_bot: Bot) -> None:
    """Draining an already-empty queue must not emit anything or raise."""
    state, _application, _updater = _make_state()
    connector = _make_connector()
    emitted = _patch_emit(connector)

    await connector._shutdown_application(CONNECTION_ID, state)

    assert emitted == []
    assert state.inbound_queue.empty()


async def test_shutdown_drain_skips_failing_item_and_emits_the_rest(ptb_bot: Bot) -> None:
    """A single ``emit_inbound`` failure during the drain is logged and
    skipped so it cannot abandon the rest of the queue — losing the
    whole backlog to one transient failure would be the same silent-loss
    class the fix targets."""
    state, _application, _updater = _make_state()
    connector = _make_connector()
    emitted: list[Any] = []
    fail_mid = 102

    async def _partial_emit(**kwargs: Any) -> None:
        eid = kwargs.get("event_id")
        if eid == _event_id(fail_mid):
            raise RuntimeError("transient emit failure")
        emitted.append(eid)

    connector.emit_inbound = _partial_emit  # type: ignore[method-assign]

    for mid in (101, 102, 103):
        await state.inbound_queue.put(_inbound(ptb_bot, message_id=mid))

    await connector._shutdown_application(CONNECTION_ID, state)

    assert set(emitted) == {_event_id(101), _event_id(103)}, "the non-failing items must emit"
    assert state.inbound_queue.empty()


# ── _drain_queue: cancellation re-enqueues the in-flight item ───────────────


async def test_drain_queue_re_enqueues_in_flight_item_on_cancel(ptb_bot: Bot) -> None:
    """When the drainer is cancelled mid-``emit_inbound`` (the TaskGroup is
    being torn down), the in-flight item is put back into the queue so the
    shutdown drain can still emit it.  Without this the in-flight item is
    abandoned exactly like the queued ones."""
    state, _application, _updater = _make_state()
    connector = _make_connector()

    in_flight_mid = 101
    started_emit = asyncio.Event()
    release_emit = asyncio.Event()

    async def _blocking_emit(**kwargs: Any) -> None:
        started_emit.set()
        await release_emit.wait()

    connector.emit_inbound = _blocking_emit  # type: ignore[method-assign]

    await state.inbound_queue.put(_inbound(ptb_bot, message_id=in_flight_mid))

    drain = asyncio.ensure_future(connector._drain_queue(CONNECTION_ID, state))
    await started_emit.wait()  # drainer picked up the item and is mid-emit
    assert state.inbound_queue.empty(), "item is in-flight (dequeued), not queued"

    drain.cancel()
    with pytest.raises(asyncio.CancelledError):
        await drain

    # The in-flight item must have been re-enqueued by the cancellation handler.
    assert state.inbound_queue.qsize() == 1, "in-flight item must be re-enqueued, not abandoned"

    # And the shutdown drain can now emit it losslessly.  The emitted
    # event_id proving it is the SAME in-flight item (message_id 101), not a
    # phantom, is the assertion below.
    emitted = _patch_emit(connector)
    await connector._drain_remaining(CONNECTION_ID, state)
    assert set(emitted) == {_event_id(in_flight_mid)}
    assert state.inbound_queue.empty()


async def test_drain_queue_idle_cancel_does_not_re_enqueue() -> None:
    """Cancelled while waiting on an empty queue (no item dequeued) must
    not re-enqueue a phantom item — there is nothing to put back, and a
    naive handler referencing the unbound item would NameError."""
    state, _application, _updater = _make_state()
    connector = _make_connector()
    emit_mock = AsyncMock()
    connector.emit_inbound = emit_mock  # type: ignore[method-assign]

    drain = asyncio.ensure_future(connector._drain_queue(CONNECTION_ID, state))
    await asyncio.sleep(0)  # let it park on `inbound_queue.get()`

    drain.cancel()
    with pytest.raises(asyncio.CancelledError):
        await drain

    assert state.inbound_queue.empty(), "no item was dequeued, so nothing to re-enqueue"
    emit_mock.assert_not_called()


# ── serve_connection: end-to-end graceful shutdown ──────────────────────────


async def test_serve_connection_drains_queue_on_graceful_shutdown(
    ptb_bot: Bot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The headline regression: on graceful cancellation, every item still
    buffered in ``inbound_queue`` (behind the in-flight ``emit_inbound``,
    the common media-heavy-chat case) is emitted before the connector
    exits.  Pre-fix the TaskGroup cancellation abandoned the queue, then
    ``updater.stop()`` confirmed/deleted those updates to Telegram, and
    they were silently, unrecoverably lost on restart."""
    state, application, updater = _make_state()
    connector = _make_connector()

    # Pre-populate the queue with the backlog that exists at shutdown time.
    mids = (101, 102, 103)
    for mid in mids:
        await state.inbound_queue.put(_inbound(ptb_bot, message_id=mid))

    # Short-circuit _build_state so serve_connection uses our pre-built,
    # pre-populated state without touching the network.
    monkeypatch.setattr(connector, "_build_state", AsyncMock(return_value=state))

    emit_started = asyncio.Event()
    release_emit = asyncio.Event()
    emitted: list[Any] = []
    call_count = 0

    async def _emit(**kwargs: Any) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First emit (the drainer's in-flight item) blocks forever; the
            # rest of the backlog stays queued behind it — the loss window.
            emit_started.set()
            await release_emit.wait()
        emitted.append(kwargs.get("event_id"))

    monkeypatch.setattr(connector, "emit_inbound", _emit)

    task = asyncio.ensure_future(
        connector.serve_connection(CONNECTION_ID, {"bot_token": "irrelevant"})
    )
    await emit_started.wait()  # drainer took item 101 and is mid-emit
    assert state.inbound_queue.qsize() == 2, "items 102,103 are queued behind 101"

    # Simulate the runner's SIGTERM/SIGINT graceful cancellation.
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Every buffered item was emitted by the shutdown drain before exit:
    # item 101 was re-enqueued by the cancelled drainer, items 102 & 103
    # were still queued.  No loss.
    assert set(emitted) == {_event_id(mid) for mid in mids}, (
        f"graceful shutdown lost buffered inbound updates: emitted {emitted!r}, "
        f"expected event_ids for message_ids {mids!r}"
    )
    assert state.inbound_queue.empty(), "inbound_queue must be drained before exit"
    updater.stop.assert_awaited_once()
    application.stop.assert_awaited_once()
    application.shutdown.assert_awaited_once()
