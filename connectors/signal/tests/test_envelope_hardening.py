"""Regression suite for #908 — harden signal connector envelope handling.

Three failure classes are exercised:

* **Parse-layer drops** (``parse_envelope`` returns ``None``): every drop
  now emits a structured ``signal.inbound.skipped`` log with a
  discriminating ``reason`` so source-less receipts / missing-content
  envelopes are observable instead of silent.
* **Read-loop survival**: a single malformed envelope (source-less
  receipt, missing-content, raw non-JSON, or a ``parse_envelope`` that
  raises) must not kill the inbound dispatcher — the loop skips the bad
  envelope and processes the next good one.
* **Reconnect**: a transient listener TCP drop while the daemon
  subprocess is still alive triggers a bounded-backoff reconnect; a drop
  with a dead subprocess stays fatal (re-raise → container restart).

Acceptance shapes mirror signal-cli 0.14.x JSON-RPC daemon output.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import pytest
from structlog.testing import capture_logs

from aios_signal import connector as connector_module
from aios_signal.connector import SignalConnector, _SignalConnectionState
from aios_signal.errors import ListenerClosedError
from aios_signal.parse import parse_envelope
from aios_signal.rpc import RpcListener
from tests.conftest import ALICE_UUID, BOT_UUID, CONNECTION_ID, PHONE

# ── shared helpers ────────────────────────────────────────────────────

ACCOUNT = "+15550009999"


def _good_dm_envelope(*, message: str = "hello there") -> dict[str, Any]:
    """A minimal DM envelope ``parse_envelope`` accepts."""
    return {
        "source": "+15551234567",
        "sourceUuid": ALICE_UUID,
        "sourceName": "Alice",
        "timestamp": 1700000000000,
        "dataMessage": {"timestamp": 1700000000000, "message": message},
    }


def _receive_line(account: str, envelope: dict[str, Any]) -> bytes:
    notification = {
        "jsonrpc": "2.0",
        "method": "receive",
        "params": {"account": account, "envelope": envelope},
    }
    return json.dumps(notification).encode() + b"\n"


async def _start_server(
    handler: Any,
) -> tuple[asyncio.Server, int]:
    server = await asyncio.start_server(handler, host="127.0.0.1", port=0)
    port = server.sockets[0].getsockname()[1]
    return server, port


def _state() -> _SignalConnectionState:
    return _SignalConnectionState(
        phone=PHONE,
        bot_uuid=BOT_UUID,
        contact_names={},
        groups=[],
    )


@pytest.fixture(autouse=True)
def _fast_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collapse reconnect backoff so reconnect tests run in ms, not seconds.

    ``raising=False`` keeps this autouse fixture from erroring before the
    constants exist (TDD red phase) — tests then fail on their own
    assertions rather than on fixture setup.
    """
    monkeypatch.setattr(connector_module, "_RECONNECT_BACKOFF_INITIAL_S", 0.0, raising=False)
    monkeypatch.setattr(connector_module, "_RECONNECT_BACKOFF_CAP_S", 0.0, raising=False)


# ── A. Parse-layer ────────────────────────────────────────────────────


def test_source_less_receipt_returns_none_and_logs(
    envelope_source_less_receipt: dict[str, Any], bot_uuid: str
) -> None:
    """A receipt with no ``sourceUuid`` (the SPQR shape) drops at the
    source-uuid guard — which runs BEFORE the receipt guard — so the
    reason is ``source_less``, not ``receipt``."""
    with capture_logs() as logs:
        result = parse_envelope(envelope_source_less_receipt, bot_account_uuid=bot_uuid)
    assert result is None
    assert any(
        e["event"] == "signal.inbound.skipped" and e["reason"] == "source_less" for e in logs
    )


def test_missing_server_guid_fields_returns_none_and_logs(
    envelope_missing_server_guid: dict[str, Any], bot_uuid: str
) -> None:
    """A structured content envelope with a ``dataMessage`` carrying no
    text/attachments/reaction drops at the no-content guard."""
    with capture_logs() as logs:
        result = parse_envelope(envelope_missing_server_guid, bot_account_uuid=bot_uuid)
    assert result is None
    assert any(e["event"] == "signal.inbound.skipped" and e["reason"] == "no_content" for e in logs)


def test_self_message_logs_self_message_reason(
    envelope_self: dict[str, Any], bot_uuid: str
) -> None:
    """Self-echo drops carry reason ``self_message`` (debug volume)."""
    with capture_logs() as logs:
        assert parse_envelope(envelope_self, bot_account_uuid=bot_uuid) is None
    assert any(
        e["event"] == "signal.inbound.skipped" and e["reason"] == "self_message" for e in logs
    )


def test_receipt_logs_receipt_reason(envelope_receipt: dict[str, Any], bot_uuid: str) -> None:
    """A receipt WITH a sourceUuid drops at the receipt guard."""
    with capture_logs() as logs:
        assert parse_envelope(envelope_receipt, bot_account_uuid=bot_uuid) is None
    assert any(e["event"] == "signal.inbound.skipped" and e["reason"] == "receipt" for e in logs)


# ── B. Read-loop survival ─────────────────────────────────────────────


async def _dispatch_and_handle(
    connector: SignalConnector,
    state: _SignalConnectionState,
    lines: list[bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[Any], AsyncMock]:
    """End-to-end inbound path against a real fake-TCP daemon.

    Writes ``lines`` to a one-shot server, runs the dispatcher (which
    drops on the server close — ``subprocess_alive`` is stubbed dead so
    that drop is fatal and the dispatcher returns via re-raise), then
    drains the per-account queue through ``_handle_envelope`` exactly as
    ``serve_connection`` does.  Returns the captured logs and the
    ``emit_inbound`` mock so callers can assert skip-reason + which
    envelopes actually emitted.
    """

    async def handler(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        for line in lines:
            w.write(line)
        await w.drain()
        w.close()

    emit = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr(connector, "emit_inbound", emit)
    monkeypatch.setattr(connector, "_send_read_receipt", AsyncMock())

    server, port = await _start_server(handler)
    async with server:
        listener = RpcListener("127.0.0.1", port)
        await listener.connect()
        connector._daemon.listener = listener  # type: ignore[union-attr]
        monkeypatch.setattr(connector._daemon, "subprocess_alive", lambda: False)
        # Capture across BOTH the dispatcher (routing) and the queue drain
        # (handling) — the parse-layer skip log fires inside _handle_envelope.
        with capture_logs() as logs:
            with pytest.raises(ListenerClosedError):
                await connector._inbound_dispatcher()
            queue = connector._inbound_queues[ACCOUNT]
            while not queue.empty():
                await connector._handle_envelope(CONNECTION_ID, state, queue.get_nowait())
        await listener.aclose()
    return logs, emit


async def test_read_loop_survives_source_less_receipt_then_processes_next(
    connector: SignalConnector,
    envelope_source_less_receipt: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acceptance (a): a source-less receipt envelope ahead of a good DM
    must not kill the inbound path — the receipt is skipped (reason
    ``source_less``) and the good DM still reaches ``emit_inbound``."""
    good = _good_dm_envelope()
    state = _state()
    connector.state[CONNECTION_ID] = state
    logs, emit = await _dispatch_and_handle(
        connector,
        state,
        [_receive_line(ACCOUNT, envelope_source_less_receipt), _receive_line(ACCOUNT, good)],
        monkeypatch,
    )

    assert any(
        e["event"] == "signal.inbound.skipped" and e["reason"] == "source_less" for e in logs
    )
    # Only the good DM emitted; the source-less receipt was dropped.
    assert emit.await_count == 1
    assert emit.await_args is not None
    assert emit.await_args.kwargs["content"] == good["dataMessage"]["message"]


async def test_read_loop_survives_missing_server_guid_then_processes_next(
    connector: SignalConnector,
    envelope_missing_server_guid: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acceptance (b): a missing-content envelope ahead of a good DM —
    skipped (reason ``no_content``), good DM still emits."""
    good = _good_dm_envelope()
    state = _state()
    connector.state[CONNECTION_ID] = state
    logs, emit = await _dispatch_and_handle(
        connector,
        state,
        [_receive_line(ACCOUNT, envelope_missing_server_guid), _receive_line(ACCOUNT, good)],
        monkeypatch,
    )

    assert any(e["event"] == "signal.inbound.skipped" and e["reason"] == "no_content" for e in logs)
    assert emit.await_count == 1
    assert emit.await_args is not None
    assert emit.await_args.kwargs["content"] == good["dataMessage"]["message"]


async def test_read_loop_survives_malformed_json_then_processes_next(
    connector: SignalConnector,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acceptance (c): a raw non-JSON line is logged ``rpc.listener.bad_json``
    (with a ``reason`` field) and skipped at the listener layer; the loop
    survives and the following good DM is routed."""
    good = _good_dm_envelope()

    async def handler(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        w.write(b"this is not json\n")
        w.write(_receive_line(ACCOUNT, good))
        await w.drain()
        w.close()

    server, port = await _start_server(handler)
    async with server:
        listener = RpcListener("127.0.0.1", port)
        await listener.connect()
        connector._daemon.listener = listener  # type: ignore[union-attr]
        monkeypatch.setattr(connector._daemon, "subprocess_alive", lambda: False)
        connector.state[CONNECTION_ID] = _state()
        with capture_logs() as logs, pytest.raises(ListenerClosedError):
            await connector._inbound_dispatcher()
        await listener.aclose()

    assert any(
        e["event"] == "rpc.listener.bad_json" and e.get("reason") == "bad_json" for e in logs
    )
    queue = connector._inbound_queues[ACCOUNT]
    assert queue.get_nowait() == good


async def test_dispatcher_survives_raising_handle_and_processes_next(
    connector: SignalConnector, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LOAD-BEARING for the per-envelope handling guard: if ``parse_envelope``
    raises on the FIRST envelope, ``_handle_envelope`` must log
    ``signal.inbound.skipped`` reason=parse_error and return — NOT propagate
    — so the SECOND envelope still reaches ``emit_inbound``."""
    calls = {"parse": 0}

    def _flaky_parse(envelope: dict[str, Any], *, bot_account_uuid: str) -> Any:
        calls["parse"] += 1
        if calls["parse"] == 1:
            raise ValueError("boom: malformed envelope dict")
        from aios_signal.parse import InboundMessage

        return InboundMessage(
            chat_type="dm",
            raw_chat_id=ALICE_UUID,
            sender_uuid=ALICE_UUID,
            sender_name="Alice",
            chat_name=None,
            timestamp_ms=1700000000000,
            text="second",
            attachments=(),
            mentions=(),
            reply=None,
            reaction=None,
        )

    monkeypatch.setattr(connector_module, "parse_envelope", _flaky_parse)
    emit = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr(connector, "emit_inbound", emit)
    monkeypatch.setattr(connector, "_send_read_receipt", AsyncMock())

    state = _state()
    with capture_logs() as logs:
        await connector._handle_envelope(CONNECTION_ID, state, {"first": True})
        await connector._handle_envelope(CONNECTION_ID, state, {"second": True})

    assert any(
        e["event"] == "signal.inbound.skipped" and e["reason"] == "parse_error" for e in logs
    )
    emit.assert_awaited_once()  # only the second envelope reached emit_inbound


# ── C. Reconnect ──────────────────────────────────────────────────────


class _ScriptedListener:
    """Fake listener that plays a list of batches.

    Each ``messages()`` call yields the current batch's ``(account,
    envelope)`` pairs then raises ``ListenerClosedError`` (a transient
    drop).  ``reconnect()`` advances to the next batch; once the batches
    are exhausted both ``reconnect()`` and ``messages()`` raise
    ``ListenerClosedError`` (a listener that won't re-establish), which
    drives the dispatcher's give-up-after-max-attempts path.
    """

    def __init__(self, batches: list[list[tuple[str, dict[str, Any]]]]) -> None:
        self._batches = batches
        self._idx = 0
        self.reconnect_calls = 0

    async def messages(self) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        if self._idx >= len(self._batches):
            raise ListenerClosedError("listener exhausted")
        for account, envelope in self._batches[self._idx]:
            yield account, envelope
        raise ListenerClosedError("transient drop")

    async def reconnect(self) -> None:
        self.reconnect_calls += 1
        self._idx += 1
        if self._idx >= len(self._batches):
            raise ListenerClosedError("no more batches")


async def test_dispatcher_reconnects_on_transient_listener_drop(
    connector: SignalConnector, monkeypatch: pytest.MonkeyPatch
) -> None:
    """First connection yields one good envelope then drops; the daemon
    subprocess is still alive, so the dispatcher reconnects and the second
    connection's envelope is routed too.  ``signal.listener.reconnect`` fires.

    After the second batch the listener has no more batches, so its
    ``reconnect()`` raises; with a low ``_RECONNECT_MAX_ATTEMPTS`` the
    dispatcher exhausts retries and re-raises, terminating the test.
    """
    monkeypatch.setattr(connector_module, "_RECONNECT_MAX_ATTEMPTS", 2)
    env_a = _good_dm_envelope(message="first")
    env_b = _good_dm_envelope(message="second")
    listener = _ScriptedListener([[(ACCOUNT, env_a)], [(ACCOUNT, env_b)]])

    connector._daemon.listener = listener  # type: ignore[union-attr]
    monkeypatch.setattr(connector._daemon, "subprocess_alive", lambda: True)
    connector.state[CONNECTION_ID] = _state()

    with capture_logs() as logs, pytest.raises(ListenerClosedError):
        await connector._inbound_dispatcher()

    assert any(e["event"] == "signal.listener.reconnect" for e in logs)
    # A real reconnect happened to advance to the second batch (and the
    # exhaustion retries after it); the first successful reconnect must be
    # counted exactly once, not skipped or doubled.
    assert listener.reconnect_calls >= 1
    queue = connector._inbound_queues[ACCOUNT]
    drained = [queue.get_nowait() for _ in range(queue.qsize())]
    assert env_a in drained
    assert env_b in drained


async def test_dispatcher_reraises_when_daemon_dead(
    connector: SignalConnector, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Listener drops AND the daemon subprocess is dead → fatal: the
    dispatcher re-raises ``ListenerClosedError`` (no reconnect attempt)."""
    listener = _ScriptedListener([[(ACCOUNT, _good_dm_envelope())]])
    connector._daemon.listener = listener  # type: ignore[union-attr]
    monkeypatch.setattr(connector._daemon, "subprocess_alive", lambda: False)
    connector.state[CONNECTION_ID] = _state()

    with pytest.raises(ListenerClosedError):
        await connector._inbound_dispatcher()

    assert listener.reconnect_calls == 0  # never tried to reconnect a dead daemon


async def test_reconnect_gives_up_after_max_attempts(
    connector: SignalConnector, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``subprocess_alive`` stays True but every ``reconnect()`` raises;
    the dispatcher gives up after ``_RECONNECT_MAX_ATTEMPTS`` and re-raises.

    LOCKS the #908 single-count fix: exactly ``_RECONNECT_MAX_ATTEMPTS``
    real ``reconnect()`` TCP calls are made — no more, no less.  A
    regression to the old double-counting shape (where a failed
    ``reconnect()`` left ``_reader is None`` and the next ``messages()``
    re-raised for the SAME logical attempt, halving the real attempts)
    fails this exact-count assertion and the ``attempts=`` log assertion.
    """
    max_attempts = 3
    monkeypatch.setattr(connector_module, "_RECONNECT_MAX_ATTEMPTS", max_attempts)

    class _AlwaysFailListener:
        def __init__(self) -> None:
            self.reconnect_calls = 0
            self.messages_calls = 0

        async def messages(self) -> AsyncIterator[tuple[str, dict[str, Any]]]:
            self.messages_calls += 1
            # Drop immediately with no progress — like the real listener
            # raising when the connection is closed before any line arrives.
            if self.reconnect_calls < 0:  # always False; forces async-generator typing
                yield ACCOUNT, {}
            raise ListenerClosedError("listener connection closed")

        async def reconnect(self) -> None:
            self.reconnect_calls += 1
            raise ListenerClosedError("reconnect failed")

    listener = _AlwaysFailListener()
    connector._daemon.listener = listener  # type: ignore[union-attr]
    monkeypatch.setattr(connector._daemon, "subprocess_alive", lambda: True)
    connector.state[CONNECTION_ID] = _state()

    with capture_logs() as logs, pytest.raises(ListenerClosedError):
        await connector._inbound_dispatcher()

    # Exactly one real reconnect() call per attempt, no double-count.
    assert listener.reconnect_calls == max_attempts
    # ``messages()`` runs once (initial drop); the failed reconnects never
    # re-enter it, so it is NOT a second counting path.
    assert listener.messages_calls == 1
    exhausted = [e for e in logs if e["event"] == "signal.listener.reconnect_exhausted"]
    assert len(exhausted) == 1
    assert exhausted[0]["attempts"] == max_attempts
    # The per-attempt reconnect log reports true 1-based attempt numbers.
    attempts_logged = sorted(
        e["attempt"] for e in logs if e["event"] == "signal.listener.reconnect"
    )
    assert attempts_logged == list(range(1, max_attempts + 1))


# ── D. Observability ──────────────────────────────────────────────────


async def test_skip_counter_rollup_emitted(
    connector: SignalConnector, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The connector keeps a running per-reason skip counter and emits a
    periodic roll-up ``signal.inbound.skip_counts`` every
    ``_SKIP_ROLLUP_EVERY`` skips, with a ``counts`` dict that discriminates
    the connector-observed reasons (``parse_none`` vs ``parse_error``).

    Fine-grained parse-internal reasons (source_less / no_content / …) ride
    on the per-drop ``signal.inbound.skipped`` logs, asserted elsewhere; the
    roll-up summarises what the handling layer itself sees."""
    monkeypatch.setattr(connector_module, "_SKIP_ROLLUP_EVERY", 3)
    monkeypatch.setattr(connector, "_send_read_receipt", AsyncMock())

    # A receipt envelope parses to None (skip reason parse_none at the
    # handle layer); a raising parse increments parse_error.  Mix both so
    # the roll-up's counts dict is genuinely keyed by reason.
    dropped = {
        "sourceUuid": ALICE_UUID,
        "timestamp": 1700000005000,
        "receiptMessage": {"when": 1, "timestamps": [1]},
    }
    raising = {"sourceUuid": ALICE_UUID, "timestamp": 1, "boom": True}

    def _maybe_raise(envelope: dict[str, Any], *, bot_account_uuid: str) -> Any:
        if envelope.get("boom"):
            raise ValueError("boom")
        return parse_envelope(envelope, bot_account_uuid=bot_account_uuid)

    monkeypatch.setattr(connector_module, "parse_envelope", _maybe_raise)

    state = _state()
    with capture_logs() as logs:
        await connector._handle_envelope(CONNECTION_ID, state, dropped)
        await connector._handle_envelope(CONNECTION_ID, state, raising)
        await connector._handle_envelope(CONNECTION_ID, state, dropped)

    rollups = [e for e in logs if e["event"] == "signal.inbound.skip_counts"]
    assert rollups, "expected a skip_counts roll-up after 3 skips"
    counts = rollups[-1]["counts"]
    assert isinstance(counts, dict)
    assert counts.get("parse_none", 0) == 2
    assert counts.get("parse_error", 0) == 1
