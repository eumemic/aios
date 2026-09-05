"""serve_connection inbound path: demux registration + queue drain →
emit_inbound, and the event_id single-source invariant (design §3.2)."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import httpx
import pytest
import structlog

from aios_sms.connector import (
    _EVENT_ID_FIELD,
    _FORBIDDEN_EVENT_ID_FIELDS,
    SmsConnector,
)
from aios_sms.webhook import InboundEnvelope

OUR_NUMBER = "+18005551234"
PEER = "+14155550000"


@pytest.fixture
def connector() -> SmsConnector:
    return SmsConnector()


async def test_emit_envelope_maps_twilio_fields(connector: SmsConnector) -> None:
    captured: list[dict[str, Any]] = []

    async def _fake_emit(**kwargs: Any) -> dict[str, Any]:
        captured.append(kwargs)
        return {"deduped": False}

    connector.emit_inbound = _fake_emit  # type: ignore[method-assign]

    env = InboundEnvelope(
        connection_id="conn_1",
        params={
            "From": PEER,
            "To": OUR_NUMBER,
            "Body": "hello world",
            "MessageSid": "SM_abc",
            "NumSegments": "2",
            # aliases present on the wire but MUST NOT be used as event_id
            "SmsSid": "SM_alias_should_be_ignored",
            "SmsMessageSid": "SM_alias_should_be_ignored",
        },
    )
    await connector._emit_envelope("conn_1", env)

    assert len(captured) == 1
    call = captured[0]
    assert call["connection_id"] == "conn_1"
    assert call["chat_id"] == PEER
    assert call["sender"] == {"display_name": PEER}
    assert call["content"] == "hello world"
    # event_id is the single-source MessageSid, never an alias.
    assert call["event_id"] == "SM_abc"
    # From provenance is stamped unverified toward the model.
    assert call["metadata"]["sender_verified"] is False
    assert call["metadata"]["num_segments"] == "2"


async def test_event_id_is_message_sid_not_aliases(connector: SmsConnector) -> None:
    """The single-source invariant: event_id derives from MessageSid and
    the SmsSid / SmsMessageSid aliases are never the source (design §3.2)."""
    assert _EVENT_ID_FIELD == "MessageSid"
    assert "SmsSid" in _FORBIDDEN_EVENT_ID_FIELDS
    assert "SmsMessageSid" in _FORBIDDEN_EVENT_ID_FIELDS

    captured: list[str | None] = []

    async def _fake_emit(**kwargs: Any) -> dict[str, Any]:
        captured.append(kwargs.get("event_id"))
        return {"deduped": False}

    connector.emit_inbound = _fake_emit  # type: ignore[method-assign]

    # Aliases differ from MessageSid; if the impl ever read an alias this
    # would surface as the wrong event_id.
    env = InboundEnvelope(
        connection_id="c",
        params={
            "From": PEER,
            "Body": "x",
            "MessageSid": "SM_real",
            "SmsSid": "SM_wrong",
            "SmsMessageSid": "SM_wrong",
        },
    )
    await connector._emit_envelope("c", env)
    assert captured == ["SM_real"]


async def test_emit_envelope_drops_when_no_message_sid(connector: SmsConnector) -> None:
    called = False

    async def _fake_emit(**kwargs: Any) -> dict[str, Any]:
        nonlocal called
        called = True
        return {}

    connector.emit_inbound = _fake_emit  # type: ignore[method-assign]
    env = InboundEnvelope(connection_id="c", params={"From": PEER, "Body": "x"})
    await connector._emit_envelope("c", env)
    assert called is False


async def test_serve_connection_registers_demux_and_drains() -> None:
    connector = SmsConnector()
    emitted: list[dict[str, Any]] = []

    async def _fake_emit(**kwargs: Any) -> dict[str, Any]:
        emitted.append(kwargs)
        return {"deduped": False}

    connector.emit_inbound = _fake_emit  # type: ignore[method-assign]

    secrets = {"from_number": "1 800 555-1234", "auth_token": "tok"}
    task = asyncio.create_task(connector.serve_connection("conn_1", secrets))
    # Let serve_connection register before we route through the listener.
    await asyncio.sleep(0.05)

    # The demux map is keyed by the normalized number.
    entry = connector._listener.lookup(OUR_NUMBER)
    assert entry is not None
    assert entry.connection_id == "conn_1"

    # Push an envelope onto the registered queue; the drain loop emits it.
    entry.queue.put_nowait(
        InboundEnvelope(
            connection_id="conn_1",
            params={"From": PEER, "Body": "drained", "MessageSid": "SM_x"},
        )
    )
    await asyncio.sleep(0.05)
    assert emitted and emitted[0]["content"] == "drained"
    assert emitted[0]["event_id"] == "SM_x"

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    # On teardown the demux registration is removed.
    assert connector._listener.lookup(OUR_NUMBER) is None


async def test_serve_connection_requires_from_number_and_token() -> None:
    connector = SmsConnector()
    with pytest.raises(RuntimeError, match="from_number"):
        await connector.serve_connection("c", {"auth_token": "t"})
    with pytest.raises(RuntimeError, match="auth_token"):
        await connector.serve_connection("c", {"from_number": OUR_NUMBER})


# ── transport-error retry / drop-with-log (#2093 transport gap) ─────────
#
# Twilio already received ``200`` for every enqueued envelope, so it will
# not redeliver. A transient ``httpx.RequestError`` on ``emit_inbound``
# (no ``response`` object, so the runner's ``is_error`` guard is
# unreachable) must NOT escape the drain loop — otherwise the ``finally``
# discards the whole 200-acked backlog and unregisters the number. The
# drain loop retries the in-flight envelope in-process and only
# logs-and-drops it after bounded attempts, so the per-number feed and
# the pending queue survive a transient aios-api blip.

_SECRETS = {"from_number": "1 800 555-1234", "auth_token": "tok"}


def _tx_error(
    exc_cls: type[httpx.RequestError], msg: str = "aios-api unreachable"
) -> httpx.RequestError:
    """Build a transport error with a request, as httpx does in production."""
    return exc_cls(msg, request=httpx.Request("POST", "http://aios-api/inbound"))


def _envelope(sid: str, body: str = "hello") -> InboundEnvelope:
    return InboundEnvelope(
        connection_id="conn_1",
        params={"From": PEER, "To": OUR_NUMBER, "Body": body, "MessageSid": sid},
    )


def _no_backoff(connector: SmsConnector) -> SmsConnector:
    """Zero the retry backoff so multi-attempt tests stay fast."""
    connector.EMIT_RETRY_BACKOFF_INITIAL = 0.0
    connector.EMIT_RETRY_BACKOFF_MAX = 0.0
    return connector


async def test_transport_error_retried_then_envelope_delivered() -> None:
    """A transient httpx.RequestError on the first emit is retried; the
    envelope is delivered on the retry and the demux registration survives
    (the number does not go dark). Pre-fix this raised out of
    serve_connection and dropped the 200-acked envelope."""
    connector = _no_backoff(SmsConnector())
    attempts: list[str | None] = []

    async def _fake_emit(**kwargs: Any) -> dict[str, Any]:
        attempts.append(kwargs.get("event_id"))
        if len(attempts) == 1:
            raise _tx_error(httpx.ConnectError)
        return {"deduped": False}

    connector.emit_inbound = _fake_emit  # type: ignore[method-assign]

    task = asyncio.create_task(connector.serve_connection("conn_1", _SECRETS))
    await asyncio.sleep(0.05)
    entry = connector._listener.lookup(OUR_NUMBER)
    assert entry is not None
    entry.queue.put_nowait(_envelope("SM_one"))
    await asyncio.sleep(0.05)

    # First attempt raised, retry succeeded — envelope delivered, not lost.
    assert attempts == ["SM_one", "SM_one"]
    # The per-number feed survived the transient blip.
    assert connector._listener.lookup(OUR_NUMBER) is not None
    assert "conn_1" in connector.state

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


@pytest.mark.parametrize(
    "exc_cls",
    [
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.ReadError,
        httpx.RemoteProtocolError,
        httpx.PoolTimeout,
    ],
)
async def test_all_transport_error_subclasses_retried(exc_cls: type[httpx.RequestError]) -> None:
    """Every httpx.RequestError subclass (the six transport families the
    bug enumerates) is retried, not dropped on the first failure."""
    connector = _no_backoff(SmsConnector())
    attempts: list[str | None] = []

    async def _fake_emit(**kwargs: Any) -> dict[str, Any]:
        attempts.append(kwargs.get("event_id"))
        if len(attempts) == 1:
            raise _tx_error(exc_cls)
        return {"deduped": False}

    connector.emit_inbound = _fake_emit  # type: ignore[method-assign]

    task = asyncio.create_task(connector.serve_connection("conn_1", _SECRETS))
    await asyncio.sleep(0.05)
    entry = connector._listener.lookup(OUR_NUMBER)
    assert entry is not None
    entry.queue.put_nowait(_envelope("SM_one"))
    await asyncio.sleep(0.05)

    assert attempts == ["SM_one", "SM_one"]
    assert connector._listener.lookup(OUR_NUMBER) is not None

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def test_transport_error_logs_retry_with_message_sid() -> None:
    """The ``sms.inbound.emit_retry`` log carries the ``MessageSid``
    (event_id) so an operator has connector-side trace for the in-flight
    envelope during the blip."""
    connector = _no_backoff(SmsConnector())
    calls = 0

    async def _fake_emit(**kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _tx_error(httpx.ReadTimeout)
        return {"deduped": False}

    connector.emit_inbound = _fake_emit  # type: ignore[method-assign]

    task = asyncio.create_task(connector.serve_connection("conn_1", _SECRETS))
    await asyncio.sleep(0.05)
    entry = connector._listener.lookup(OUR_NUMBER)
    assert entry is not None
    with structlog.testing.capture_logs() as records:
        entry.queue.put_nowait(_envelope("SM_trace"))
        await asyncio.sleep(0.05)

    retries = [r for r in records if r.get("event") == "sms.inbound.emit_retry"]
    assert len(retries) == 1
    assert retries[0]["event_id"] == "SM_trace"
    assert retries[0]["attempt"] == 1
    assert retries[0]["error"] == "ReadTimeout"

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def test_transport_error_exhausts_retries_drops_with_log_and_drains_next() -> None:
    """After bounded retries are exhausted the envelope is logged
    (``sms.inbound.emit_dropped`` carries the ``MessageSid``, the first
    per-envelope connector-side trace for the backlog population) and
    dropped; the drain continues to the next envelope, which is emitted
    normally. The feed does not tear down."""
    connector = _no_backoff(SmsConnector())
    calls = 0

    async def _fake_emit(**kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        # First 3 calls = SM_one's attempts (all fail); 4th = SM_two (success).
        if calls <= 3:
            raise _tx_error(httpx.ConnectError)
        return {"deduped": False}

    connector.emit_inbound = _fake_emit  # type: ignore[method-assign]

    task = asyncio.create_task(connector.serve_connection("conn_1", _SECRETS))
    await asyncio.sleep(0.05)
    entry = connector._listener.lookup(OUR_NUMBER)
    assert entry is not None
    entry.queue.put_nowait(_envelope("SM_one"))
    entry.queue.put_nowait(_envelope("SM_two"))
    with structlog.testing.capture_logs() as records:
        await asyncio.sleep(0.1)

    # SM_one attempted 3x (default max), then dropped; SM_two emitted once.
    assert calls == 4
    dropped = [r for r in records if r.get("event") == "sms.inbound.emit_dropped"]
    assert len(dropped) == 1
    assert dropped[0]["event_id"] == "SM_one"
    assert dropped[0]["attempts"] == connector.EMIT_RETRY_MAX_ATTEMPTS
    emitted = [r for r in records if r.get("event") == "sms.inbound.emitted"]
    assert any(r["event_id"] == "SM_two" for r in emitted)
    # Feed survived.
    assert connector._listener.lookup(OUR_NUMBER) is not None
    assert "conn_1" in connector.state

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def test_transport_error_does_not_lose_pending_envelopes() -> None:
    """A transient blip on the first envelope does not discard the
    pending envelopes behind it: after the retry succeeds the drain
    continues and every enqueued envelope is delivered."""
    connector = _no_backoff(SmsConnector())
    calls = 0

    async def _fake_emit(**kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _tx_error(httpx.ConnectError)
        return {"deduped": False}

    connector.emit_inbound = _fake_emit  # type: ignore[method-assign]

    task = asyncio.create_task(connector.serve_connection("conn_1", _SECRETS))
    await asyncio.sleep(0.05)
    entry = connector._listener.lookup(OUR_NUMBER)
    assert entry is not None
    entry.queue.put_nowait(_envelope("SM_a"))
    entry.queue.put_nowait(_envelope("SM_b"))
    entry.queue.put_nowait(_envelope("SM_c"))
    await asyncio.sleep(0.1)

    # SM_a: raise then retry-success (2 calls); SM_b, SM_c: one call each.
    assert calls == 4
    assert connector._listener.lookup(OUR_NUMBER) is not None

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def test_fatal_auth_status_error_propagates_and_unregisters() -> None:
    """A fatal 401/403 (httpx.HTTPStatusError, NOT a RequestError) still
    propagates and tears the connection down — broken auth is not a
    transient blip and must not be swallowed by the transport retry."""
    connector = _no_backoff(SmsConnector())

    async def _fake_emit(**kwargs: Any) -> dict[str, Any]:
        raise httpx.HTTPStatusError(
            "401 Unauthorized",
            request=httpx.Request("POST", "http://aios-api/inbound"),
            response=httpx.Response(401),
        )

    connector.emit_inbound = _fake_emit  # type: ignore[method-assign]

    task = asyncio.create_task(connector.serve_connection("conn_1", _SECRETS))
    await asyncio.sleep(0.05)
    entry = connector._listener.lookup(OUR_NUMBER)
    assert entry is not None
    entry.queue.put_nowait(_envelope("SM_auth"))
    await asyncio.sleep(0.05)

    # The fatal-auth error escaped: serve_connection ended with it, and the
    # finally block unregistered the number + dropped the state.
    assert task.done()
    assert isinstance(task.exception(), httpx.HTTPStatusError)
    assert connector._listener.lookup(OUR_NUMBER) is None
    assert "conn_1" not in connector.state


async def test_nonfatal_5xx_returning_none_is_not_retried() -> None:
    """A non-fatal 5xx (emit_inbound returns None) is dropped per-message,
    NOT retried — the retry is scoped to transport errors
    (httpx.RequestError), not to response errors the #2093 fix already
    handles. Confirms the fix does not change the 5xx drop posture."""
    connector = _no_backoff(SmsConnector())
    attempts: list[str | None] = []

    async def _fake_emit(**kwargs: Any) -> dict[str, Any] | None:
        attempts.append(kwargs.get("event_id"))
        return None  # 503-style non-fatal response error

    connector.emit_inbound = _fake_emit  # type: ignore[method-assign]

    task = asyncio.create_task(connector.serve_connection("conn_1", _SECRETS))
    await asyncio.sleep(0.05)
    entry = connector._listener.lookup(OUR_NUMBER)
    assert entry is not None
    entry.queue.put_nowait(_envelope("SM_5xx"))
    entry.queue.put_nowait(_envelope("SM_5xx_two"))
    await asyncio.sleep(0.1)

    # One emit call per envelope, no retries.
    assert attempts == ["SM_5xx", "SM_5xx_two"]
    assert connector._listener.lookup(OUR_NUMBER) is not None

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def test_emit_dropped_after_exhaustion_takes_priority_over_emitted() -> None:
    """When all attempts fail,``sms.inbound.emit_dropped`` is logged and
    ``sms.inbound.emitted`` is NOT (the envelope is dropped, not
    delivered). Guards against a control-flow bug that logs both."""
    connector = _no_backoff(SmsConnector())

    async def _fake_emit(**kwargs: Any) -> dict[str, Any]:
        raise _tx_error(httpx.RemoteProtocolError)

    connector.emit_inbound = _fake_emit  # type: ignore[method-assign]

    task = asyncio.create_task(connector.serve_connection("conn_1", _SECRETS))
    await asyncio.sleep(0.05)
    entry = connector._listener.lookup(OUR_NUMBER)
    assert entry is not None
    with structlog.testing.capture_logs() as records:
        entry.queue.put_nowait(_envelope("SM_lost"))
        await asyncio.sleep(0.1)

    assert len([r for r in records if r.get("event") == "sms.inbound.emit_dropped"]) == 1
    assert len([r for r in records if r.get("event") == "sms.inbound.emitted"]) == 0
    assert connector._listener.lookup(OUR_NUMBER) is not None

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
