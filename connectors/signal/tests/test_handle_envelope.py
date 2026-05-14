"""Cover ``_handle_envelope``'s response to ``emit_inbound`` outcomes.

The drop-and-continue policy itself now lives in
:meth:`HttpConnector.emit_inbound` and is unit-tested in
``packages/aios-connector-http/tests/test_runner.py``
(``TestEmitInbound4xxDrop``).  Here we verify that
``_handle_envelope`` honors the SDK's two return shapes:

* ``None`` → envelope was dropped (routine 4xx); skip the read receipt
  since the inbound never landed in the session log.
* ``dict`` → envelope persisted; fire the explicit read receipt.

5xx / 401 / 403 raise out of ``emit_inbound`` and must keep
propagating so the container crashes + restarts on auth-broken /
server-outage failures.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from aios_signal.connector import SignalConnector, _SignalConnectionState
from tests.conftest import ALICE_UUID, BOT_UUID, CONNECTION_ID, PHONE


def _make_envelope(sender_uuid: str = ALICE_UUID) -> dict[str, Any]:
    """Minimal valid DM envelope that ``parse_envelope`` accepts."""
    return {
        "sourceUuid": sender_uuid,
        "sourceName": "Alice",
        "timestamp": 1700000000000,
        "dataMessage": {
            "timestamp": 1700000000000,
            "message": "hello",
        },
    }


def _http_status_error(status_code: int, body: str = "") -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "/v1/connectors/runtime/inbound")
    response = httpx.Response(status_code, content=body.encode(), request=request)
    return httpx.HTTPStatusError(f"{status_code}", request=request, response=response)


def _state() -> _SignalConnectionState:
    return _SignalConnectionState(
        phone=PHONE,
        bot_uuid=BOT_UUID,
        contact_names={},
        groups=[],
    )


async def test_handle_envelope_skips_receipt_when_inbound_dropped(
    connector: SignalConnector, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``emit_inbound`` returns ``None`` when the SDK drops a routine 4xx;
    no read receipt should fire since the inbound never persisted."""
    monkeypatch.setattr(connector, "emit_inbound", AsyncMock(return_value=None))
    receipt = AsyncMock()
    monkeypatch.setattr(connector, "_send_read_receipt", receipt)

    await connector._handle_envelope(CONNECTION_ID, _state(), _make_envelope())

    receipt.assert_not_awaited()


async def test_handle_envelope_raises_on_401(
    connector: SignalConnector, monkeypatch: pytest.MonkeyPatch
) -> None:
    """401 means the runtime token is busted — fatal for every connection."""
    monkeypatch.setattr(connector, "emit_inbound", AsyncMock(side_effect=_http_status_error(401)))

    with pytest.raises(httpx.HTTPStatusError):
        await connector._handle_envelope(CONNECTION_ID, _state(), _make_envelope())


async def test_handle_envelope_raises_on_500(
    connector: SignalConnector, monkeypatch: pytest.MonkeyPatch
) -> None:
    """5xx is a server-side outage; let the container crash + restart."""
    monkeypatch.setattr(connector, "emit_inbound", AsyncMock(side_effect=_http_status_error(503)))

    with pytest.raises(httpx.HTTPStatusError):
        await connector._handle_envelope(CONNECTION_ID, _state(), _make_envelope())


async def test_handle_envelope_sends_receipt_on_success(
    connector: SignalConnector, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Happy path: ``_send_read_receipt`` fires after a successful inbound."""
    monkeypatch.setattr(connector, "emit_inbound", AsyncMock(return_value={"ok": True}))
    receipt = AsyncMock()
    monkeypatch.setattr(connector, "_send_read_receipt", receipt)

    await connector._handle_envelope(CONNECTION_ID, _state(), _make_envelope())

    receipt.assert_awaited_once()
