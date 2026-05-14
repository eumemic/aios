"""Cover ``_handle_envelope``'s drop-and-continue behavior on routine 4xx.

A single bad envelope going to the api must not tear the container down:
422s and other routine 4xx are logged + dropped so the connector keeps
serving other connections.  401/403 (token revoked / runtime
misconfigured) and 5xx (server outage or bug) still propagate so the
operator sees red on real problems.
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


async def test_handle_envelope_drops_on_422(
    connector: SignalConnector, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 422 from the api affects one envelope; the container survives."""
    monkeypatch.setattr(
        connector, "emit_inbound", AsyncMock(side_effect=_http_status_error(422, '{"detail":"bad"}'))
    )
    receipt = AsyncMock()
    monkeypatch.setattr(connector, "_send_read_receipt", receipt)

    await connector._handle_envelope(CONNECTION_ID, _state(), _make_envelope())

    receipt.assert_not_awaited()


async def test_handle_envelope_drops_on_400(
    connector: SignalConnector, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        connector, "emit_inbound", AsyncMock(side_effect=_http_status_error(400))
    )
    receipt = AsyncMock()
    monkeypatch.setattr(connector, "_send_read_receipt", receipt)

    await connector._handle_envelope(CONNECTION_ID, _state(), _make_envelope())

    receipt.assert_not_awaited()


async def test_handle_envelope_raises_on_401(
    connector: SignalConnector, monkeypatch: pytest.MonkeyPatch
) -> None:
    """401 means the runtime token is busted — fatal for every connection."""
    monkeypatch.setattr(
        connector, "emit_inbound", AsyncMock(side_effect=_http_status_error(401))
    )

    with pytest.raises(httpx.HTTPStatusError):
        await connector._handle_envelope(CONNECTION_ID, _state(), _make_envelope())


async def test_handle_envelope_raises_on_500(
    connector: SignalConnector, monkeypatch: pytest.MonkeyPatch
) -> None:
    """5xx is a server-side outage; let the container crash + restart."""
    monkeypatch.setattr(
        connector, "emit_inbound", AsyncMock(side_effect=_http_status_error(503))
    )

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
