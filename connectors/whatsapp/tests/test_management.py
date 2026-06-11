"""Tests for the WhatsappManagementMixin pairing handlers."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from aios_connector_http import ManagementHandlerError

from aios_whatsapp.connector import WhatsappConnector
from aios_whatsapp.management import normalize_phone

from .conftest import CONNECTION_ID, PHONE


async def test_start_pairing_calls_daemon_and_returns_code(
    connector: WhatsappConnector,
) -> None:
    connector.state[CONNECTION_ID].daemon.start_pairing = (  # type: ignore[attr-defined]
        _async_return({"code": "2@first-qr-code"})
    )
    result = await connector.startPairing(external_account_id=PHONE)
    connector.state[CONNECTION_ID].daemon.start_pairing.assert_awaited_once()  # type: ignore[attr-defined]
    assert result == {"external_account_id": PHONE, "code": "2@first-qr-code"}


async def test_confirm_pairing_returns_paired_outcome(connector: WhatsappConnector) -> None:
    connector.state[CONNECTION_ID].daemon.confirm_pairing = (  # type: ignore[attr-defined]
        _async_return(
            {
                "status": "success",
                "jid": "15551112222@s.whatsapp.net",
                "push_name": "Bot",
            }
        )
    )
    result = await connector.confirmPairing(external_account_id=PHONE)
    assert result == {
        "external_account_id": PHONE,
        "status": "success",
        "jid": "15551112222@s.whatsapp.net",
        "push_name": "Bot",
    }


async def test_confirm_pairing_drops_empty_optional_fields(connector: WhatsappConnector) -> None:
    """A ``timeout`` outcome carries ``status`` only — jid/push_name/reason
    must not surface as empty strings."""
    connector.state[CONNECTION_ID].daemon.confirm_pairing = (  # type: ignore[attr-defined]
        _async_return({"status": "timeout"})
    )
    result = await connector.confirmPairing(external_account_id=PHONE)
    assert result == {"external_account_id": PHONE, "status": "timeout"}


async def test_confirm_pairing_carries_error_reason(connector: WhatsappConnector) -> None:
    connector.state[CONNECTION_ID].daemon.confirm_pairing = (  # type: ignore[attr-defined]
        _async_return({"status": "error", "reason": "boom"})
    )
    result = await connector.confirmPairing(external_account_id=PHONE)
    assert result == {"external_account_id": PHONE, "status": "error", "reason": "boom"}


async def test_unpair_invokes_daemon(connector: WhatsappConnector) -> None:
    connector.state[CONNECTION_ID].daemon.unpair = _async_return(None)  # type: ignore[attr-defined]
    result = await connector.unpair(external_account_id=PHONE)
    connector.state[CONNECTION_ID].daemon.unpair.assert_awaited_once()  # type: ignore[attr-defined]
    assert result == {"external_account_id": PHONE, "status": "ok"}


async def test_unknown_phone_raises_management_error(connector: WhatsappConnector) -> None:
    with pytest.raises(ManagementHandlerError) as ei:
        await connector.startPairing(external_account_id="+19999999999")
    assert ei.value.payload == {
        "status": "no_active_connection",
        "external_account_id": "+19999999999",
    }


async def test_start_pairing_empty_code_raises_structured_error(
    connector: WhatsappConnector,
) -> None:
    """A daemon-side contract bug (empty code) must surface as a
    structured ManagementHandlerError, not a 200 OK with a blank QR."""
    connector.state[CONNECTION_ID].daemon.start_pairing = (  # type: ignore[attr-defined]
        _async_return({"code": ""})
    )
    with pytest.raises(ManagementHandlerError) as ei:
        await connector.startPairing(external_account_id=PHONE)
    assert ei.value.payload["status"] == "error"
    assert ei.value.payload["external_account_id"] == PHONE
    assert "empty pairing code" in ei.value.payload["reason"]


async def test_confirm_pairing_empty_status_falls_back_to_error(
    connector: WhatsappConnector,
) -> None:
    """If the daemon emits an empty status (e.g. ConfirmPairing race), the
    mixin defaults to 'error' so the route's Literal validator doesn't 500."""
    connector.state[CONNECTION_ID].daemon.confirm_pairing = (  # type: ignore[attr-defined]
        _async_return({"status": ""})
    )
    result = await connector.confirmPairing(external_account_id=PHONE)
    assert result == {"external_account_id": PHONE, "status": "error"}


async def test_confirm_pairing_mismatch_unpairs_and_errors(
    connector: WhatsappConnector,
) -> None:
    """A device scanned into the wrong account is live exposure: the
    mismatch must hard-fail, the offending jid surface, and the device
    be auto-unpaired — never a success-looking response."""
    connector.state[CONNECTION_ID].daemon.confirm_pairing = (  # type: ignore[attr-defined]
        _async_return(
            {
                "status": "success",
                "jid": "16268643289@s.whatsapp.net",
                "push_name": "Mallory",
            }
        )
    )
    unpair = _async_return(None)
    connector.state[CONNECTION_ID].daemon.unpair = unpair  # type: ignore[attr-defined]
    result = await connector.confirmPairing(external_account_id=PHONE)
    unpair.assert_awaited_once()  # type: ignore[attr-defined]
    assert result["status"] == "error"
    assert result["jid"] == "16268643289@s.whatsapp.net"
    assert "16268643289" in result["reason"]
    assert "15551112222" in result["reason"]
    assert "unpaired" in result["reason"]
    assert "push_name" not in result


async def test_confirm_pairing_match_returns_success_without_unpair(
    connector: WhatsappConnector,
) -> None:
    """The scanned account matches the connection — pass through untouched,
    and never call unpair on the live device."""
    connector.state[CONNECTION_ID].daemon.confirm_pairing = (  # type: ignore[attr-defined]
        _async_return(
            {
                "status": "success",
                "jid": "15551112222@s.whatsapp.net",
                "push_name": "Bot",
            }
        )
    )
    unpair = _async_return(None)
    connector.state[CONNECTION_ID].daemon.unpair = unpair  # type: ignore[attr-defined]
    result = await connector.confirmPairing(external_account_id=PHONE)
    unpair.assert_not_awaited()  # type: ignore[attr-defined]
    assert result == {
        "external_account_id": PHONE,
        "status": "success",
        "jid": "15551112222@s.whatsapp.net",
        "push_name": "Bot",
    }


async def test_confirm_pairing_mismatch_unpair_failure_still_errors(
    connector: WhatsappConnector,
) -> None:
    """If auto-unpair itself fails, we still must not return success — the
    error response surfaces both the mismatch and the unpair-failure fact."""
    connector.state[CONNECTION_ID].daemon.confirm_pairing = (  # type: ignore[attr-defined]
        _async_return(
            {
                "status": "success",
                "jid": "16268643289@s.whatsapp.net",
                "push_name": "Mallory",
            }
        )
    )
    unpair = _async_raise(RuntimeError("rpc down"))
    connector.state[CONNECTION_ID].daemon.unpair = unpair  # type: ignore[attr-defined]
    result = await connector.confirmPairing(external_account_id=PHONE)
    unpair.assert_awaited_once()  # type: ignore[attr-defined]
    assert result["status"] == "error"
    assert result["status"] != "success"
    assert result["jid"] == "16268643289@s.whatsapp.net"
    assert "16268643289" in result["reason"]
    assert "15551112222" in result["reason"]
    assert "unpair" in result["reason"].lower()


@pytest.mark.parametrize(
    "jid",
    ["15551112222@s.whatsapp.net", "15551112222:12@s.whatsapp.net"],
)
async def test_confirm_pairing_accepts_both_jid_formats(
    connector: WhatsappConnector, jid: str
) -> None:
    """Both bare ``NNN@`` and device-suffixed ``NNN:D@`` jids are accepted
    when the phone part matches the connection."""
    connector.state[CONNECTION_ID].daemon.confirm_pairing = (  # type: ignore[attr-defined]
        _async_return({"status": "success", "jid": jid, "push_name": "Bot"})
    )
    unpair = _async_return(None)
    connector.state[CONNECTION_ID].daemon.unpair = unpair  # type: ignore[attr-defined]
    result = await connector.confirmPairing(external_account_id=PHONE)
    unpair.assert_not_awaited()  # type: ignore[attr-defined]
    assert result["status"] == "success"
    assert result["jid"] == jid


@pytest.mark.parametrize(
    "jid",
    ["16268643289@s.whatsapp.net", "16268643289:7@s.whatsapp.net"],
)
async def test_confirm_pairing_mismatch_detected_for_both_jid_formats(
    connector: WhatsappConnector, jid: str
) -> None:
    """A mismatch is caught for both bare and device-suffixed jid formats."""
    connector.state[CONNECTION_ID].daemon.confirm_pairing = (  # type: ignore[attr-defined]
        _async_return({"status": "success", "jid": jid})
    )
    unpair = _async_return(None)
    connector.state[CONNECTION_ID].daemon.unpair = unpair  # type: ignore[attr-defined]
    result = await connector.confirmPairing(external_account_id=PHONE)
    unpair.assert_awaited_once()  # type: ignore[attr-defined]
    assert result["status"] == "error"
    assert result["jid"] == jid
    assert "16268643289" in result["reason"]


@pytest.mark.parametrize(
    ("input_phone", "expected"),
    [
        ("+15551112222", "+15551112222"),
        ("15551112222", "+15551112222"),
        ("+1 555 111 2222", "+15551112222"),
        ("+1-555-111-2222", "+15551112222"),
        ("  +15551112222  ", "+15551112222"),
        ("1-555-111-2222", "+15551112222"),
        ("", ""),
    ],
)
def test_normalize_phone(input_phone: str, expected: str) -> None:
    assert normalize_phone(input_phone) == expected


async def test_state_lookup_matches_formatting_variants(
    connector: WhatsappConnector,
) -> None:
    """Operator-supplied external_account_id need only match the connection's
    phone after normalization — formatting differences (spaces, dashes,
    missing leading +) are tolerated."""
    connector.state[CONNECTION_ID].daemon.start_pairing = (  # type: ignore[attr-defined]
        _async_return({"code": "2@code"})
    )
    # PHONE is "+15551112222"; operator submits various equivalents.
    for variant in ("+15551112222", "15551112222", "+1-555-111-2222", "+1 555 111 2222"):
        result = await connector.startPairing(external_account_id=variant)
        assert result["code"] == "2@code"
        # The response echoes the operator's input verbatim, not the
        # canonical form — keeps the response shape diagnosable.
        assert result["external_account_id"] == variant


def _async_return(value: object) -> object:
    """Build an AsyncMock that returns ``value`` from a single call."""
    return AsyncMock(return_value=value)


def _async_raise(exc: BaseException) -> object:
    """Build an AsyncMock that raises ``exc`` when awaited."""
    return AsyncMock(side_effect=exc)
