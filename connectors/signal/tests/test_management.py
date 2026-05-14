"""Unit tests for :mod:`aios_signal.management` (#348).

The mixin's only domain-specific logic is the captcha-required predicate:
signal-cli surfaces captcha-required as an RPC error whose ``data``
field carries the challenge.  ``_is_captcha_required`` must catch both
the structured signal (a known error code, captcha keyword in data) and
the loose signal (the word ``captcha`` in the message string), so the
handler stays robust across signal-cli versions.

The plain happy / verify / updateProfile paths are exercised here too,
asserting the daemon wrappers get called with the right kwargs and the
operator-facing result shapes are returned verbatim.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from aios_connector_http import ManagementHandlerError

from aios_signal.errors import RpcError
from aios_signal.management import SignalManagementMixin, _is_captcha_required


class _Probe(SignalManagementMixin):
    """Minimal host that satisfies the mixin's ``self._daemon`` contract."""

    def __init__(self, daemon: Any) -> None:
        self._daemon = daemon


@pytest.fixture
def daemon() -> MagicMock:
    d = MagicMock()
    d.register = AsyncMock()
    d.verify = AsyncMock(return_value={"uuid": "u-abc"})
    d.update_profile = AsyncMock()
    return d


@pytest.fixture
def probe(daemon: MagicMock) -> _Probe:
    return _Probe(daemon)


class TestRegister:
    @pytest.mark.asyncio
    async def test_happy_path_returns_sms_sent(self, probe: _Probe, daemon: MagicMock) -> None:
        result = await probe.register(account="+15551234567")
        assert result == {"account": "+15551234567", "status": "sms_sent"}
        daemon.register.assert_awaited_once_with(phone="+15551234567", captcha=None, voice=False)

    @pytest.mark.asyncio
    async def test_voice_routes_to_voice_sent(self, probe: _Probe, daemon: MagicMock) -> None:
        result = await probe.register(account="+1", voice=True)
        assert result["status"] == "voice_sent"
        daemon.register.assert_awaited_once_with(phone="+1", captcha=None, voice=True)

    @pytest.mark.asyncio
    async def test_passes_captcha_token_to_daemon(self, probe: _Probe, daemon: MagicMock) -> None:
        await probe.register(account="+1", captcha="signalcaptcha://x")
        daemon.register.assert_awaited_once_with(
            phone="+1", captcha="signalcaptcha://x", voice=False
        )

    @pytest.mark.asyncio
    async def test_captcha_required_becomes_structured_error(
        self, probe: _Probe, daemon: MagicMock
    ) -> None:
        daemon.register.side_effect = RpcError("Captcha required", code=-3, data={"captcha": True})
        with pytest.raises(ManagementHandlerError) as exc_info:
            await probe.register(account="+1")
        assert exc_info.value.payload == {
            "status": "captcha_required",
            "captcha_url": "https://signalcaptchas.org/registration/generate",
            "account": "+1",
        }

    @pytest.mark.asyncio
    async def test_non_captcha_rpc_error_propagates(self, probe: _Probe, daemon: MagicMock) -> None:
        daemon.register.side_effect = RpcError("phone already registered elsewhere")
        with pytest.raises(RpcError, match="already registered"):
            await probe.register(account="+1")


class TestVerify:
    @pytest.mark.asyncio
    async def test_returns_uuid(self, probe: _Probe, daemon: MagicMock) -> None:
        result = await probe.verify(account="+1", code="123456")
        assert result == {"account": "+1", "uuid": "u-abc"}
        daemon.verify.assert_awaited_once_with(phone="+1", code="123456", pin=None)

    @pytest.mark.asyncio
    async def test_pin_threaded_through(self, probe: _Probe, daemon: MagicMock) -> None:
        await probe.verify(account="+1", code="000000", pin="9999")
        daemon.verify.assert_awaited_once_with(phone="+1", code="000000", pin="9999")


class TestUpdateProfile:
    @pytest.mark.asyncio
    async def test_passes_only_non_none_fields(self, probe: _Probe, daemon: MagicMock) -> None:
        await probe.update_profile(account="+1", given_name="Alice", about=None)
        daemon.update_profile.assert_awaited_once_with(
            phone="+1", given_name="Alice", family_name=None, about=None
        )


class TestCaptchaPredicate:
    def test_known_code(self) -> None:
        assert _is_captcha_required(RpcError("denied", code=-3))

    def test_captcha_keyword_in_data(self) -> None:
        assert _is_captcha_required(RpcError("denied", data={"captcha_token": "x"}))

    def test_captcha_keyword_in_message(self) -> None:
        assert _is_captcha_required(RpcError("Captcha required"))

    def test_unrelated_error_returns_false(self) -> None:
        assert not _is_captcha_required(RpcError("rate limited", code=-100))


class TestRequiresDaemon:
    @pytest.mark.asyncio
    async def test_register_without_daemon_raises(self) -> None:
        probe = _Probe(None)
        with pytest.raises(RuntimeError, match="daemon not started"):
            await probe.register(account="+1")
