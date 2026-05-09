"""Tests for signal connector's pydantic Settings.

In the new single-phone-per-container model (#301) ``phone`` is a
plain ``str`` — multi-phone deployments use multiple containers,
each with its own ``AIOS_SIGNAL_PHONE``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aios_signal.config import Settings


def test_phone_env_value_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIOS_SIGNAL_PHONE", "+15551234567")
    monkeypatch.setenv("AIOS_SIGNAL_CONFIG_DIR", "/tmp/aios-signal-test")
    s = Settings()
    assert s.phone == "+15551234567"


def test_constructor_kwarg() -> None:
    s = Settings(phone="+15551234567", config_dir="/tmp/aios-signal-test")
    assert s.phone == "+15551234567"


def test_phone_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AIOS_SIGNAL_PHONE", raising=False)
    monkeypatch.setenv("AIOS_SIGNAL_CONFIG_DIR", "/tmp/aios-signal-test")
    with pytest.raises(ValidationError):
        Settings()
