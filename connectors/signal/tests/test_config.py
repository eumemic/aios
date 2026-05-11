"""Tests for signal connector's pydantic Settings.

The phone (account identity) is no longer in env — it lives on the
connection record's encrypted secrets blob and is fetched at
``setup()`` time.  Settings now carries only deployment-shape fields:
the host filesystem path to the signal-cli config dir, the binary,
and the local TCP daemon address.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aios_signal.config import Settings


def test_config_dir_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AIOS_SIGNAL_CONFIG_DIR", raising=False)
    with pytest.raises(ValidationError):
        Settings()


def test_constructor_kwarg() -> None:
    s = Settings(config_dir="/tmp/aios-signal-test")
    assert s.config_dir.as_posix() == "/tmp/aios-signal-test"


def test_defaults_for_optional_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIOS_SIGNAL_CONFIG_DIR", "/tmp/aios-signal-test")
    s = Settings()
    assert s.cli_bin == "signal-cli"
    assert s.daemon_host == "127.0.0.1"
    assert s.daemon_port == 7583
