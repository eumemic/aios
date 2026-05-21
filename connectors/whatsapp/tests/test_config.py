"""Tests for the whatsapp connector's pydantic Settings.

The phone (account identity) is not in env — it lives on the
connection record's encrypted secrets blob and arrives per-connection
in ``serve_connection``.  Settings carries only deployment-shape
fields: the data dir, the daemon binary, and the loopback bind.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aios_whatsapp.config import Settings


def test_data_dir_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AIOS_WHATSAPP_DATA_DIR", raising=False)
    with pytest.raises(ValidationError):
        Settings()


def test_constructor_kwarg() -> None:
    s = Settings(data_dir="/tmp/aios-whatsapp-test")
    assert s.data_dir.as_posix() == "/tmp/aios-whatsapp-test"


def test_defaults_for_optional_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIOS_WHATSAPP_DATA_DIR", "/tmp/aios-whatsapp-test")
    s = Settings()
    assert s.daemon_bin == "whatsapp-daemon"
    assert s.daemon_host == "127.0.0.1"
    assert s.daemon_port == 7584


def test_env_override_for_daemon_bin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIOS_WHATSAPP_DATA_DIR", "/tmp/aios-whatsapp-test")
    monkeypatch.setenv("AIOS_WHATSAPP_DAEMON_BIN", "/usr/local/bin/whatsapp-daemon-staging")
    s = Settings()
    assert s.daemon_bin == "/usr/local/bin/whatsapp-daemon-staging"
