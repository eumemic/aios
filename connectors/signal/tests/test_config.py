"""Tests for signal connector's pydantic Settings.

The ``phones`` field accepts CSV from ``AIOS_SIGNAL_PHONES`` env vars
via ``Annotated[..., NoDecode]`` + ``mode='before'`` validator;
without these, pydantic-settings v2 tries to JSON-parse the env value
and crashes the connector at startup with a ``SettingsError``.
"""

from __future__ import annotations

import pytest

from aios_signal.config import Settings


def test_csv_env_value_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIOS_SIGNAL_PHONES", "+15551234567,+15559876543")
    monkeypatch.setenv("AIOS_SIGNAL_CONFIG_DIR", "/tmp/aios-signal-test")
    s = Settings()
    assert s.phones == ["+15551234567", "+15559876543"]


def test_csv_env_handles_whitespace_and_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIOS_SIGNAL_PHONES", " +15551234567 , +15559876543 , ")
    monkeypatch.setenv("AIOS_SIGNAL_CONFIG_DIR", "/tmp/aios-signal-test")
    s = Settings()
    assert s.phones == ["+15551234567", "+15559876543"]


def test_single_phone_csv_works(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIOS_SIGNAL_PHONES", "+15551234567")
    monkeypatch.setenv("AIOS_SIGNAL_CONFIG_DIR", "/tmp/aios-signal-test")
    s = Settings()
    assert s.phones == ["+15551234567"]


def test_constructor_kwarg_takes_list_directly() -> None:
    """Constructor kwargs (used by tests) bypass the CSV-split path."""
    s = Settings(phones=["+15551234567"], config_dir="/tmp/aios-signal-test")  # type: ignore[arg-type]
    assert s.phones == ["+15551234567"]
