"""Unit tests for pure helpers in daemon.py.

The subprocess + TCP integration is exercised by test_integration.py.
"""

from __future__ import annotations

from aios_signal.daemon import _find_account_uuid


def test_find_account_uuid_match() -> None:
    accounts = [
        {"number": "+15551111111", "uuid": "uuid-a"},
        {"number": "+15552222222", "uuid": "uuid-b"},
    ]
    assert _find_account_uuid(accounts, "+15552222222") == "uuid-b"


def test_find_account_uuid_normalizes_whitespace() -> None:
    accounts = [{"number": "  +15551234567 ", "uuid": "abc"}]
    assert _find_account_uuid(accounts, "+15551234567") == "abc"


def test_find_account_uuid_no_match() -> None:
    accounts = [{"number": "+15551111111", "uuid": "uuid-a"}]
    assert _find_account_uuid(accounts, "+15559999999") is None


def test_find_account_uuid_empty_list() -> None:
    assert _find_account_uuid([], "+15550000000") is None


def test_find_account_uuid_malformed_response() -> None:
    assert _find_account_uuid("not-a-list", "+15550000000") is None
    assert _find_account_uuid([{"no_number": True}], "+15550000000") is None
    assert _find_account_uuid([{"number": "+1", "uuid": ""}], "+1") is None
