"""Unit tests for MCP inbound service validation helpers."""

from __future__ import annotations

import pytest

from aios.errors import ValidationError
from aios.services.inbound import parse_account_relative_channel


def test_parse_account_relative_channel_returns_path() -> None:
    assert parse_account_relative_channel("acct/chat/1", account_id="acct") == "chat/1"


def test_parse_account_relative_channel_rejects_wrong_account() -> None:
    with pytest.raises(ValidationError, match="account does not match"):
        parse_account_relative_channel("other/chat", account_id="acct")


def test_parse_account_relative_channel_rejects_missing_path() -> None:
    with pytest.raises(ValidationError, match="<account>/<path>"):
        parse_account_relative_channel("acct", account_id="acct")
