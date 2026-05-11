"""Unit tests for connector_token models (#301)."""

from __future__ import annotations

from datetime import datetime

import pytest

from aios.models.connector_tokens import (
    ConnectorToken,
    ConnectorTokenIssue,
    ConnectorTokenIssued,
)


class TestConnectorTokenIssue:
    def test_minimal(self) -> None:
        body = ConnectorTokenIssue(connection_id="conn_1")
        assert body.connection_id == "conn_1"
        assert body.label is None

    def test_with_label(self) -> None:
        body = ConnectorTokenIssue(connection_id="conn_1", label="prod-bot")
        assert body.label == "prod-bot"

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValueError):
            ConnectorTokenIssue.model_validate({"connection_id": "conn_1", "secret": "leaked"})

    def test_rejects_overlong_label(self) -> None:
        with pytest.raises(ValueError):
            ConnectorTokenIssue(connection_id="conn_1", label="x" * 129)


class TestConnectorTokenReadView:
    def test_no_plaintext_field(self) -> None:
        """The read view must NEVER carry plaintext — that's the contract."""
        assert "plaintext" not in ConnectorToken.model_fields

    def test_round_trip(self) -> None:
        now = datetime(2026, 5, 8)
        t = ConnectorToken(
            id="ctok_1",
            connection_id="conn_1",
            label="prod",
            created_at=now,
        )
        assert t.last_used_at is None
        assert t.revoked_at is None


class TestConnectorTokenIssued:
    def test_carries_plaintext(self) -> None:
        """The issue response is the ONLY surface that exposes plaintext."""
        now = datetime(2026, 5, 8)
        issued = ConnectorTokenIssued(
            id="ctok_1",
            connection_id="conn_1",
            label=None,
            plaintext="aios_conn_xxxxx",
            created_at=now,
        )
        assert issued.plaintext == "aios_conn_xxxxx"
