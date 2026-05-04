"""Unit tests for the multi-instance ``connectors_enabled`` parser.

Covers :func:`parse_connector_entry` and the
:class:`Settings._validate_connectors_enabled` field validator that
backs ``AIOS_CONNECTORS_ENABLED`` env parsing.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aios.config import (
    ConnectorInstance,
    Settings,
    parse_connector_entry,
)

# Settings has required fields (api_key, vault_key, db_url); fixtures
# fill them with throwaway values so the validator tests can construct
# instances without env state.
_BASE_KWARGS = {
    "api_key": "test-key",
    "vault_key": "dGVzdC12YXVsdC1rZXktMzItYnl0ZXMtdGVzdC0=",
    "db_url": "postgresql://test:test@localhost/test",
}


class TestParseConnectorEntry:
    """``parse_connector_entry`` shape matrix."""

    def test_default_instance_when_colon_omitted(self) -> None:
        result = parse_connector_entry("signal")
        assert result == ConnectorInstance(connector="signal", instance="signal")

    def test_explicit_instance(self) -> None:
        result = parse_connector_entry("telegram:support")
        assert result == ConnectorInstance(connector="telegram", instance="support")

    def test_underscore_in_names(self) -> None:
        result = parse_connector_entry("my_connector:bot_one")
        assert result == ConnectorInstance(connector="my_connector", instance="bot_one")

    def test_digit_in_names(self) -> None:
        result = parse_connector_entry("telegram:bot1")
        assert result.instance == "bot1"

    def test_uppercase_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"connector name 'Signal'"):
            parse_connector_entry("Signal")

    def test_uppercase_in_instance_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"instance name 'Support'"):
            parse_connector_entry("telegram:Support")

    def test_hyphen_rejected(self) -> None:
        # Hyphens would corrupt the AIOS_<INSTANCE_UPPER>_* env-var re-export.
        with pytest.raises(ValueError, match=r"instance name 'support-bot'"):
            parse_connector_entry("telegram:support-bot")

    def test_leading_digit_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"instance name '1bot'"):
            parse_connector_entry("telegram:1bot")

    def test_empty_instance_after_colon_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"instance name ''"):
            parse_connector_entry("telegram:")

    def test_slash_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"instance name 'a/b'"):
            parse_connector_entry("telegram:a/b")


class TestConnectorsEnabledValidator:
    """The Settings field validator wraps the parser per entry."""

    def test_empty_default(self) -> None:
        s = Settings(**_BASE_KWARGS)
        assert s.connectors_enabled == []
        assert s.connector_instances() == []

    def test_default_instance_passes(self) -> None:
        s = Settings(connectors_enabled=["signal"], **_BASE_KWARGS)
        assert s.connector_instances() == [ConnectorInstance(connector="signal", instance="signal")]

    def test_mixed_default_and_explicit_instances(self) -> None:
        s = Settings(
            connectors_enabled=["signal:main", "telegram:support", "telegram:alerts"],
            **_BASE_KWARGS,
        )
        parsed = s.connector_instances()
        assert parsed == [
            ConnectorInstance(connector="signal", instance="main"),
            ConnectorInstance(connector="telegram", instance="support"),
            ConnectorInstance(connector="telegram", instance="alerts"),
        ]

    def test_duplicate_instances_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"duplicate connector instance signal:main"):
            Settings(connectors_enabled=["signal:main", "signal:main"], **_BASE_KWARGS)

    def test_default_instance_collides_with_same_explicit(self) -> None:
        # signal == signal:signal; if both are listed it's a duplicate.
        with pytest.raises(ValidationError, match=r"duplicate connector instance signal:signal"):
            Settings(connectors_enabled=["signal", "signal:signal"], **_BASE_KWARGS)

    def test_invalid_entry_surfaces_through_validator(self) -> None:
        with pytest.raises(ValidationError, match=r"connector name 'Bad'"):
            Settings(connectors_enabled=["Bad"], **_BASE_KWARGS)
