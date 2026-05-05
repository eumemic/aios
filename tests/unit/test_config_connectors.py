"""Unit tests for the multi-instance ``connectors_enabled`` parser.

Covers :func:`parse_connector_entry` and the
:class:`Settings._validate_connectors_enabled` field validator that
backs ``AIOS_CONNECTORS_ENABLED`` env parsing.
"""

from __future__ import annotations

from pathlib import Path

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

    def test_empty_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AIOS_CONNECTORS_ENABLED", raising=False)
        s = Settings(_env_file=None, **_BASE_KWARGS)
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

    def test_csv_env_value_parses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``AIOS_CONNECTORS_ENABLED=signal:main,telegram:bot1`` must parse as CSV.

        Pydantic-settings v2 default-parses ``list[str]`` env vars as
        JSON; the ``NoDecode`` annotation + ``mode='before'`` validator
        on :class:`Settings` accepts CSV instead.  Without this, the
        documented operator format ``AIOS_CONNECTORS_ENABLED=a,b``
        crashes at startup with a ``SettingsError``.
        """
        monkeypatch.setenv("AIOS_API_KEY", _BASE_KWARGS["api_key"])
        monkeypatch.setenv("AIOS_VAULT_KEY", _BASE_KWARGS["vault_key"])
        monkeypatch.setenv("AIOS_DB_URL", _BASE_KWARGS["db_url"])
        monkeypatch.setenv("AIOS_CONNECTORS_ENABLED", "signal:main,telegram:bot1")
        s = Settings()
        assert s.connectors_enabled == ["signal:main", "telegram:bot1"]
        assert s.connector_instances() == [
            ConnectorInstance(connector="signal", instance="main"),
            ConnectorInstance(connector="telegram", instance="bot1"),
        ]

    def test_csv_env_handles_whitespace_and_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Whitespace around CSV entries is stripped; trailing commas drop empties."""
        monkeypatch.setenv("AIOS_API_KEY", _BASE_KWARGS["api_key"])
        monkeypatch.setenv("AIOS_VAULT_KEY", _BASE_KWARGS["vault_key"])
        monkeypatch.setenv("AIOS_DB_URL", _BASE_KWARGS["db_url"])
        monkeypatch.setenv("AIOS_CONNECTORS_ENABLED", " signal , telegram:bot1 , ")
        s = Settings()
        assert s.connectors_enabled == ["signal", "telegram:bot1"]


class TestConnectorsDirCloister:
    """``connectors_dir`` defaults to a per-instance cloister under
    ``~/.aios/instances/<instance_id>/connectors`` (#238).

    The cloister keeps two worktree dev instances running connectors
    concurrently from clobbering each other's spool databases at the
    legacy shared ``~/.aios/connectors`` path.  Operators that override
    ``AIOS_CONNECTORS_DIR`` explicitly keep their override.
    """

    def test_default_instance_resolves_to_default_cloister(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("AIOS_CONNECTORS_DIR", raising=False)
        monkeypatch.delenv("AIOS_INSTANCE_ID", raising=False)
        s = Settings(_env_file=None, **_BASE_KWARGS)
        assert s.instance_id == "default"
        assert s.connectors_dir == Path.home() / ".aios" / "instances" / "default" / "connectors"

    def test_custom_instance_id_resolves_per_instance_cloister(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("AIOS_CONNECTORS_DIR", raising=False)
        s = Settings(_env_file=None, instance_id="aios_dev_abc12345", **_BASE_KWARGS)
        assert (
            s.connectors_dir
            == Path.home() / ".aios" / "instances" / "aios_dev_abc12345" / "connectors"
        )

    def test_explicit_connectors_dir_override_wins(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """An explicit ``AIOS_CONNECTORS_DIR`` (env or kwarg) preserves the
        operator's override even with a non-default instance_id.  This
        is the escape hatch for ops that pre-bake connectors elsewhere.
        """
        monkeypatch.delenv("AIOS_CONNECTORS_DIR", raising=False)
        custom = tmp_path / "operator-chosen-dir"
        s = Settings(
            _env_file=None,
            instance_id="aios_dev_abc12345",
            connectors_dir=custom,
            **_BASE_KWARGS,
        )
        assert s.connectors_dir == custom

    def test_env_var_override_wins(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """``AIOS_CONNECTORS_DIR`` env var wins over the cloister default."""
        custom = tmp_path / "via-env"
        monkeypatch.setenv("AIOS_CONNECTORS_DIR", str(custom))
        monkeypatch.setenv("AIOS_API_KEY", _BASE_KWARGS["api_key"])
        monkeypatch.setenv("AIOS_VAULT_KEY", _BASE_KWARGS["vault_key"])
        monkeypatch.setenv("AIOS_DB_URL", _BASE_KWARGS["db_url"])
        monkeypatch.setenv("AIOS_INSTANCE_ID", "aios_dev_abc12345")
        s = Settings()
        assert s.connectors_dir == custom
