"""Unit tests for ``AccountConfig`` — per-account timezone validation.

The inheritance resolution (``resolve_effective_timezone``) and the within-JSONB
config merge are SQL-backed and live in ``tests/integration/test_account_config_db.py``;
the end-to-end ``received=`` rendering in a zone is exercised in ``test_context.py``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aios.models.accounts import AccountConfig


class TestAccountConfigTimezone:
    def test_accepts_valid_iana_zone(self) -> None:
        assert AccountConfig(timezone="America/Los_Angeles").timezone == "America/Los_Angeles"
        assert AccountConfig(timezone="UTC").timezone == "UTC"

    def test_unset_timezone_is_none(self) -> None:
        # Unset (or explicit null) means "inherit from the parent account".
        assert AccountConfig().timezone is None
        assert AccountConfig(timezone=None).timezone is None

    def test_rejects_unknown_zone(self) -> None:
        # Fail hard at config-set time so the render path never sees a bad zone.
        with pytest.raises(ValidationError):
            AccountConfig(timezone="Not/AZone")

    def test_rejects_extra_config_keys(self) -> None:
        # extra='forbid': an unknown config item is a 422, not silently dropped.
        with pytest.raises(ValidationError):
            AccountConfig.model_validate({"timezone": "UTC", "bogus": 1})
