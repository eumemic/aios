"""Unit coverage for ``require_operator_or_connector_auth``.

The dual-auth dep is what guards ``POST /v1/sessions/<id>/files`` (#324),
so the dispatch table — operator → ``("operator", None)``, connector →
``("connector", connection_id)``, neither → 401 — has to be precise.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import asyncpg
import pytest

from aios.api.deps import require_operator_or_connector_auth
from aios.config import get_settings
from aios.errors import UnauthorizedError
from aios.services.connector_tokens import ResolvedToken


@pytest.fixture
def _settings(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Force a known operator key so the constant-time compare is exact."""
    s = get_settings()
    monkeypatch.setattr(s, "api_key", type(s.api_key)("operator-key-for-test"))
    return s


def _pool() -> Any:
    return cast("asyncpg.Pool[Any]", MagicMock())


class TestRequireOperatorOrConnectorAuth:
    async def test_operator_key_returns_operator_mode(self, _settings: Any) -> None:
        result = await require_operator_or_connector_auth(
            _settings,
            _pool(),
            authorization="Bearer operator-key-for-test",
        )
        assert result == ("operator", None)

    async def test_operator_key_short_circuits_before_db(self, _settings: Any) -> None:
        """Operator path must never hit the DB — otherwise every operator
        request pays a connector_tokens lookup."""
        with patch(
            "aios.api.deps.connector_tokens_service.resolve",
            AsyncMock(side_effect=RuntimeError("should not be called")),
        ):
            result = await require_operator_or_connector_auth(
                _settings,
                _pool(),
                authorization="Bearer operator-key-for-test",
            )
        assert result == ("operator", None)

    async def test_valid_connector_token_returns_connector_mode(self, _settings: Any) -> None:
        with patch(
            "aios.api.deps.connector_tokens_service.resolve",
            AsyncMock(return_value=ResolvedToken(token_id="ctok_x", connection_id="conn_abc")),
        ):
            result = await require_operator_or_connector_auth(
                _settings,
                _pool(),
                authorization="Bearer aios_conn_somecredential",
            )
        assert result == ("connector", "conn_abc")

    async def test_unknown_token_raises_401(self, _settings: Any) -> None:
        with (
            patch(
                "aios.api.deps.connector_tokens_service.resolve",
                AsyncMock(return_value=None),
            ),
            pytest.raises(UnauthorizedError) as excinfo,
        ):
            await require_operator_or_connector_auth(
                _settings,
                _pool(),
                authorization="Bearer aios_conn_bogus",
            )
        assert excinfo.value.status_code == 401

    async def test_missing_header_raises_401(self, _settings: Any) -> None:
        with pytest.raises(UnauthorizedError):
            await require_operator_or_connector_auth(_settings, _pool(), authorization=None)

    async def test_non_bearer_scheme_raises_401(self, _settings: Any) -> None:
        # ``_extract_bearer_token`` rejects anything that isn't ``Bearer …``.
        from fastapi import HTTPException

        with pytest.raises((UnauthorizedError, HTTPException)):
            await require_operator_or_connector_auth(
                _settings, _pool(), authorization="Basic dXNlcjpwYXNz"
            )
