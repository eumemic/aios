from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

from aios.api.routers.sessions import get_egress
from aios.models.sessions import SessionEgressHost, SessionEgressResponse


async def test_get_egress_returns_live_metadata_only() -> None:
    expected = SessionEgressResponse(
        hosts=[
            SessionEgressHost(
                host="api.mailgun.com",
                intercepted=True,
                source_credential_id="vcred_mailgun",
                secret_name="MAILGUN_API_KEY",
            )
        ],
        provisioned_at=datetime(2026, 7, 23, tzinfo=UTC),
        sandbox_generation=7,
    )
    pool = object()
    query = AsyncMock(return_value=expected)

    with patch("aios.api.routers.sessions.service.get_session_egress", query):
        assert await get_egress("sess_1", pool, "acct_1") == expected

    query.assert_awaited_once_with(pool, "sess_1", account_id="acct_1")
    assert "secret_value" not in expected.model_dump_json()
    assert "placeholder" not in expected.model_dump_json()
