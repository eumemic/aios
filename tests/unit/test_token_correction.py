"""Unit tests for ``recent_token_correction``.

The helper derives a provider-to-local tokenizer ratio from the most recent
successful model_request_end + its paired context_build_end. We unit-test the
Python branching (fallback to 1.0 when no calibrated pair is available, ratio
arithmetic when one is). SQL correctness — correct pairing, filtering of
errored/zero-token calls — is exercised via integration/e2e paths that have
a real Postgres.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aios.db.queries import recent_token_correction


@pytest.fixture
def conn() -> Any:
    c = MagicMock()
    c.fetchrow = AsyncMock()
    return c


class TestRecentTokenCorrection:
    async def test_no_prior_pair_returns_one(self, conn: Any) -> None:
        """First turn of a session, or no context_build_end has yet been
        stamped with local_token_estimate: behave as today."""
        conn.fetchrow.return_value = None
        assert await recent_token_correction(conn, "sess_x") == 1.0

    async def test_valid_pair_returns_ratio(self, conn: Any) -> None:
        conn.fetchrow.return_value = {"provider_tokens": 634052, "local_tokens": 417894}
        got = await recent_token_correction(conn, "sess_x")
        assert got == pytest.approx(634052 / 417894)

    async def test_identity_ratio_when_tokenizers_agree(self, conn: Any) -> None:
        conn.fetchrow.return_value = {"provider_tokens": 500, "local_tokens": 500}
        assert await recent_token_correction(conn, "sess_x") == 1.0
