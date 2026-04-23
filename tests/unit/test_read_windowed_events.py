"""Unit tests for the correction-applied drop math in ``read_windowed_events``.

Issue #158: ``window_min`` / ``window_max`` are operator-facing budgets on
provider tokens, but ``cumulative_tokens`` is stored in the local tokenizer's
space (systematically low by ~34% on Anthropic). The reader now applies a
correction factor derived from the most recent successful turn.

These tests patch the lookup helpers and assert that the SQL range predicate
sees a ``drop_local`` translated from ``drop_provider`` — i.e., the windowing
decision happens in provider space but the column comparison stays in local
space.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.db import queries


@pytest.fixture
def conn() -> Any:
    c = MagicMock()
    c.fetch = AsyncMock(return_value=[])
    return c


class TestReadWindowedEventsCorrection:
    async def test_correction_grows_drop_boundary(self, conn: Any) -> None:
        """With correction=1.5, a session the local sum says is 300k lands at
        450k in provider space. window_max=400k → drop in provider space, then
        translated back to local space for the SQL predicate.
        """
        with (
            patch(
                "aios.db.queries._latest_cumulative_tokens",
                AsyncMock(return_value=300_000),
            ),
            patch(
                "aios.db.queries.recent_token_correction",
                AsyncMock(return_value=1.5),
            ),
        ):
            await queries.read_windowed_events(
                conn, "sess_x", window_min=300_000, window_max=400_000
            )

        conn.fetch.assert_awaited_once()
        drop_local = conn.fetch.await_args.args[2]
        # provider total = 300k * 1.5 = 450k; over window_max by 50k; one chunk
        # of (max-min)=100k gets dropped → drop_provider = 100k.
        # drop_local = 100k / 1.5 ≈ 66,666.
        assert drop_local == pytest.approx(66_666, abs=2)

    async def test_identity_correction_matches_legacy_behavior(self, conn: Any) -> None:
        """correction=1.0 → drop_local == drop_provider — same as pre-#158."""
        with (
            patch(
                "aios.db.queries._latest_cumulative_tokens",
                AsyncMock(return_value=500_000),
            ),
            patch(
                "aios.db.queries.recent_token_correction",
                AsyncMock(return_value=1.0),
            ),
        ):
            await queries.read_windowed_events(
                conn, "sess_x", window_min=300_000, window_max=400_000
            )

        drop_local = conn.fetch.await_args.args[2]
        # provider total = 500k; over max by 100k; one chunk of 100k dropped.
        assert drop_local == 100_000

    async def test_under_budget_skips_range_scan(self, conn: Any) -> None:
        """No drop needed — full load path (same as pre-#158)."""
        with (
            patch(
                "aios.db.queries._latest_cumulative_tokens",
                AsyncMock(return_value=100_000),
            ),
            patch(
                "aios.db.queries.recent_token_correction",
                AsyncMock(return_value=1.5),
            ),
            patch(
                "aios.db.queries.read_message_events",
                AsyncMock(return_value=[]),
            ) as full_load,
        ):
            await queries.read_windowed_events(
                conn, "sess_x", window_min=300_000, window_max=400_000
            )
        full_load.assert_awaited_once()
        conn.fetch.assert_not_awaited()

    async def test_no_cumulative_data_falls_back_to_full_load(self, conn: Any) -> None:
        """Pre-backfill sessions — behavior unchanged."""
        with (
            patch(
                "aios.db.queries._latest_cumulative_tokens",
                AsyncMock(return_value=None),
            ),
            patch(
                "aios.db.queries.read_message_events",
                AsyncMock(return_value=[]),
            ) as full_load,
        ):
            await queries.read_windowed_events(
                conn, "sess_x", window_min=300_000, window_max=400_000
            )
        full_load.assert_awaited_once()
