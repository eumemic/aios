"""Tests for ``metadata.channel`` validation in ``append_user_message``."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import asyncpg
import pytest

from aios.errors import ValidationError
from aios.services.sessions import append_user_message


def _pool_with_conn() -> MagicMock:
    pool = MagicMock()
    conn = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    return pool


class TestAppendUserMessageChannelValidation:
    async def test_unknown_metadata_channel_raises(self) -> None:
        pool = _pool_with_conn()
        with (
            patch(
                "aios.services.sessions.queries.list_session_channels",
                new=AsyncMock(return_value=[]),
            ),
            pytest.raises(ValidationError, match="not a bound channel"),
        ):
            await append_user_message(
                cast("asyncpg.Pool[Any]", pool),
                "sess_x",
                "hi",
                metadata={"channel": "dev"},
            )

    async def test_known_metadata_channel_passes_validation(self) -> None:
        pool = _pool_with_conn()
        with (
            patch(
                "aios.services.sessions.queries.list_session_channels",
                new=AsyncMock(return_value=["signal/bot/chat"]),
            ),
            patch(
                "aios.services.sessions.queries.append_event",
                new=AsyncMock(return_value="event-fixture"),
            ),
            patch(
                "aios.services.sessions.queries.flip_idle_to_pending",
                new=AsyncMock(return_value=None),
            ),
        ):
            result = await append_user_message(
                cast("asyncpg.Pool[Any]", pool),
                "sess_x",
                "hi",
                metadata={"channel": "signal/bot/chat"},
            )
            assert result == "event-fixture"

    async def test_no_metadata_skips_channel_check(self) -> None:
        pool = _pool_with_conn()
        with (
            patch(
                "aios.services.sessions.queries.list_session_channels",
                new=AsyncMock(return_value=[]),
            ) as list_mock,
            patch(
                "aios.services.sessions.queries.append_event",
                new=AsyncMock(return_value="event-fixture"),
            ),
            patch(
                "aios.services.sessions.queries.flip_idle_to_pending",
                new=AsyncMock(return_value=None),
            ),
        ):
            await append_user_message(
                cast("asyncpg.Pool[Any]", pool),
                "sess_x",
                "hi",
            )
            list_mock.assert_not_awaited()
