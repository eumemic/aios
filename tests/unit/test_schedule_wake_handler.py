"""Unit tests for the ``schedule_wake`` tool handler."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from aios.tools.schedule_wake import (
    ScheduleWakeArgumentError,
    schedule_wake_handler,
)


class TestScheduleWakeHandler:
    async def test_valid_delay_calls_defer_wake(self, monkeypatch: Any) -> None:
        mock_defer = AsyncMock()
        monkeypatch.setattr("aios.tools.schedule_wake.defer_wake", mock_defer)

        result = await schedule_wake_handler(
            "sess_01TEST",
            {"delay_seconds": 30, "reason": "check back later"},
        )

        mock_defer.assert_awaited_once_with(
            "sess_01TEST",
            cause="scheduled",
            delay_seconds=30,
            wake_reason="check back later",
        )
        assert result["scheduled"] is True
        assert result["delay_seconds"] == 30
        assert result["reason"] == "check back later"

    async def test_delay_below_one_rejects(self, monkeypatch: Any) -> None:
        monkeypatch.setattr("aios.tools.schedule_wake.defer_wake", AsyncMock())

        with pytest.raises(ScheduleWakeArgumentError, match=r"1.*3600"):
            await schedule_wake_handler(
                "sess_01TEST",
                {"delay_seconds": 0, "reason": "bad"},
            )

    async def test_delay_above_one_hour_rejects(self, monkeypatch: Any) -> None:
        monkeypatch.setattr("aios.tools.schedule_wake.defer_wake", AsyncMock())

        with pytest.raises(ScheduleWakeArgumentError, match=r"1.*3600"):
            await schedule_wake_handler(
                "sess_01TEST",
                {"delay_seconds": 3601, "reason": "too long"},
            )

    async def test_non_integer_delay_rejects(self, monkeypatch: Any) -> None:
        monkeypatch.setattr("aios.tools.schedule_wake.defer_wake", AsyncMock())

        with pytest.raises(ScheduleWakeArgumentError, match="integer"):
            await schedule_wake_handler(
                "sess_01TEST",
                {"delay_seconds": "sleep_thirty", "reason": "bad"},
            )

    async def test_missing_reason_rejects(self, monkeypatch: Any) -> None:
        monkeypatch.setattr("aios.tools.schedule_wake.defer_wake", AsyncMock())

        with pytest.raises(ScheduleWakeArgumentError, match="reason"):
            await schedule_wake_handler("sess_01TEST", {"delay_seconds": 5})

    async def test_boundary_delays_accepted(self, monkeypatch: Any) -> None:
        """Exact boundaries (1s and 3600s) land on the accepted side."""
        mock_defer = AsyncMock()
        monkeypatch.setattr("aios.tools.schedule_wake.defer_wake", mock_defer)

        await schedule_wake_handler("sess_01TEST", {"delay_seconds": 1, "reason": "min"})
        await schedule_wake_handler("sess_01TEST", {"delay_seconds": 3600, "reason": "max"})

        assert mock_defer.await_count == 2
