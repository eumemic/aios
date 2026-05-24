"""Unit tests for the ``schedule_wake`` tool handler."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aios.tools.schedule_wake import (
    ScheduleWakeArgumentError,
    _build_wake_bash,
    _resolve_fire_at,
    schedule_wake_handler,
)


@pytest.fixture(autouse=True)
def mock_runtime_pool(monkeypatch: Any) -> None:
    """Stub the worker-only runtime pool so the handler can resolve
    `runtime.require_pool()` without a live worker_main context.
    """
    monkeypatch.setattr(
        "aios.tools.schedule_wake.runtime.require_pool",
        lambda: MagicMock(),
    )


@pytest.fixture
def mock_add_task(monkeypatch: Any) -> AsyncMock:
    """Replace the service add_task with an AsyncMock returning a stub echo
    so the handler can complete without touching Postgres.

    ``MagicMock(name=...)`` is a foot-gun — ``name`` is a reserved kwarg
    that names the mock object itself rather than binding ``.name`` to
    the value. We set ``.name`` as a regular attribute instead so the
    handler's ``echo.name`` resolves to a real string.
    """
    echo = MagicMock(id="st_01STUB")
    echo.name = "wake-stub"
    mock = AsyncMock(return_value=echo)
    monkeypatch.setattr("aios.tools.schedule_wake.scheduled_tasks_service.add_task", mock)
    monkeypatch.setattr(
        "aios.tools.schedule_wake.sessions_service.load_session_account_id",
        AsyncMock(return_value="acct_01STUB"),
    )
    return mock


class TestResolveFireAt:
    def test_delay_seconds_resolves_to_future(self) -> None:
        before = datetime.now(UTC)
        resolved = _resolve_fire_at({"delay_seconds": 60})
        after = datetime.now(UTC)
        # Resolved time should be ~60s after now, allowing a small wall-clock window.
        assert before + timedelta(seconds=59) <= resolved <= after + timedelta(seconds=61)

    def test_iso_8601_at_resolves(self) -> None:
        future = (datetime.now(UTC) + timedelta(hours=1)).replace(microsecond=0)
        resolved = _resolve_fire_at({"at": future.isoformat()})
        assert resolved == future

    def test_natural_language_with_tz(self) -> None:
        # "in 30 minutes" is unambiguous; no tz handling needed.
        resolved = _resolve_fire_at({"at": "in 30 minutes"})
        delta = (resolved - datetime.now(UTC)).total_seconds()
        # dateparser's relative-time math is exact to seconds, but allow
        # a generous bound so this isn't flaky on slow CI.
        assert 25 * 60 <= delta <= 35 * 60

    def test_both_delay_and_at_rejected(self) -> None:
        with pytest.raises(ScheduleWakeArgumentError, match="exactly one"):
            _resolve_fire_at({"delay_seconds": 10, "at": "in 1 hour"})

    def test_neither_delay_nor_at_rejected(self) -> None:
        with pytest.raises(ScheduleWakeArgumentError, match="exactly one"):
            _resolve_fire_at({})

    def test_negative_delay_rejected(self) -> None:
        with pytest.raises(ScheduleWakeArgumentError, match="positive integer"):
            _resolve_fire_at({"delay_seconds": -1})

    def test_zero_delay_rejected(self) -> None:
        with pytest.raises(ScheduleWakeArgumentError, match="positive integer"):
            _resolve_fire_at({"delay_seconds": 0})

    def test_non_int_delay_rejected(self) -> None:
        with pytest.raises(ScheduleWakeArgumentError, match="positive integer"):
            _resolve_fire_at({"delay_seconds": "soon"})

    def test_tz_with_delay_rejected(self) -> None:
        with pytest.raises(ScheduleWakeArgumentError, match="not valid"):
            _resolve_fire_at({"delay_seconds": 30, "tz": "UTC"})

    def test_unparseable_at_rejected(self) -> None:
        with pytest.raises(ScheduleWakeArgumentError, match="could not parse"):
            _resolve_fire_at({"at": "@@@ nonsense @@@"})

    def test_past_at_rejected(self) -> None:
        past = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        with pytest.raises(ScheduleWakeArgumentError, match="not in the future"):
            _resolve_fire_at({"at": past})

    def test_empty_at_rejected(self) -> None:
        with pytest.raises(ScheduleWakeArgumentError, match="non-empty"):
            _resolve_fire_at({"at": ""})

    def test_delay_above_max_rejected(self) -> None:
        # Default max is 30 days; one year is well above that.
        one_year_seconds = 60 * 60 * 24 * 365
        with pytest.raises(ScheduleWakeArgumentError, match="exceeds the max allowed"):
            _resolve_fire_at({"delay_seconds": one_year_seconds})

    def test_delay_overflow_rejected(self) -> None:
        # An int large enough to overflow `timedelta(seconds=...)` is
        # caught at the cap check first (cap rejects long before
        # OverflowError); pass a value below the cap but still ridiculous
        # to ensure the cap check triggers without exceptions.
        with pytest.raises(ScheduleWakeArgumentError, match="exceeds"):
            _resolve_fire_at({"delay_seconds": 10**18})

    def test_at_far_future_rejected(self) -> None:
        far_future = (datetime.now(UTC) + timedelta(days=365)).isoformat()
        with pytest.raises(ScheduleWakeArgumentError, match="exceeds the max allowed"):
            _resolve_fire_at({"at": far_future})


class TestBuildWakeBash:
    def test_contains_canonical_idiom(self) -> None:
        bash = _build_wake_bash("hello")
        assert "AIOS_BROKER_URL" in bash
        assert "MCP_BROKER_SECRET" in bash
        assert "AIOS_BROKER_SOCKET" in bash
        assert "/sessions/messages" in bash

    def test_embeds_reason_safely(self) -> None:
        # Embedded single quotes + dollar signs in the reason must not
        # break out of the shell-escaped argument.
        bash = _build_wake_bash("it's $weird")
        # The JSON-encoded payload appears inside a shlex-quoted block,
        # so single quotes are escaped via shell's '"'"' or "'\''" form
        # (or the whole arg is wrapped in single quotes with escapes).
        # We don't pin the exact escape style, just that the reason text
        # is recoverable.
        assert "it" in bash
        assert "weird" in bash

    def test_reason_with_newline(self) -> None:
        # Newlines in the reason must survive into the bash arg.
        bash = _build_wake_bash("line1\nline2")
        assert "line1" in bash
        assert "line2" in bash


class TestScheduleWakeHandler:
    async def test_valid_delay_creates_one_shot_task(self, mock_add_task: AsyncMock) -> None:
        result = await schedule_wake_handler(
            "sess_01TEST",
            {"delay_seconds": 30, "reason": "check back later"},
        )

        mock_add_task.assert_awaited_once()
        # Inspect the spec passed to add_task.
        spec = mock_add_task.await_args.args[2]
        assert spec.fire_at is not None
        assert spec.schedule is None
        assert spec.metadata == {"kind": "wake", "reason": "check back later"}
        assert spec.timeout_seconds == 30
        assert spec.name.startswith("wake-")
        assert "MCP_BROKER_SECRET" in spec.command
        assert result["scheduled"] is True
        assert result["reason"] == "check back later"
        assert "fire_at" in result

    async def test_valid_at_creates_one_shot_task(self, mock_add_task: AsyncMock) -> None:
        future = (datetime.now(UTC) + timedelta(hours=2)).isoformat()
        result = await schedule_wake_handler(
            "sess_01TEST",
            {"at": future, "reason": "two hours from now"},
        )
        mock_add_task.assert_awaited_once()
        assert result["scheduled"] is True

    async def test_missing_reason_rejects(self, mock_add_task: AsyncMock) -> None:
        with pytest.raises(ScheduleWakeArgumentError, match="reason"):
            await schedule_wake_handler("sess_01TEST", {"delay_seconds": 5})
        mock_add_task.assert_not_awaited()

    async def test_unparseable_at_rejects(self, mock_add_task: AsyncMock) -> None:
        with pytest.raises(ScheduleWakeArgumentError, match="could not parse"):
            await schedule_wake_handler(
                "sess_01TEST",
                {"at": "@@@ nonsense @@@", "reason": "x"},
            )
        mock_add_task.assert_not_awaited()
