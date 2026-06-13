"""Unit tests for the ``schedule_wake`` tool handler (#818).

``schedule_wake`` is now sugar over ``triggers_service.add_trigger``: it
emits a one-shot trigger whose action is ``wake_owner`` (in-worker
self-delivery), not a sandbox_command running ``tool wake_self``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aios.tools.schedule_wake import (
    ScheduleWakeArgumentError,
    _resolve_fire_at,
    _wake_content,
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
def mock_add_trigger(monkeypatch: Any) -> AsyncMock:
    """Replace the service add_trigger with an AsyncMock returning a stub echo
    so the handler can complete without touching Postgres.

    ``MagicMock(name=...)`` is a foot-gun — ``name`` is a reserved kwarg that
    names the mock object itself rather than binding ``.name`` to the value.
    We set ``.name`` as a regular attribute instead.
    """
    echo = MagicMock(id="trig_01STUB")
    echo.name = "wake-stub"
    mock = AsyncMock(return_value=echo)
    monkeypatch.setattr("aios.tools.schedule_wake.triggers_service.add_trigger", mock)
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
        assert before + timedelta(seconds=59) <= resolved <= after + timedelta(seconds=61)

    def test_iso_8601_at_resolves(self) -> None:
        future = (datetime.now(UTC) + timedelta(hours=1)).replace(microsecond=0)
        resolved = _resolve_fire_at({"at": future.isoformat()})
        assert resolved == future

    def test_natural_language_with_tz(self) -> None:
        resolved = _resolve_fire_at({"at": "in 30 minutes"})
        delta = (resolved - datetime.now(UTC)).total_seconds()
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

    def test_invalid_tz_rejected(self) -> None:
        # An unresolvable IANA name is a model-input error: it must raise the
        # client-class ScheduleWakeArgumentError (symmetric with the other
        # rejections), NOT escape as a raw pytz UnknownTimeZoneError — the
        # latter is a non-AiosError that the tool dispatcher classifies as a
        # server fault and punishes with session-container eviction.
        with pytest.raises(ScheduleWakeArgumentError, match="unknown timezone"):
            _resolve_fire_at({"at": "tomorrow at 9am", "tz": "Not/AZone"})

    def test_lowercase_tz_accepted(self) -> None:
        # pytz (dateparser's backend) is case-insensitive, so 'utc' is valid.
        # The guard must defer to dateparser, NOT pre-reject with a
        # case-sensitive stdlib ZoneInfo check — which raises on 'utc' under a
        # case-sensitive tzdb (e.g. the Linux deploy image), over-rejecting a
        # plausible model input.
        resolved = _resolve_fire_at({"at": "in 30 minutes", "tz": "utc"})
        assert resolved > datetime.now(UTC)

    def test_fixed_offset_tz_accepted(self) -> None:
        # dateparser accepts fixed-offset zone names (e.g. 'UTC+5') via its own
        # StaticTzInfo path before pytz; the guard must not over-reject them
        # (a stdlib ZoneInfo check raises ZoneInfoNotFoundError on 'UTC+5').
        resolved = _resolve_fire_at({"at": "in 30 minutes", "tz": "UTC+5"})
        assert resolved > datetime.now(UTC)

    def test_delay_above_max_rejected(self) -> None:
        one_year_seconds = 60 * 60 * 24 * 365
        with pytest.raises(ScheduleWakeArgumentError, match="exceeds the max allowed"):
            _resolve_fire_at({"delay_seconds": one_year_seconds})

    def test_delay_overflow_rejected(self) -> None:
        with pytest.raises(ScheduleWakeArgumentError, match="exceeds"):
            _resolve_fire_at({"delay_seconds": 10**18})

    def test_at_far_future_rejected(self) -> None:
        far_future = (datetime.now(UTC) + timedelta(days=365)).isoformat()
        with pytest.raises(ScheduleWakeArgumentError, match="exceeds the max allowed"):
            _resolve_fire_at({"at": far_future})


class TestWakeContent:
    def test_marker_is_byte_identical(self) -> None:
        # The delivered marker must match the prior sandbox_command path's
        # string exactly (#818 §8) — zero behavior change for the reader.
        assert (
            _wake_content("check the deploy")
            == "[Your scheduled wake fired. Reason: check the deploy]"
        )

    def test_embeds_reason_with_special_chars(self) -> None:
        content = _wake_content("it's $weird\nnewline")
        assert "it's $weird\nnewline" in content


class TestScheduleWakeHandler:
    async def test_valid_delay_creates_one_shot_wake_owner(
        self, mock_add_trigger: AsyncMock
    ) -> None:
        result = await schedule_wake_handler(
            "sess_01TEST",
            {"delay_seconds": 30, "reason": "check back later"},
        )

        mock_add_trigger.assert_awaited_once()
        assert mock_add_trigger.await_args is not None
        spec = mock_add_trigger.await_args.args[2]
        # Source is a one-shot; action is a wake_owner (in-worker delivery).
        assert spec.source.kind == "one_shot"
        assert spec.source.fire_at is not None
        assert spec.action.kind == "wake_owner"
        assert spec.action.content == "[Your scheduled wake fired. Reason: check back later]"
        assert spec.metadata == {"kind": "wake", "reason": "check back later"}
        assert spec.name.startswith("wake-")
        assert result["scheduled"] is True
        assert result["reason"] == "check back later"
        assert result["trigger_id"] == "trig_01STUB"
        assert "fire_at" in result

    async def test_valid_at_creates_one_shot_trigger(self, mock_add_trigger: AsyncMock) -> None:
        future = (datetime.now(UTC) + timedelta(hours=2)).isoformat()
        result = await schedule_wake_handler(
            "sess_01TEST",
            {"at": future, "reason": "two hours from now"},
        )
        mock_add_trigger.assert_awaited_once()
        assert result["scheduled"] is True

    async def test_missing_reason_rejects(self, mock_add_trigger: AsyncMock) -> None:
        with pytest.raises(ScheduleWakeArgumentError, match="reason"):
            await schedule_wake_handler("sess_01TEST", {"delay_seconds": 5})
        mock_add_trigger.assert_not_awaited()

    async def test_unparseable_at_rejects(self, mock_add_trigger: AsyncMock) -> None:
        with pytest.raises(ScheduleWakeArgumentError, match="could not parse"):
            await schedule_wake_handler(
                "sess_01TEST",
                {"at": "@@@ nonsense @@@", "reason": "x"},
            )
        mock_add_trigger.assert_not_awaited()
