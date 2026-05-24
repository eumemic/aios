"""Pydantic + helper validation for scheduled_tasks models (#636).

Pure in-memory: no Postgres, no Docker.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from aios.models.scheduled_tasks import (
    MAX_COMMAND_CHARS,
    MAX_NAME_CHARS,
    MAX_SCHEDULE_CHARS,
    MAX_SCHEDULED_TASKS_PER_SESSION,
    ScheduledTaskCreate,
    ScheduledTaskUpdate,
    compute_next_fire,
    validate_scheduled_tasks,
)
from aios.models.sessions import SessionCreate, SessionUpdate


def _spec(**overrides: object) -> ScheduledTaskCreate:
    base: dict[str, object] = {
        "name": "poll",
        "schedule": "*/5 * * * *",
        "command": "echo hi",
    }
    base.update(overrides)
    return ScheduledTaskCreate.model_validate(base)


class TestScheduledTaskCreateName:
    def test_accepts_simple(self) -> None:
        assert _spec(name="poll").name == "poll"

    def test_accepts_with_dashes_and_underscores(self) -> None:
        _spec(name="poll-x_y_z9")

    def test_rejects_leading_dash(self) -> None:
        with pytest.raises(ValidationError):
            _spec(name="-poll")

    def test_rejects_leading_underscore(self) -> None:
        with pytest.raises(ValidationError):
            _spec(name="_poll")

    def test_rejects_space(self) -> None:
        with pytest.raises(ValidationError):
            _spec(name="poll x")

    def test_rejects_slash(self) -> None:
        with pytest.raises(ValidationError):
            _spec(name="poll/x")

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValidationError):
            _spec(name="")

    def test_rejects_too_long(self) -> None:
        with pytest.raises(ValidationError):
            _spec(name="a" * (MAX_NAME_CHARS + 1))


class TestScheduledTaskCreateSchedule:
    @pytest.mark.parametrize(
        "schedule",
        [
            "* * * * *",
            "*/5 * * * *",
            "0 9 * * *",
            "0 9 * * 1-5",
            "30 2 1 * *",
        ],
    )
    def test_accepts_valid_cron(self, schedule: str) -> None:
        _spec(schedule=schedule)

    @pytest.mark.parametrize(
        "schedule",
        [
            "not a cron",
            "* * * *",  # 4 fields
            "60 * * * *",  # minute out of range
            "",
        ],
    )
    def test_rejects_invalid(self, schedule: str) -> None:
        with pytest.raises(ValidationError):
            _spec(schedule=schedule)

    def test_rejects_too_long(self) -> None:
        with pytest.raises(ValidationError):
            _spec(schedule="*/5 * * * *" + " " * MAX_SCHEDULE_CHARS)


class TestScheduledTaskCreateCommand:
    def test_rejects_empty(self) -> None:
        with pytest.raises(ValidationError):
            _spec(command="")

    def test_rejects_too_long(self) -> None:
        with pytest.raises(ValidationError):
            _spec(command="x" * (MAX_COMMAND_CHARS + 1))

    def test_accepts_at_limit(self) -> None:
        _spec(command="x" * MAX_COMMAND_CHARS)


class TestScheduledTaskCreateNumericFields:
    def test_timeout_lower_bound(self) -> None:
        _spec(timeout_seconds=1)
        with pytest.raises(ValidationError):
            _spec(timeout_seconds=0)

    def test_timeout_upper_bound(self) -> None:
        _spec(timeout_seconds=3600)
        with pytest.raises(ValidationError):
            _spec(timeout_seconds=3601)

    def test_max_output_bounds(self) -> None:
        _spec(max_output_bytes=1024)
        _spec(max_output_bytes=1_048_576)
        with pytest.raises(ValidationError):
            _spec(max_output_bytes=1023)
        with pytest.raises(ValidationError):
            _spec(max_output_bytes=1_048_577)

    def test_enabled_defaults_true(self) -> None:
        assert _spec().enabled is True


class TestScheduledTaskUpdate:
    def test_empty_payload_valid(self) -> None:
        # All fields optional — empty update means "no change anywhere."
        u = ScheduledTaskUpdate.model_validate({})
        assert u.schedule is None
        assert u.command is None
        assert u.enabled is None

    def test_validates_schedule_when_provided(self) -> None:
        ScheduledTaskUpdate.model_validate({"schedule": "*/10 * * * *"})
        with pytest.raises(ValidationError):
            ScheduledTaskUpdate.model_validate({"schedule": "not a cron"})

    def test_schedule_none_is_passthrough(self) -> None:
        u = ScheduledTaskUpdate.model_validate({"schedule": None})
        assert u.schedule is None

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            ScheduledTaskUpdate.model_validate({"name": "rename-attempt"})


class TestValidateScheduledTasks:
    def test_accepts_empty(self) -> None:
        validate_scheduled_tasks([])

    def test_accepts_distinct_names(self) -> None:
        validate_scheduled_tasks([_spec(name="a"), _spec(name="b")])

    def test_rejects_duplicate_names(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            validate_scheduled_tasks([_spec(name="a"), _spec(name="a")])


class TestComputeNextFire:
    def test_every_five_minutes_advances_one_slot(self) -> None:
        # 2026-01-01 00:01 UTC → next "*/5 * * * *" is 00:05 UTC.
        base = datetime(2026, 1, 1, 0, 1, 0, tzinfo=UTC)
        nxt = compute_next_fire("*/5 * * * *", base)
        assert nxt == datetime(2026, 1, 1, 0, 5, 0, tzinfo=UTC)

    def test_daily_9am(self) -> None:
        # On 2026-01-01 at 10:00 UTC, next "0 9 * * *" is the following day 09:00.
        base = datetime(2026, 1, 1, 10, 0, 0, tzinfo=UTC)
        nxt = compute_next_fire("0 9 * * *", base)
        assert nxt == datetime(2026, 1, 2, 9, 0, 0, tzinfo=UTC)

    def test_returns_tz_aware_when_from_time_is_tz_aware(self) -> None:
        base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        nxt = compute_next_fire("* * * * *", base)
        assert nxt.tzinfo is not None

    def test_strictly_after_from_time(self) -> None:
        # On a minute boundary, next_fire must be strictly later, not equal.
        base = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
        nxt = compute_next_fire("* * * * *", base)
        assert nxt > base


class TestSessionCreateScheduledTasks:
    def _session_payload(self, **overrides: object) -> dict[str, object]:
        body: dict[str, object] = {
            "agent_id": "agent_01HQR2K7VXBZ9MNPL3WYCT8FAA",
            "environment_id": "env_01HQR2K7VXBZ9MNPL3WYCT8FAA",
        }
        body.update(overrides)
        return body

    def test_default_empty(self) -> None:
        sc = SessionCreate.model_validate(self._session_payload())
        assert sc.scheduled_tasks == []

    def test_accepts_initial_list(self) -> None:
        sc = SessionCreate.model_validate(
            self._session_payload(
                scheduled_tasks=[
                    {"name": "a", "schedule": "* * * * *", "command": "true"},
                    {"name": "b", "schedule": "0 9 * * *", "command": "echo b"},
                ]
            )
        )
        assert len(sc.scheduled_tasks) == 2

    def test_rejects_duplicate_names_at_create(self) -> None:
        with pytest.raises(ValidationError):
            SessionCreate.model_validate(
                self._session_payload(
                    scheduled_tasks=[
                        {"name": "dup", "schedule": "* * * * *", "command": "true"},
                        {"name": "dup", "schedule": "* * * * *", "command": "true"},
                    ]
                )
            )

    def test_rejects_over_cap(self) -> None:
        too_many = [
            {"name": f"t{i}", "schedule": "* * * * *", "command": "true"}
            for i in range(MAX_SCHEDULED_TASKS_PER_SESSION + 1)
        ]
        with pytest.raises(ValidationError):
            SessionCreate.model_validate(self._session_payload(scheduled_tasks=too_many))

    def test_session_update_rejects_scheduled_tasks_field(self) -> None:
        # Granular ops only (#270). SessionUpdate must not accept the field.
        with pytest.raises(ValidationError):
            SessionUpdate.model_validate(
                {"scheduled_tasks": [{"name": "x", "schedule": "* * * * *", "command": "true"}]}
            )
