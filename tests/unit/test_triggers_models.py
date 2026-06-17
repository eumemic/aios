"""Pydantic + helper validation for the triggers models (#818).

Pure in-memory: no Postgres, no Docker.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from aios.models.sessions import SessionCreate, SessionUpdate
from aios.models.triggers import (
    MAX_COMMAND_CHARS,
    MAX_NAME_CHARS,
    MAX_SCHEDULE_CHARS,
    MAX_TRIGGERS_PER_SESSION,
    MAX_WAKE_CONTENT_CHARS,
    TRIGGER_ACTION_ADAPTER,
    TRIGGER_SOURCE_ADAPTER,
    CronSource,
    OneShotSource,
    TriggerCreate,
    TriggerEcho,
    TriggerUpdate,
    WakeSessionAction,
    compute_initial_next_fire,
    compute_next_fire,
    validate_triggers,
)


def _spec(**overrides: object) -> TriggerCreate:
    base: dict[str, object] = {
        "name": "poll",
        "source": {"kind": "cron", "schedule": "*/5 * * * *"},
        "action": {"kind": "sandbox_command", "command": "echo hi"},
    }
    base.update(overrides)
    return TriggerCreate.model_validate(base)


class TestTriggerCreateName:
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


class TestCronSource:
    @pytest.mark.parametrize(
        "schedule",
        ["* * * * *", "*/5 * * * *", "0 9 * * *", "0 9 * * 1-5", "30 2 1 * *"],
    )
    def test_accepts_valid_cron(self, schedule: str) -> None:
        _spec(source={"kind": "cron", "schedule": schedule})

    @pytest.mark.parametrize(
        "schedule",
        ["not a cron", "* * * *", "60 * * * *", ""],
    )
    def test_rejects_invalid_grammar(self, schedule: str) -> None:
        with pytest.raises(ValidationError):
            _spec(source={"kind": "cron", "schedule": schedule})

    def test_rejects_too_long(self) -> None:
        with pytest.raises(ValidationError):
            _spec(source={"kind": "cron", "schedule": "*/5 * * * *" + " " * MAX_SCHEDULE_CHARS})

    def test_rejects_no_occurrence_within_horizon(self) -> None:
        # Grammar-valid but never fires (Feb 30 doesn't exist) — rejected at
        # write time by the 1-year occurrence horizon (#818 §7).
        with pytest.raises(ValidationError, match="no occurrence"):
            _spec(source={"kind": "cron", "schedule": "0 0 30 2 *"})


class TestOneShotSource:
    def test_accepts_tz_aware(self) -> None:
        future = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
        spec = _spec(source={"kind": "one_shot", "fire_at": future})
        assert isinstance(spec.source, OneShotSource)

    def test_past_fire_at_allowed_at_create(self) -> None:
        # Past fire_at is legal at create (fires immediately — today's semantic).
        past = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        _spec(source={"kind": "one_shot", "fire_at": past})

    def test_rejects_naive_datetime(self) -> None:
        with pytest.raises(ValidationError, match="timezone-aware"):
            _spec(source={"kind": "one_shot", "fire_at": "2026-06-11T09:00:00"})


class TestSandboxCommandAction:
    def test_rejects_empty_command(self) -> None:
        with pytest.raises(ValidationError):
            _spec(action={"kind": "sandbox_command", "command": ""})

    def test_rejects_command_too_long(self) -> None:
        with pytest.raises(ValidationError):
            _spec(action={"kind": "sandbox_command", "command": "x" * (MAX_COMMAND_CHARS + 1)})

    def test_timeout_bounds(self) -> None:
        _spec(action={"kind": "sandbox_command", "command": "x", "timeout_seconds": 1})
        _spec(action={"kind": "sandbox_command", "command": "x", "timeout_seconds": 3600})
        with pytest.raises(ValidationError):
            _spec(action={"kind": "sandbox_command", "command": "x", "timeout_seconds": 0})
        with pytest.raises(ValidationError):
            _spec(action={"kind": "sandbox_command", "command": "x", "timeout_seconds": 3601})

    def test_max_output_bounds(self) -> None:
        _spec(action={"kind": "sandbox_command", "command": "x", "max_output_bytes": 1024})
        with pytest.raises(ValidationError):
            _spec(action={"kind": "sandbox_command", "command": "x", "max_output_bytes": 1023})

    def test_defaults_materialized(self) -> None:
        spec = _spec()
        dumped = spec.action.model_dump()
        assert dumped["timeout_seconds"] == 300
        assert dumped["max_output_bytes"] == 65536


class TestWakeOwnerAction:
    def test_accepts(self) -> None:
        spec = _spec(action={"kind": "wake_owner", "content": "wake up"})
        assert spec.action.model_dump() == {"kind": "wake_owner", "content": "wake up"}

    def test_rejects_empty_content(self) -> None:
        with pytest.raises(ValidationError):
            _spec(action={"kind": "wake_owner", "content": ""})

    def test_rejects_content_too_long(self) -> None:
        with pytest.raises(ValidationError):
            _spec(action={"kind": "wake_owner", "content": "x" * (MAX_WAKE_CONTENT_CHARS + 1)})

    def test_wake_owner_rejects_command_key(self) -> None:
        # extra="forbid" — a wake_owner can't carry a sandbox_command key.
        with pytest.raises(ValidationError):
            _spec(action={"kind": "wake_owner", "content": "hi", "command": "echo"})


class TestWakeSessionAction:
    """The explicit-target cross-session wake action kind (#1280)."""

    def test_accepts_and_round_trips(self) -> None:
        spec = _spec(
            action={
                "kind": "wake_session",
                "target_session_id": "sess_01TARGET",
                "content": "go look at run X",
            }
        )
        assert isinstance(spec.action, WakeSessionAction)
        assert spec.action.model_dump() == {
            "kind": "wake_session",
            "target_session_id": "sess_01TARGET",
            "content": "go look at run X",
        }

    def test_discriminator_routes_through_action_adapter(self) -> None:
        action = TRIGGER_ACTION_ADAPTER.validate_python(
            {"kind": "wake_session", "target_session_id": "sess_01T", "content": "hi"}
        )
        assert isinstance(action, WakeSessionAction)

    def test_missing_target_session_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _spec(action={"kind": "wake_session", "content": "hi"})

    def test_missing_content_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _spec(action={"kind": "wake_session", "target_session_id": "sess_01T"})

    def test_empty_target_session_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _spec(action={"kind": "wake_session", "target_session_id": "", "content": "hi"})

    def test_empty_content_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _spec(action={"kind": "wake_session", "target_session_id": "sess_01T", "content": ""})

    def test_content_too_long_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _spec(
                action={
                    "kind": "wake_session",
                    "target_session_id": "sess_01T",
                    "content": "x" * (MAX_WAKE_CONTENT_CHARS + 1),
                }
            )

    def test_extra_key_rejected(self) -> None:
        # extra="forbid" — a wake_session can't carry a stray key.
        with pytest.raises(ValidationError):
            _spec(
                action={
                    "kind": "wake_session",
                    "target_session_id": "sess_01T",
                    "content": "hi",
                    "command": "echo",
                }
            )

    def test_accepted_on_update_no_replace_twin(self) -> None:
        # No Replace twin: both fields are required-at-create, so the same
        # member serves the update side.
        u = TriggerUpdate.model_validate(
            {"action": {"kind": "wake_session", "target_session_id": "sess_01T", "content": "hi"}}
        )
        assert isinstance(u.action, WakeSessionAction)


class TestTriggerCreateOrthogonality:
    def test_requires_source_and_action(self) -> None:
        with pytest.raises(ValidationError):
            TriggerCreate.model_validate(
                {"name": "x", "source": {"kind": "cron", "schedule": "* * * * *"}}
            )
        with pytest.raises(ValidationError):
            TriggerCreate.model_validate(
                {"name": "x", "action": {"kind": "wake_owner", "content": "hi"}}
            )

    def test_cron_wake_owner_is_legal(self) -> None:
        # A recurring model wake (deployment-style heartbeat).
        spec = _spec(
            source={"kind": "cron", "schedule": "0 9 * * *"},
            action={"kind": "wake_owner", "content": "morning check"},
        )
        assert isinstance(spec.source, CronSource)

    def test_one_shot_sandbox_command_is_legal(self) -> None:
        future = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
        spec = _spec(
            source={"kind": "one_shot", "fire_at": future},
            action={"kind": "sandbox_command", "command": "true"},
        )
        assert isinstance(spec.source, OneShotSource)

    def test_enabled_defaults_true(self) -> None:
        assert _spec().enabled is True

    def test_rejects_unknown_source_kind(self) -> None:
        with pytest.raises(ValidationError):
            _spec(source={"kind": "webhook", "url": "https://x"})


class TestTriggerUpdate:
    def test_empty_payload_valid(self) -> None:
        u = TriggerUpdate.model_validate({})
        assert u.source is None
        assert u.action is None
        assert u.enabled is None

    def test_validates_cron_when_provided(self) -> None:
        TriggerUpdate.model_validate({"source": {"kind": "cron", "schedule": "*/10 * * * *"}})
        with pytest.raises(ValidationError):
            TriggerUpdate.model_validate({"source": {"kind": "cron", "schedule": "not a cron"}})

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            TriggerUpdate.model_validate({"name": "rename-attempt"})

    def test_partial_sandbox_command_action_rejected(self) -> None:
        # Update-side action is a Replace variant: timeout_seconds +
        # max_output_bytes are REQUIRED, so a partial sandbox_command 422s
        # instead of silently resetting to create-time defaults (#818 §2.2).
        with pytest.raises(ValidationError):
            TriggerUpdate.model_validate(
                {"action": {"kind": "sandbox_command", "command": "echo hi"}}
            )

    def test_full_sandbox_command_action_accepted(self) -> None:
        u = TriggerUpdate.model_validate(
            {
                "action": {
                    "kind": "sandbox_command",
                    "command": "echo hi",
                    "timeout_seconds": 60,
                    "max_output_bytes": 2048,
                }
            }
        )
        assert u.action is not None

    def test_wake_owner_action_accepted_on_update(self) -> None:
        u = TriggerUpdate.model_validate({"action": {"kind": "wake_owner", "content": "hi"}})
        assert u.action is not None


class TestReadPathAcceptsRareCron:
    """The read path is structure-only — it must accept every row the write
    path ever accepted, including a legally-persisted rare cron that would
    fail today's create-time occurrence horizon (#818 §2.2)."""

    def test_adapter_accepts_rare_cron(self) -> None:
        src = TRIGGER_SOURCE_ADAPTER.validate_python({"kind": "cron", "schedule": "0 0 29 2 *"})
        assert isinstance(src, CronSource)

    def test_echo_accepts_rare_cron(self) -> None:
        now = datetime.now(UTC)
        echo = TriggerEcho.model_validate(
            {
                "id": "trig_x",
                "name": "leap",
                "source": {"kind": "cron", "schedule": "0 0 29 2 *"},
                "action": {
                    "kind": "sandbox_command",
                    "command": "true",
                    "timeout_seconds": 300,
                    "max_output_bytes": 65536,
                },
                "enabled": True,
                "next_fire": now,
                "last_fire_at": None,
                "last_fire_status": None,
                "consecutive_failures": 0,
                "metadata": {},
                "created_at": now,
                "updated_at": now,
            }
        )
        assert isinstance(echo.source, CronSource)


class TestComputeNextFire:
    def test_every_five_minutes_advances_one_slot(self) -> None:
        base = datetime(2026, 1, 1, 0, 1, 0, tzinfo=UTC)
        nxt = compute_next_fire("*/5 * * * *", base)
        assert nxt == datetime(2026, 1, 1, 0, 5, 0, tzinfo=UTC)

    def test_daily_9am(self) -> None:
        base = datetime(2026, 1, 1, 10, 0, 0, tzinfo=UTC)
        nxt = compute_next_fire("0 9 * * *", base)
        assert nxt == datetime(2026, 1, 2, 9, 0, 0, tzinfo=UTC)

    def test_strictly_after_from_time(self) -> None:
        base = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
        nxt = compute_next_fire("* * * * *", base)
        assert nxt > base


class TestComputeInitialNextFire:
    def test_cron_returns_next_slot(self) -> None:
        base = datetime(2026, 1, 1, 0, 1, 0, tzinfo=UTC)
        nxt = compute_initial_next_fire(CronSource(schedule="*/5 * * * *"), base)
        assert nxt == datetime(2026, 1, 1, 0, 5, 0, tzinfo=UTC)

    def test_one_shot_returns_fire_at_verbatim_even_if_past(self) -> None:
        past = datetime(2020, 1, 1, 0, 0, 0, tzinfo=UTC)
        nxt = compute_initial_next_fire(OneShotSource(fire_at=past), datetime.now(UTC))
        assert nxt == past


class TestValidateTriggers:
    def test_accepts_empty(self) -> None:
        validate_triggers([])

    def test_accepts_distinct_names(self) -> None:
        validate_triggers([_spec(name="a"), _spec(name="b")])

    def test_rejects_duplicate_names(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            validate_triggers([_spec(name="a"), _spec(name="a")])


class TestSessionCreateTriggers:
    def _session_payload(self, **overrides: object) -> dict[str, object]:
        body: dict[str, object] = {
            "agent_id": "agent_01HQR2K7VXBZ9MNPL3WYCT8FAA",
            "environment_id": "env_01HQR2K7VXBZ9MNPL3WYCT8FAA",
        }
        body.update(overrides)
        return body

    def _trigger(self, name: str) -> dict[str, object]:
        return {
            "name": name,
            "source": {"kind": "cron", "schedule": "* * * * *"},
            "action": {"kind": "sandbox_command", "command": "true"},
        }

    def test_default_empty(self) -> None:
        sc = SessionCreate.model_validate(self._session_payload())
        assert sc.triggers == []

    def test_accepts_initial_list(self) -> None:
        sc = SessionCreate.model_validate(
            self._session_payload(triggers=[self._trigger("a"), self._trigger("b")])
        )
        assert len(sc.triggers) == 2

    def test_rejects_duplicate_names_at_create(self) -> None:
        with pytest.raises(ValidationError):
            SessionCreate.model_validate(
                self._session_payload(triggers=[self._trigger("dup"), self._trigger("dup")])
            )

    def test_rejects_over_cap(self) -> None:
        too_many = [self._trigger(f"t{i}") for i in range(MAX_TRIGGERS_PER_SESSION + 1)]
        with pytest.raises(ValidationError):
            SessionCreate.model_validate(self._session_payload(triggers=too_many))

    def test_session_update_rejects_triggers_field(self) -> None:
        # Granular ops only (#270). SessionUpdate must not accept the field.
        with pytest.raises(ValidationError):
            SessionUpdate.model_validate({"triggers": [self._trigger("x")]})
