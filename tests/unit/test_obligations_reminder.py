"""The durable obligations reminder render and its window reserves.

``render_obligations_reminder`` is the content of the durable reminder row the
composer writes when the open set changes or the listing scrolls out of the
window. It must be a pure function of the open set — byte-stable across
wall-clock time — or the change-gate would churn every minute and the
prompt-prefix cache with it. ``max_obligations_reminder_local`` bounds this
step's render for the window budget; ``OBLIGATIONS_EMPTY_UPPER_BOUND_LOCAL``
bounds the one-liner written when the set empties.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from aios.harness.context import _USER_MESSAGE_SEPARATOR_CONTENT
from aios.harness.obligations import (
    _HEADER,
    OBLIGATIONS_EMPTY_CONTENT,
    OBLIGATIONS_EMPTY_UPPER_BOUND_LOCAL,
    max_obligations_reminder_local,
    render_obligations_reminder,
)
from aios.harness.tokens import approx_tokens
from aios.models.sessions import Obligation

_OPENED = datetime(2026, 9, 5, 12, 0, 0, tzinfo=UTC)


def _ob(
    rid: str,
    *,
    caller_kind: str = "run",
    caller_id: str | None = None,
    opened_at: datetime = _OPENED,
    summary: str | None = "do the thing",
    output_schema: dict[str, Any] | None = None,
) -> Obligation:
    return Obligation(
        request_id=rid,
        caller_kind=caller_kind,
        caller_id=caller_id,
        opened_at=opened_at,
        summary=summary,
        output_schema=output_schema,
    )


class TestRenderObligationsReminder:
    def test_uses_absolute_opened_timestamp_not_a_relative_age(self) -> None:
        text = render_obligations_reminder([_ob("req_1")], session_id="sess_x")
        assert text.startswith(_HEADER)
        assert "(opened 2026-09-05T12:00:00+00:00)" in text
        assert "(open " not in text

    def test_byte_stable_across_wall_clock(self) -> None:
        obs = [_ob("req_1"), _ob("req_2", opened_at=_OPENED - timedelta(days=3))]
        a = render_obligations_reminder(obs, session_id="sess_x")
        # No ``now`` input exists to vary; the render is a function of the set.
        b = render_obligations_reminder(list(obs), session_id="sess_x")
        assert a == b

    def test_self_goal_origin_and_schema_contract_render(self) -> None:
        ob = _ob(
            "req_self",
            caller_kind="session",
            caller_id="sess_x",
            output_schema={"type": "object", "required": ["done"]},
        )
        text = render_obligations_reminder([ob], session_id="sess_x")
        assert "[self]" in text
        assert "expected output_schema" in text


class TestMaxObligationsReminderLocal:
    def test_zero_on_empty_set(self) -> None:
        assert max_obligations_reminder_local([]) == 0

    def test_bound_covers_the_real_render_with_separator(self) -> None:
        for n in (1, 3, 10, 14):
            obs = [
                _ob(f"req_{i}", summary="a task " * (i + 1), output_schema={"type": "object"})
                for i in range(n)
            ]
            bound = max_obligations_reminder_local(obs)
            real = approx_tokens(
                [
                    {"role": "assistant", "content": _USER_MESSAGE_SEPARATOR_CONTENT},
                    {
                        "role": "user",
                        "content": render_obligations_reminder(obs, session_id="sess_x"),
                    },
                ]
            )
            assert real <= bound, (n, real, bound)


class TestObligationsEmptyReserve:
    def test_one_liner_render_under_bound(self) -> None:
        priced = approx_tokens(
            [
                {"role": "assistant", "content": _USER_MESSAGE_SEPARATOR_CONTENT},
                {"role": "user", "content": OBLIGATIONS_EMPTY_CONTENT},
            ]
        )
        assert 0 < priced <= OBLIGATIONS_EMPTY_UPPER_BOUND_LOCAL
        assert OBLIGATIONS_EMPTY_CONTENT.startswith(_HEADER.split(" (")[0])
