"""Unit tests for harness/obligations.py — the obligations reminder row (#1413).

Pure-function coverage of the reminder content that survives context windowing
erasure of the original request user message: ``render_obligations_reminder``
(header + one line per open awaited obligation, ``self`` origin for a self-goal,
count cap + ``+K more``, the neutral abridged pointer past the task cap, the
loud unavailable marker) and ``max_obligations_reminder_local`` (a
never-under-reserving upper bound, bounded regardless of count or task size).
The byte-stability and absolute-timestamp properties the durable row depends on
are pinned in ``test_obligations_reminder.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from aios.harness.obligations import (
    _HEADER,
    _TASK_MAX,
    MAX_RENDERED_OBLIGATIONS,
    max_obligations_reminder_local,
    render_obligations_reminder,
)
from aios.harness.tokens import approx_tokens
from aios.models.sessions import Obligation

_NOW = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)


def _ob(
    rid: str,
    *,
    caller_kind: str = "run",
    caller_id: str | None = None,
    age: timedelta = timedelta(seconds=0),
    summary: str | None = "do the thing",
) -> Obligation:
    return Obligation(
        request_id=rid,
        caller_kind=caller_kind,
        caller_id=caller_id,
        opened_at=_NOW - age,
        summary=summary,
    )


def _render(obs: list[Obligation]) -> str:
    return render_obligations_reminder(obs, session_id="sess_x")


class TestRenderObligationsReminder:
    def test_header_then_one_line_per_obligation_with_literal_request_id(self) -> None:
        obs = [_ob("req_aaa"), _ob("req_bbb")]
        lines = _render(obs).splitlines()
        assert lines[0] == _HEADER
        # header + one line per obligation, no +K more under the cap
        assert len(lines) == 1 + len(obs)
        assert "req_aaa" in lines[1]
        assert "req_bbb" in lines[2]

    def test_oldest_first_ordering_preserved(self) -> None:
        # The caller fetches ORDER BY seq ASC; the renderer preserves input order.
        older = _ob("req_old", age=timedelta(hours=2))
        newer = _ob("req_new", age=timedelta(seconds=5))
        lines = _render([older, newer]).splitlines()
        assert "req_old" in lines[1]
        assert "req_new" in lines[2]

    def test_origin_labels_api_session_run(self) -> None:
        content = _render(
            [
                _ob("req_api", caller_kind="api"),
                _ob("req_sess", caller_kind="session", caller_id="sess_other"),
                _ob("req_run", caller_kind="run"),
            ]
        )
        assert "[api]" in content
        assert "[session]" in content
        assert "[run]" in content

    def test_self_label_when_caller_is_the_session_itself(self) -> None:
        ob = _ob("req_goal", caller_kind="session", caller_id="sess_x")
        assert "[self]" in _render([ob])

    def test_task_payload_over_4kb_is_rendered_verbatim(self) -> None:
        task = "first line\n" + "x" * 4096 + "\nlast line"
        content = _render([_ob("req_long", summary=task)])
        assert task in content
        assert "ABRIDGED" not in content

    def test_summary_exactly_at_cap_renders_verbatim(self) -> None:
        task = "y" * _TASK_MAX
        content = _render([_ob("req_boundary", summary=task)])
        assert task in content
        assert "ABRIDGED" not in content

    def test_oversized_task_renders_a_neutral_pointer_never_a_refusal(self) -> None:
        # The row is persisted and replayed for as long as the window keeps it,
        # so it can never prove the original ask is gone: a refuse-order here
        # would be obeyed over a task that may well sit earlier in the prompt
        # (#2221), and a content prefix is #2080's improvisation hazard.
        task = "BUILD TASK — implement the widget. " * 700
        assert len(task) > _TASK_MAX
        content = _render([_ob("req_big", summary=task)])
        assert "TASK ABRIDGED IN THIS REMINDER" in content
        assert str(len(task)) in content
        assert "original request message" in content
        assert "return an error" not in content
        assert "do not act on or infer" not in content
        assert "TASK TRUNCATED" not in content
        assert "BUILD TASK" not in content

    def test_oversized_render_is_bounded_not_proportional(self) -> None:
        small_over = "z" * (_TASK_MAX + 1)
        huge_over = "z" * (_TASK_MAX * 40)
        rendered = [len(_render([_ob("req_x", summary=t)])) for t in (small_over, huge_over)]
        # Only the char-count digits differ between the two renders.
        assert abs(rendered[0] - rendered[1]) < 20
        assert max(rendered) < 600

    def test_missing_task_fails_loud(self) -> None:
        line = _render([_ob("req_nosum", summary=None)]).splitlines()[1]
        assert "req_nosum" in line
        assert "TASK CONTENT UNAVAILABLE" in line
        assert "do not infer" in line

    def test_count_cap_renders_M_lines_plus_K_more(self) -> None:
        n = MAX_RENDERED_OBLIGATIONS + 5
        lines = _render([_ob(f"req_{i}") for i in range(n)]).splitlines()
        # header + M rendered + 1 "+K more" marker
        assert len(lines) == 1 + MAX_RENDERED_OBLIGATIONS + 1
        assert "+5 more" in lines[-1]

    def test_no_more_marker_at_exactly_the_cap(self) -> None:
        content = _render([_ob(f"req_{i}") for i in range(MAX_RENDERED_OBLIGATIONS)])
        assert "more)" not in content


class TestMaxObligationsReminderLocal:
    def test_zero_on_empty(self) -> None:
        assert max_obligations_reminder_local([]) == 0

    def test_upper_bound_never_under_reserves(self) -> None:
        for n in (1, 3, MAX_RENDERED_OBLIGATIONS, MAX_RENDERED_OBLIGATIONS + 50):
            obs = [_ob(f"req_{i}", summary="s" * 80) for i in range(n)]
            bound = max_obligations_reminder_local(obs)
            actual = approx_tokens([{"role": "user", "content": _render(obs)}])
            assert bound >= actual, f"under-reserved for n={n}: {bound} < {actual}"

    def test_bound_covers_oversized_tasks(self) -> None:
        for n in (1, MAX_RENDERED_OBLIGATIONS, MAX_RENDERED_OBLIGATIONS + 5):
            obs = [_ob(f"req_{i}", summary="w" * 23_631) for i in range(n)]
            bound = max_obligations_reminder_local(obs)
            actual = approx_tokens([{"role": "user", "content": _render(obs)}])
            assert bound >= actual, f"under-reserved for n={n}: {bound} < {actual}"

    def test_bound_stays_bounded_for_oversized_tasks(self) -> None:
        # The abridged pointer must not scale with task size: a 40x larger task
        # cannot inflate the reserved budget.
        modest = max_obligations_reminder_local(
            [
                _ob(f"req_{i}", summary="w" * (_TASK_MAX + 1))
                for i in range(MAX_RENDERED_OBLIGATIONS)
            ]
        )
        enormous = max_obligations_reminder_local(
            [
                _ob(f"req_{i}", summary="w" * (_TASK_MAX * 40))
                for i in range(MAX_RENDERED_OBLIGATIONS)
            ]
        )
        assert enormous - modest < 50

    def test_bound_stays_bounded_regardless_of_count(self) -> None:
        # The whole point of the cap: a huge count does not inflate the bound past
        # the capped (M + marker) render.
        small = max_obligations_reminder_local(
            [_ob(f"req_{i}", summary="s" * 80) for i in range(MAX_RENDERED_OBLIGATIONS)]
        )
        huge = max_obligations_reminder_local(
            [_ob(f"req_{i}", summary="s" * 80) for i in range(MAX_RENDERED_OBLIGATIONS + 1000)]
        )
        # huge has only the extra "+K more" marker over small — a small, fixed delta,
        # not an unbounded inflation.
        assert huge - small < 50
