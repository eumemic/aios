"""Unit tests for harness/obligations.py — the obligation formatting helpers.

#1514 removed the per-step tail-injected obligations block (the always-on #1413
surface). What remains in ``harness/obligations.py`` is the small set of
formatting helpers still used by the goal-management tool surface
(``list_goals`` renders an ``age``) and shared origin/summary formatting. These
tests pin those pure helpers; the per-step block builders are gone, and the
outstanding-task surface (with acceptance contracts) is asserted at the
quiescence nudge in ``tests/unit/test_quiescence_nudge.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from aios.harness.obligations import _format_age, _origin_label, _truncate_summary
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


def test_no_per_step_block_builders_remain() -> None:
    """The per-step injection (#1413) was removed (#1514): the block builders
    are no longer importable from the module."""
    import aios.harness.obligations as obl

    assert not hasattr(obl, "build_obligations_tail_block")
    assert not hasattr(obl, "max_obligations_block_local")


class TestFormatAge:
    def test_seconds(self) -> None:
        assert _format_age(_NOW - timedelta(seconds=3), _NOW) == "3s"

    def test_minutes(self) -> None:
        assert _format_age(_NOW - timedelta(minutes=5), _NOW) == "5m"

    def test_hours(self) -> None:
        assert _format_age(_NOW - timedelta(hours=2), _NOW) == "2h"

    def test_days(self) -> None:
        assert _format_age(_NOW - timedelta(days=4), _NOW) == "4d"

    def test_future_clamped_to_zero(self) -> None:
        assert _format_age(_NOW + timedelta(seconds=30), _NOW) == "0s"


class TestOriginLabel:
    def test_run_origin(self) -> None:
        assert _origin_label(_ob("r", caller_kind="run"), session_id="sess_x") == "run"

    def test_self_goal_origin(self) -> None:
        ob = _ob("r", caller_kind="session", caller_id="sess_x")
        assert _origin_label(ob, session_id="sess_x") == "self"

    def test_other_session_origin(self) -> None:
        ob = _ob("r", caller_kind="session", caller_id="sess_other")
        assert _origin_label(ob, session_id="sess_x") == "session"

    def test_missing_kind_renders_bare(self) -> None:
        assert _origin_label(_ob("r", caller_kind=""), session_id="sess_x") == "?"


class TestTruncateSummary:
    def test_none_is_empty(self) -> None:
        assert _truncate_summary(None) == ""

    def test_short_passes_through(self) -> None:
        assert _truncate_summary("hi") == "hi"

    def test_long_truncated_with_ellipsis(self) -> None:
        out = _truncate_summary("x" * 200)
        assert out.endswith("…")
        assert len(out) <= 61
