"""Unit tests for harness/obligations.py — the tail-injected obligations block (#1413).

Pure-function coverage of the always-on reminder surface that survives context
windowing erasure of the original request user message: ``build_obligations_tail_block``
(render one line per open awaited obligation, ``None`` on empty, ``self`` origin
for a self-goal, count cap + ``+K more``) and ``max_obligations_block_local`` (a
never-under-reserving upper bound, bounded regardless of count).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from aios.harness.obligations import (
    MAX_RENDERED_OBLIGATIONS,
    build_obligations_tail_block,
    max_obligations_block_local,
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


class TestBuildObligationsTailBlock:
    def test_empty_set_returns_none(self) -> None:
        assert build_obligations_tail_block([], session_id="sess_x") is None

    def test_shape_is_user_role(self) -> None:
        block = build_obligations_tail_block([_ob("req_1")], session_id="sess_x", now=_NOW)
        assert block is not None
        assert block["role"] == "user"
        assert isinstance(block["content"], str)

    def test_one_line_per_obligation_with_literal_request_id(self) -> None:
        obs = [_ob("req_aaa"), _ob("req_bbb")]
        block = build_obligations_tail_block(obs, session_id="sess_x", now=_NOW)
        assert block is not None
        lines = block["content"].splitlines()
        # header + one line per obligation, no +K more under the cap
        assert len(lines) == 1 + len(obs)
        assert "req_aaa" in lines[1]
        assert "req_bbb" in lines[2]

    def test_oldest_first_ordering_preserved(self) -> None:
        # The caller fetches ORDER BY seq ASC; the renderer preserves input order.
        older = _ob("req_old", age=timedelta(hours=2))
        newer = _ob("req_new", age=timedelta(seconds=5))
        block = build_obligations_tail_block([older, newer], session_id="sess_x", now=_NOW)
        assert block is not None
        lines = block["content"].splitlines()
        assert "req_old" in lines[1]
        assert "req_new" in lines[2]

    def test_origin_labels_api_session_run(self) -> None:
        obs = [
            _ob("req_api", caller_kind="api"),
            _ob("req_sess", caller_kind="session", caller_id="sess_other"),
            _ob("req_run", caller_kind="run"),
        ]
        block = build_obligations_tail_block(obs, session_id="sess_x", now=_NOW)
        assert block is not None
        content = block["content"]
        assert "[api]" in content
        assert "[session]" in content
        assert "[run]" in content

    def test_self_label_when_caller_is_the_session_itself(self) -> None:
        ob = _ob("req_goal", caller_kind="session", caller_id="sess_x")
        block = build_obligations_tail_block([ob], session_id="sess_x", now=_NOW)
        assert block is not None
        assert "[self]" in block["content"]

    def test_summary_quoted_and_truncated_to_60(self) -> None:
        ob = _ob("req_long", summary="x" * 200)
        block = build_obligations_tail_block([ob], session_id="sess_x", now=_NOW)
        assert block is not None
        line = block["content"].splitlines()[1]
        # The quoted preview body is <= 60 chars (+ ellipsis), never the full 200.
        assert '"' in line
        body = line.split('"')[1]
        assert len(body.rstrip("…")) <= 60

    def test_missing_summary_renders_id_only_no_crash(self) -> None:
        ob = _ob("req_nosum", summary=None)
        block = build_obligations_tail_block([ob], session_id="sess_x", now=_NOW)
        assert block is not None
        line = block["content"].splitlines()[1]
        assert "req_nosum" in line
        # No empty quoted clause for an absent summary.
        assert '""' not in line

    def test_age_clause_present(self) -> None:
        ob = _ob("req_age", age=timedelta(minutes=5))
        block = build_obligations_tail_block([ob], session_id="sess_x", now=_NOW)
        assert block is not None
        assert "(open 5m)" in block["content"]

    def test_age_never_negative(self) -> None:
        # opened in the "future" relative to now → clamps to 0s, never crashes.
        ob = _ob("req_future", age=timedelta(seconds=-30))
        block = build_obligations_tail_block([ob], session_id="sess_x", now=_NOW)
        assert block is not None
        assert "(open 0s)" in block["content"]

    def test_count_cap_renders_M_lines_plus_K_more(self) -> None:
        n = MAX_RENDERED_OBLIGATIONS + 5
        obs = [_ob(f"req_{i}") for i in range(n)]
        block = build_obligations_tail_block(obs, session_id="sess_x", now=_NOW)
        assert block is not None
        lines = block["content"].splitlines()
        # header + M rendered + 1 "+K more" marker
        assert len(lines) == 1 + MAX_RENDERED_OBLIGATIONS + 1
        assert "+5 more" in lines[-1]

    def test_no_more_marker_at_exactly_the_cap(self) -> None:
        obs = [_ob(f"req_{i}") for i in range(MAX_RENDERED_OBLIGATIONS)]
        block = build_obligations_tail_block(obs, session_id="sess_x", now=_NOW)
        assert block is not None
        assert "more)" not in block["content"]


class TestMaxObligationsBlockLocal:
    def test_zero_on_empty(self) -> None:
        assert max_obligations_block_local([]) == 0

    def test_upper_bound_never_under_reserves(self) -> None:
        for n in (1, 3, MAX_RENDERED_OBLIGATIONS, MAX_RENDERED_OBLIGATIONS + 50):
            obs = [_ob(f"req_{i}", summary="s" * 80) for i in range(n)]
            bound = max_obligations_block_local(obs)
            block = build_obligations_tail_block(obs, session_id="sess_x", now=_NOW)
            assert block is not None
            actual = approx_tokens([block])
            assert bound >= actual, f"under-reserved for n={n}: {bound} < {actual}"

    def test_bound_stays_bounded_regardless_of_count(self) -> None:
        # The whole point of the cap: a huge count does not inflate the bound past
        # the capped (M + marker) render.
        small = max_obligations_block_local(
            [_ob(f"req_{i}", summary="s" * 80) for i in range(MAX_RENDERED_OBLIGATIONS)]
        )
        huge = max_obligations_block_local(
            [_ob(f"req_{i}", summary="s" * 80) for i in range(MAX_RENDERED_OBLIGATIONS + 1000)]
        )
        # huge has only the extra "+K more" marker over small — a small, fixed delta,
        # not an unbounded inflation.
        assert huge - small < 50
