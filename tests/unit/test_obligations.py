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
    _TASK_MAX,
    _TASK_PREVIEW,
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

    def test_task_payload_over_4kb_is_rendered_verbatim(self) -> None:
        task = "first line\n" + "x" * 4096 + "\nlast line"
        ob = _ob("req_long", summary=task)
        block = build_obligations_tail_block([ob], session_id="sess_x", now=_NOW)
        assert block is not None
        content = block["content"]
        assert "verbatim task: " + task in content
        assert "TRUNCATED" not in content

    def test_oversized_task_fails_loud_instead_of_showing_a_silent_prefix(self) -> None:
        # ORIGINAL EVICTED (no present_request_ids): the task is genuinely
        # unrecoverable from context, so #2080's loud marker is correct.
        ob = _ob("req_too_long", summary="dangerous instruction " * 1000)
        block = build_obligations_tail_block([ob], session_id="sess_x", now=_NOW)
        assert block is not None
        content = block["content"]
        assert "TASK TRUNCATED" in content
        assert "return an error; do not act on or infer" in content
        assert "dangerous instruction" not in content

    def test_oversized_but_present_abridges_and_points_instead_of_refusing(self) -> None:
        # THE #2221 DEFECT. The original ask is still in the window, so the task is
        # NOT lost — it sits earlier in this same prompt. The reminder must abridge
        # + point, never order a refusal: this block is the LAST user-role content
        # the model reads, so a refuse-instruction here overrides the real task.
        task = "BUILD TASK — implement the widget. " * 700
        assert len(task) > _TASK_MAX
        ob = _ob("req_present", summary=task)
        block = build_obligations_tail_block(
            [ob], session_id="sess_x", now=_NOW, present_request_ids=frozenset({"req_present"})
        )
        assert block is not None
        content = block["content"]
        assert "TASK ABRIDGED IN THIS REMINDER" in content
        assert str(len(task)) in content
        # The refuse-instruction — in every phrasing — must be absent.
        assert "TASK TRUNCATED" not in content
        assert "return an error" not in content
        assert "do not act on or infer" not in content
        # A real prefix of the task is shown, so the reminder still carries signal.
        assert task[:200] in content

    def test_abridged_preview_is_bounded_not_merely_shorter(self) -> None:
        # The preview is a FIXED bound, not a fraction: a 10x larger task yields
        # the same preview width, so the reserved tail can't be inflated by size.
        small_over = "z" * (_TASK_MAX + 1)
        huge_over = "z" * (_TASK_MAX * 40)
        ids = frozenset({"req_x"})
        rendered = []
        for task in (small_over, huge_over):
            block = build_obligations_tail_block(
                [_ob("req_x", summary=task)],
                session_id="sess_x",
                now=_NOW,
                present_request_ids=ids,
            )
            assert block is not None
            rendered.append(len(block["content"]))
        # Both renders are bounded by the preview budget + a fixed pointer line,
        # and crucially the 40x-larger task does NOT produce a larger render
        # (only the char-count digits differ).
        assert abs(rendered[0] - rendered[1]) < 20
        assert max(rendered) < _TASK_PREVIEW + 500
        # And the abridged render never exceeds the AT-CAP VERBATIM render the
        # reserved budget already permits today.
        at_cap = build_obligations_tail_block(
            [_ob("req_x", summary="z" * _TASK_MAX)], session_id="sess_x", now=_NOW
        )
        assert at_cap is not None
        assert max(rendered) < len(at_cap["content"])

    def test_summary_exactly_at_cap_renders_verbatim(self) -> None:
        task = "y" * _TASK_MAX
        ob = _ob("req_boundary", summary=task)
        block = build_obligations_tail_block([ob], session_id="sess_x", now=_NOW)
        assert block is not None
        content = block["content"]
        assert "verbatim task: " + task in content
        assert "TRUNCATED" not in content
        assert "ABRIDGED" not in content

    def test_under_cap_task_is_verbatim_even_when_original_present(self) -> None:
        # Presence must not perturb the under-cap path: still byte-for-byte.
        task = "k" * 5000
        ob = _ob("req_small", summary=task)
        block = build_obligations_tail_block(
            [ob], session_id="sess_x", now=_NOW, present_request_ids=frozenset({"req_small"})
        )
        assert block is not None
        content = block["content"]
        assert "verbatim task: " + task in content
        assert "ABRIDGED" not in content
        assert "TRUNCATED" not in content

    def test_missing_summary_fails_loud_even_when_request_id_is_present(self) -> None:
        # Presence of the ORIGINAL ask cannot conjure content that was never
        # persisted on the frame — the unavailable marker must survive.
        ob = _ob("req_nosum2", summary=None)
        block = build_obligations_tail_block(
            [ob], session_id="sess_x", now=_NOW, present_request_ids=frozenset({"req_nosum2"})
        )
        assert block is not None
        content = block["content"]
        assert "TASK CONTENT UNAVAILABLE" in content
        assert "do not infer" in content

    def test_presence_is_per_request_not_global(self) -> None:
        # A present oversized task abridges; an absent one in the SAME block still
        # fails loud. The signal must not leak across obligations.
        big = "q" * (_TASK_MAX + 10)
        obs = [_ob("req_here", summary=big), _ob("req_gone", summary=big)]
        block = build_obligations_tail_block(
            obs, session_id="sess_x", now=_NOW, present_request_ids=frozenset({"req_here"})
        )
        assert block is not None
        lines = block["content"].splitlines()
        here = next(ln for ln in lines if "req_here" in ln)
        gone = next(ln for ln in lines if "req_gone" in ln)
        assert "ABRIDGED" in here and "TASK TRUNCATED" not in here
        assert "TASK TRUNCATED" in gone and "ABRIDGED" not in gone

    def test_missing_task_fails_loud(self) -> None:
        ob = _ob("req_nosum", summary=None)
        block = build_obligations_tail_block([ob], session_id="sess_x", now=_NOW)
        assert block is not None
        line = block["content"].splitlines()[1]
        assert "req_nosum" in line
        assert "TASK CONTENT UNAVAILABLE" in line
        assert "do not infer" in line

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

    def test_bound_covers_the_abridged_branch_not_just_the_marker(self) -> None:
        # THE FENCE (#2221). The bound is computed at WINDOWING time, before the
        # slate exists, so it cannot know which oversized tasks will render
        # abridged (fat) versus marker-replaced (thin). It must reserve the FATTER
        # branch. Costing the marker instead under-reserves — and an under-reserved
        # tail is exactly the read_windowed_events budget overflow (step crash)
        # that _TASK_MAX exists to prevent.
        #
        # The pre-existing bound test above uses 80-char summaries, so it never
        # crosses the cap and cannot catch this. This one does.
        for n in (1, MAX_RENDERED_OBLIGATIONS, MAX_RENDERED_OBLIGATIONS + 5):
            obs = [_ob(f"req_{i}", summary="w" * 23_631) for i in range(n)]
            bound = max_obligations_block_local(obs)
            # The worst case at send time: every original still in the window, so
            # every oversized task takes the fatter abridged branch.
            block = build_obligations_tail_block(
                obs,
                session_id="sess_x",
                now=_NOW,
                present_request_ids=frozenset(o.request_id for o in obs),
            )
            assert block is not None
            actual = approx_tokens([block])
            assert bound >= actual, f"under-reserved abridged tail n={n}: {bound} < {actual}"

    def test_bound_stays_bounded_for_oversized_tasks(self) -> None:
        # The abridged render must not scale with task size: a 40x larger task
        # cannot inflate the reserved tail (the Chesterton's-Fence property —
        # the cap holds, only the behaviour AT the cap changed).
        modest = max_obligations_block_local(
            [
                _ob(f"req_{i}", summary="w" * (_TASK_MAX + 1))
                for i in range(MAX_RENDERED_OBLIGATIONS)
            ]
        )
        enormous = max_obligations_block_local(
            [
                _ob(f"req_{i}", summary="w" * (_TASK_MAX * 40))
                for i in range(MAX_RENDERED_OBLIGATIONS)
            ]
        )
        assert enormous - modest < 50

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
