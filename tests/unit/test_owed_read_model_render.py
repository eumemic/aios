"""Unit tests for the shared contract-bearing owed-read-model render (#1522).

The KEYSTONE DRY surface: ``render_owed_entry`` is the ONE place "outstanding
obligation + its acceptance contract" is formatted, and both contract-bearing
consumers (the quiescence-attempt surfacing via ``render_owed_listing`` and the
``list_obligations`` tool) feed from it. Covered here:

* ``render_owed_entry`` projects request_id / caller_kind / origin (incl. self) /
  summary / age / output_schema, with the schema bounded/elided.
* a large output_schema is elided in the render (the #1522 schema-side cap, the
  analogue of the 60-char summary cap).
* the per-step obligations tail budget bound (``max_obligations_block_local``)
  holds for the (schema-free) tail even when obligations carry large schemas —
  the contract-bearing render is the nudge/tool path, not the reserved tail.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from aios.harness.obligations import (
    _SCHEMA_MAX,
    _TASK_MAX,
    MAX_RENDERED_OBLIGATIONS,
    build_obligations_tail_block,
    max_obligations_block_local,
    render_owed_entry,
    render_owed_listing,
)
from aios.harness.tokens import approx_tokens
from aios.models.sessions import Obligation

_NOW = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
_SCHEMA = {
    "type": "object",
    "properties": {"shipped": {"type": "boolean"}},
    "required": ["shipped"],
}


def _ob(
    rid: str,
    *,
    caller_kind: str = "run",
    caller_id: str | None = None,
    age: timedelta = timedelta(seconds=0),
    summary: str | None = "do the thing",
    output_schema: dict[str, Any] | None = None,
) -> Obligation:
    return Obligation(
        request_id=rid,
        caller_kind=caller_kind,
        caller_id=caller_id,
        opened_at=_NOW - age,
        summary=summary,
        output_schema=output_schema,
    )


class TestRenderOwedEntry:
    def test_carries_all_fields(self) -> None:
        ob = _ob("req_1", caller_kind="api", age=timedelta(minutes=5), output_schema=_SCHEMA)
        entry = render_owed_entry(ob, session_id="sess_x", now=_NOW)
        assert entry["request_id"] == "req_1"
        assert entry["caller_kind"] == "api"
        assert entry["origin"] == "api"
        assert entry["summary"] == "do the thing"
        assert entry["age"] == "5m"
        # schema is a bounded single-line string preview, not the raw dict
        assert isinstance(entry["output_schema"], str)
        assert "shipped" in entry["output_schema"]

    def test_origin_self_for_self_goal(self) -> None:
        ob = _ob("req_goal", caller_kind="session", caller_id="sess_x")
        entry = render_owed_entry(ob, session_id="sess_x", now=_NOW)
        assert entry["origin"] == "self"

    def test_origin_peer_session_not_self(self) -> None:
        ob = _ob("req_peer", caller_kind="session", caller_id="sess_other")
        entry = render_owed_entry(ob, session_id="sess_x", now=_NOW)
        assert entry["origin"] == "session"

    def test_no_schema_renders_none(self) -> None:
        entry = render_owed_entry(_ob("req_n", output_schema=None), session_id="sess_x", now=_NOW)
        assert entry["output_schema"] is None

    def test_oversized_task_does_not_order_a_refusal(self) -> None:
        # THE #2221-round-2 DEFECT, on the read-model projection. This entry feeds
        # BOTH the quiescence nudge (persisted as a DURABLE user event, up to
        # REQUEST_NUDGE_BUDGET times) and the list_obligations tool. Neither surface
        # can establish that the original ask is gone — the request is open by
        # construction and its message may well still be in the window — so neither
        # may order a refusal. A refuse-order here is strictly worse than the tail
        # -block bug: the tail is ephemeral, this one persists in the event log.
        task = "BUILD TASK — implement the widget. " * 700
        assert len(task) > _TASK_MAX
        entry = render_owed_entry(_ob("req_big_task", summary=task), session_id="s", now=_NOW)
        summary = entry["summary"]
        assert isinstance(summary, str)
        # The refuse-instruction, in every phrasing that appears in the module.
        assert "TASK TRUNCATED" not in summary
        assert "return an error" not in summary
        assert "do not act on or infer" not in summary
        # Still a useful, honest pointer: the size, and where the real task lives.
        assert str(len(task)) in summary
        assert "original request message" in summary

    def test_oversized_task_render_is_bounded_and_leaks_no_prefix(self) -> None:
        # Two properties at once. (a) BOUNDED: this render is persisted on the nudge
        # path, so a 40x larger task must not produce a 40x larger durable event.
        # (b) NO PREFIX: unlike the tail block — which shows a preview only because
        # it has PROVEN the original is present — this surface cannot prove that, and
        # a plausible-looking prefix of a possibly-unrecoverable task is exactly
        # #2080's improvisation hazard. So it points, and shows no content at all.
        marker = "SECRET-TASK-CONTENT-MARKER"
        small_over = marker + "z" * (_TASK_MAX + 1)
        huge_over = marker + "z" * (_TASK_MAX * 40)
        lengths = []
        for task in (small_over, huge_over):
            entry = render_owed_entry(_ob("req_x", summary=task), session_id="s", now=_NOW)
            summary = entry["summary"]
            assert isinstance(summary, str)
            assert marker not in summary, "reminder surface must not echo task content"
            lengths.append(len(summary))
        # Only the char-count digits differ between the two renders.
        assert abs(lengths[0] - lengths[1]) < 20
        assert max(lengths) < 400

    def test_absent_summary_still_fails_loud(self) -> None:
        # The OTHER direction — the discrimination that proves the fix above did not
        # simply delete the guard. ``summary is None`` means nothing was ever
        # persisted on the frame, so the task is unrecoverable from ANY surface and
        # there is nothing to point at. #2080's fail-loud property is correct here
        # and must survive.
        entry = render_owed_entry(_ob("req_nosum", summary=None), session_id="s", now=_NOW)
        summary = entry["summary"]
        assert isinstance(summary, str)
        assert "TASK CONTENT UNAVAILABLE" in summary
        assert "return an error" in summary
        assert "do not infer" in summary

    def test_under_cap_task_is_verbatim(self) -> None:
        # The third direction: the ordinary case must stay byte-for-byte. A fix that
        # abridged everything would pass both tests above and still be wrong.
        task = "k" * (_TASK_MAX - 1)
        entry = render_owed_entry(_ob("req_small", summary=task), session_id="s", now=_NOW)
        assert entry["summary"] == task

    def test_large_schema_is_elided(self) -> None:
        big = {"type": "object", "properties": {f"k{i}": {"type": "string"} for i in range(500)}}
        entry = render_owed_entry(_ob("req_big", output_schema=big), session_id="sess_x", now=_NOW)
        rendered = entry["output_schema"]
        assert isinstance(rendered, str)
        # elided to the cap + an ellipsis — never the full, unbounded schema
        assert rendered.endswith("…")
        assert len(rendered) <= _SCHEMA_MAX + 1
        assert len(rendered) < len(str(big))


class TestRenderOwedListing:
    def test_header_and_one_line_per_obligation(self) -> None:
        obs = [_ob("req_a"), _ob("req_b")]
        text = render_owed_listing(obs, session_id="sess_x", header="HEAD", now=_NOW)
        assert text.startswith("HEAD")
        assert "req_a" in text
        assert "req_b" in text

    def test_schema_contract_surfaced(self) -> None:
        text = render_owed_listing(
            [_ob("req_c", output_schema=_SCHEMA)], session_id="sess_x", header="H", now=_NOW
        )
        assert "output_schema" in text
        assert "shipped" in text

    def test_oversized_task_does_not_order_a_refusal_in_the_nudge(self) -> None:
        # The DURABLE surface, end-to-end. This exact string is written to the event
        # log by services.sessions as a {"role": "user"} message (up to
        # REQUEST_NUDGE_BUDGET times) when a session tries to quiesce while owing an
        # obligation — i.e. the ordinary case of "child got the task, worked, ended a
        # turn without answering". #2080's evidence is that a model obeys a trailing
        # refuse-order, so this content ordering a refusal permanently re-parks the
        # oversized-payload sessions #2221 exists to unpark.
        task = "BUILD TASK — implement the widget. " * 700
        assert len(task) > _TASK_MAX
        text = render_owed_listing(
            [_ob("req_nudge", summary=task)], session_id="s", header="H", now=_NOW
        )
        assert "TASK TRUNCATED" not in text
        assert "return an error" not in text
        assert "do not act on or infer" not in text
        # The obligation is still identified and pointed at — a reminder, not silence.
        assert "req_nudge" in text
        assert "original request message" in text

    def test_nudge_absent_summary_still_fails_loud(self) -> None:
        # Other direction on the durable surface: a task that was never persisted is
        # unrecoverable from anywhere, so the nudge must still fail loud (#2080).
        text = render_owed_listing(
            [_ob("req_nosum", summary=None)], session_id="s", header="H", now=_NOW
        )
        assert "TASK CONTENT UNAVAILABLE" in text
        assert "return an error" in text

    def test_nudge_discriminates_per_obligation(self) -> None:
        # Both branches in ONE render: an oversized task must not drag the absent one
        # into silence, and the absent one must not drag the oversized one into a
        # refusal. A fix that keyed off the listing as a whole would fail here.
        text = render_owed_listing(
            [
                _ob("req_over", summary="w" * (_TASK_MAX + 10)),
                _ob("req_none", summary=None),
            ],
            session_id="s",
            header="H",
            now=_NOW,
        )
        over_line = next(ln for ln in text.splitlines() if "req_over" in ln)
        none_line = next(ln for ln in text.splitlines() if "req_none" in ln)
        assert "return an error" not in over_line
        assert "ABRIDGED" in over_line
        assert "TASK CONTENT UNAVAILABLE" in none_line
        assert "return an error" in none_line

    def test_nudge_render_stays_small_regardless_of_task_size(self) -> None:
        # This render is PERSISTED. An unbounded task must not produce an unbounded
        # durable event — the property that makes the neutral pointer safe to write
        # up to REQUEST_NUDGE_BUDGET times.
        obs = [_ob(f"req_{i}", summary="z" * (_TASK_MAX * 30)) for i in range(3)]
        text = render_owed_listing(obs, session_id="s", header="H", now=_NOW)
        assert len(text) < 1500

    def test_count_capped_with_more_marker(self) -> None:
        n = MAX_RENDERED_OBLIGATIONS + 4
        obs = [_ob(f"req_{i}") for i in range(n)]
        text = render_owed_listing(obs, session_id="sess_x", header="H", now=_NOW)
        assert "+4 more" in text

    def test_large_schema_does_not_blow_listing(self) -> None:
        # The whole point of the schema cap: a huge schema per entry is bounded,
        # so the rendered listing stays within a fixed per-entry envelope.
        big = {"type": "object", "properties": {f"k{i}": {"type": "string"} for i in range(2000)}}
        obs = [_ob(f"req_{i}", output_schema=big) for i in range(MAX_RENDERED_OBLIGATIONS)]
        text = render_owed_listing(obs, session_id="sess_x", header="H", now=_NOW)
        # Each entry's schema chunk is <= _SCHEMA_MAX (+ ellipsis); total stays
        # bounded by count*cap rather than count*huge.
        assert len(text) < MAX_RENDERED_OBLIGATIONS * (_SCHEMA_MAX + 200)


class TestTailBudgetBoundHoldsWithSchemas:
    def test_per_step_tail_bound_holds_even_with_large_schemas(self) -> None:
        # The reserved per-step obligations tail (build_obligations_tail_block) is
        # schema-FREE by design — the contract-bearing render is the nudge/tool
        # path. A large output_schema on the obligations therefore must NOT make
        # the actual tail exceed the reserved upper bound.
        big = {"type": "object", "properties": {f"k{i}": {"type": "string"} for i in range(1000)}}
        for n in (1, MAX_RENDERED_OBLIGATIONS, MAX_RENDERED_OBLIGATIONS + 25):
            obs = [_ob(f"req_{i}", summary="s" * 80, output_schema=big) for i in range(n)]
            bound = max_obligations_block_local(obs)
            block = build_obligations_tail_block(obs, session_id="sess_x", now=_NOW)
            assert block is not None
            actual = approx_tokens([block])
            assert bound >= actual, f"under-reserved for n={n}: {bound} < {actual}"
