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
