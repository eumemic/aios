"""Unit tests for the quiescence-guard nudge content (#1514).

The nudge fired at a quiescence attempt with open obligations now surfaces the
outstanding-task list AND, for each, its **acceptance contract** — the request's
``output_schema`` (definition-of-done). Source-agnostic: the obligations carry
their own schema whatever the caller, and an obligation with no schema renders a
plain "any returned result closes it" note (binary open/closed; no met-tracking).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from aios.models.sessions import Obligation
from aios.services.sessions import _nudge_content, _render_acceptance_contract

_NOW = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)


def _ob(rid: str, *, output_schema: dict[str, Any] | None = None) -> Obligation:
    return Obligation(
        request_id=rid,
        caller_kind="run",
        caller_id="run_owner",
        opened_at=_NOW,
        summary="do the thing",
        output_schema=output_schema,
    )


class TestRenderAcceptanceContract:
    def test_none_schema_renders_any_result_note(self) -> None:
        out = _render_acceptance_contract(None)
        assert "no output_schema" in out
        assert "any returned result" in out

    def test_empty_schema_renders_any_result_note(self) -> None:
        # An empty dict is falsy — treated as "no contract".
        assert "any returned result" in _render_acceptance_contract({})

    def test_schema_rendered_as_json(self) -> None:
        schema = {"type": "object", "required": ["answer"]}
        out = _render_acceptance_contract(schema)
        assert "output_schema" in out
        # The actual schema content is present (compact JSON).
        assert '"type":"object"' in out or '"type": "object"' in out
        assert "answer" in out

    def test_oversized_schema_truncated(self) -> None:
        big = {"type": "object", "properties": {f"k{i}": {"type": "string"} for i in range(500)}}
        out = _render_acceptance_contract(big)
        assert out.endswith("…")


class TestNudgeContent:
    def test_lists_each_request_id(self) -> None:
        content = _nudge_content([_ob("req-a"), _ob("req-b")])
        assert "req-a" in content
        assert "req-b" in content

    def test_surfaces_contract_per_obligation(self) -> None:
        schema = {"type": "object", "required": ["result"]}
        content = _nudge_content([_ob("req-a", output_schema=schema)])
        assert "req-a" in content
        assert "acceptance contract" in content
        assert "result" in content
        # The literal schema must be present (the definition-of-done).
        assert json.dumps(schema, separators=(",", ":"), sort_keys=True) in content

    def test_source_agnostic_mixed_schemas(self) -> None:
        # A caller-task with a schema and a self-goal without one (pre-#1512):
        # both render, each with its own contract clause.
        content = _nudge_content(
            [
                _ob("req-with", output_schema={"type": "string"}),
                _ob("req-without", output_schema=None),
            ]
        )
        assert "req-with" in content
        assert "req-without" in content
        assert "any returned result" in content  # the schema-less obligation
        assert '"type":"string"' in content  # the schema-bearing obligation

    def test_mentions_return_and_error(self) -> None:
        content = _nudge_content([_ob("req-a")])
        assert "return(" in content
        assert "error(" in content
