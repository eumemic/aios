"""Unit tests for agent() structured output: value validation + prompt guidance.

The runtime pieces (storage, the per-request DB read, the end-to-end spawn→return
flow) are covered in tests/integration/test_wf_step.py. These cover the two pure
functions: the ``return`` value validator and the per-request schema guidance the
context builder renders into the child's request message.
"""

from __future__ import annotations

import math

import pytest

from aios.harness.context import render_user_event
from aios.tools.workflow_completion import _validate_value
from aios.workflows.determinism import canonical_schema_json
from aios.workflows.step import _unresolvable_ref

_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def test_validate_value_accepts_conforming() -> None:
    assert _validate_value({"answer": "hi"}, _SCHEMA) is None


def test_validate_value_rejects_with_path_and_retry_hint() -> None:
    err = _validate_value({"answer": 1}, _SCHEMA)
    assert err is not None
    assert "value.answer" in err  # the failing path, scoped under `value`
    assert "call return again" in err  # the model is told to retry
    # A bare-scalar schema is honored too (output_schema replaces `value` wholesale).
    assert _validate_value("hi", {"type": "number"}) is not None
    assert _validate_value(3, {"type": "number"}) is None


def test_render_surfaces_request_schema_per_request() -> None:
    event = {
        "role": "user",
        "content": "do the thing",
        "metadata": {"request": {"request_id": "r1", "output_schema": _SCHEMA}},
    }
    msg = render_user_event(event, None, None)
    assert "request_id: r1" in msg["content"]
    assert "must match this JSON Schema" in msg["content"]
    assert '"answer"' in msg["content"]  # the schema itself is rendered
    assert "do the thing" in msg["content"]  # original content preserved


def test_render_without_schema_is_unchanged() -> None:
    event = {
        "role": "user",
        "content": "hi",
        "metadata": {"request": {"request_id": "r1"}},
    }
    msg = render_user_event(event, None, None)
    assert "request_id: r1" in msg["content"]
    assert "JSON Schema" not in msg["content"]


def test_canonical_schema_json_admits_floats_deterministically() -> None:
    # A schema's decimal numeric constraints (which canonical_json bans for data) are
    # admitted here and serialized stably + key-order-independent.
    a = canonical_schema_json({"type": "number", "minimum": 1.5, "maximum": 9.9})
    b = canonical_schema_json({"maximum": 9.9, "minimum": 1.5, "type": "number"})
    assert a == b  # sort_keys → key order irrelevant
    assert "1.5" in a and "9.9" in a
    # NaN/Inf stay rejected (genuinely non-deterministic).
    with pytest.raises(ValueError):
        canonical_schema_json({"const": math.inf})


def test_unresolvable_ref_accepts_valid_rejects_broken() -> None:
    # No refs / valid self-contained local ref → resolves → None.
    assert _unresolvable_ref({"type": "object", "properties": {"a": {"type": "number"}}}) is None
    assert (
        _unresolvable_ref(
            {
                "$defs": {"N": {"type": "string"}},
                "properties": {"a": {"$ref": "#/$defs/N"}},
            }
        )
        is None
    )
    # Dangling local ref, remote ref, dangling $dynamicRef → the offending ref.
    assert (
        _unresolvable_ref({"properties": {"a": {"$ref": "#/$defs/missing"}}}) == "#/$defs/missing"
    )
    assert _unresolvable_ref({"$ref": "https://example.com/s.json"}) == "https://example.com/s.json"
    assert _unresolvable_ref({"properties": {"a": {"$dynamicRef": "#nope"}}}) == "#nope"
