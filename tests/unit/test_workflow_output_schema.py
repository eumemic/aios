"""Unit tests for agent() structured output: value validation + prompt guidance.

The runtime pieces (storage, the per-request DB read, the end-to-end spawn→return
flow) are covered in tests/integration/test_wf_step.py. These cover the two pure
functions: the ``return`` value validator and the per-request schema guidance the
context builder renders into the child's request message.
"""

from __future__ import annotations

from aios.harness.context import render_user_event
from aios.tools.workflow_completion import _validate_value

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
