"""Unit tests for agent() structured output: value validation + prompt guidance.

The runtime pieces (storage, the per-request DB read, the end-to-end spawn→return
flow) are covered in tests/integration/test_wf_step.py. These cover the two pure
functions: the ``return`` value validator and the per-request schema guidance the
context builder renders into the child's request message.
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

import pytest

from aios.harness.context import render_user_event
from aios.tools.workflow_completion import _validate_value
from aios.workflows.determinism import canonical_schema_json
from aios.workflows.step import (
    _reject_invalid_output_schema,
    _SpawnResult,
    _unresolvable_ref,
)

_CREATED_AT = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)

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
    msg = render_user_event(event, None, None, _CREATED_AT)
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
    msg = render_user_event(event, None, None, _CREATED_AT)
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


@pytest.mark.asyncio
async def test_reject_invalid_output_schema_parity_between_arms() -> None:
    """The single-source output_schema validity gate must reject identically for the
    agent() and invoke_workflow() arms, modulo the call-name prefix + reject-kind.

    This is only expressible because the two arms now share ``_reject_invalid_output_schema``;
    on master each arm had its own copy and there was no single helper to call. If a future
    edit diverges one arm's message, the prefix-stripped tails stop matching and this fails.
    """
    invalid_schemas = [
        True,  # non-dict (bare boolean)
        {"type": "nonsense-keyword-value", "minimum": "x"},  # check_schema failure
        {"$ref": "#/$defs/missing"},  # unresolvable ref
    ]

    arms = [
        ("agent()", "bad_agent_call"),
        ("invoke_workflow()", "bad_invoke_workflow"),
    ]

    def _make_recording_reject(
        sink: list[tuple[str, str]],
    ) -> Callable[[str, str], Awaitable[_SpawnResult]]:
        async def _reject(kind: str, message: str) -> _SpawnResult:
            sink.append((kind, message))
            return _SpawnResult(rejected=True, needs_rewake=False)

        return _reject

    for schema in invalid_schemas:
        tails = []
        for call_name, reject_kind in arms:
            recorded: list[tuple[str, str]] = []
            result = await _reject_invalid_output_schema(
                schema,
                call_name=call_name,
                reject_kind=reject_kind,
                reject=_make_recording_reject(recorded),
            )
            # rejected → returns the (already-journaled) _SpawnResult, not None.
            assert result is not None
            assert result.rejected is True
            assert len(recorded) == 1
            kind, message = recorded[0]
            assert kind == reject_kind  # the passed reject-kind is journaled verbatim
            assert message.startswith(call_name)  # message carries the call-name prefix
            tails.append(message[len(call_name) :])

        # The ONLY legitimate divergence between arms is the prefix + kind: tails identical.
        assert tails[0] == tails[1]

    # A valid schema sails through both arms → None (no reject recorded).
    async def _reject_must_not_fire(kind: str, message: str) -> _SpawnResult:  # pragma: no cover
        raise AssertionError("valid schema must not reject")

    for call_name, reject_kind in arms:
        assert (
            await _reject_invalid_output_schema(
                {"type": "object"},
                call_name=call_name,
                reject_kind=reject_kind,
                reject=_reject_must_not_fire,
            )
            is None
        )
