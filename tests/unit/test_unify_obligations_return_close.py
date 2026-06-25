"""unify-obligations #1517 (VERIFY): pin the invariants that gate retiring
``complete_goal``/``fail_goal`` (child #2) in favor of the general
``return``/``error`` close verbs.

This is a *verification* child: it adds no production code. The deliverable is a
written confirmation (posted on the issue) of three claims, with file:line
evidence. These tests lock the load-bearing one — (b), the servicer-side
``output_schema`` gate parity — so a future refactor that diverges the two close
paths fails here loudly rather than silently dropping #1513's validation.

Claims (full prose + evidence live on the issue/PR):

(a) An open self-goal makes ``owes_request`` true, so ``return``/``error`` are
    surfaced. The gate is caller-agnostic: ``owes_request = bool(obligations)``
    where ``obligations = get_open_obligations(...)`` (an awaited anti-join that
    does NOT filter by ``caller_kind``), so a ``caller.kind == "session"``
    self-edge counts exactly like any other owed request
    (``src/aios/harness/step_context.py:226-231``,
    ``src/aios/db/queries/sessions.py:525``).

(b) The general ``return`` path enforces the request's persisted
    ``output_schema`` SERVICER-SIDE, the same schema ``complete_goal`` enforces.
    ``return_handler`` → ``_enforce_output_schema`` → ``get_request_output_schema``
    → ``_validate_value`` (``src/aios/tools/workflow_completion.py:226-247``);
    ``complete_goal`` → ``get_request_output_schema`` → ``_validate_output``
    (``src/aios/tools/goal_management.py:271-272``). Both read the SAME schema off
    the trusted ``request_opened`` edge via the SAME query, and both run the SAME
    ``jsonschema.Draft202012Validator(schema).iter_errors(value)`` engine. The two
    validators differ only in error-string shape — both reject a non-conforming
    value (keeping the request open) and pass a conforming one. The tests below
    pin that behavioral equivalence.

(c) The nudge budget does not auto-``no_return`` a long-lived self-goal that is
    actively working: ``count_request_nudges`` counts only nudges whose ``seq`` is
    greater than the latest assistant tool-call turn, and
    ``append_assistant_and_guard_quiescence`` short-circuits on any turn carrying
    ``tool_calls`` — so any activity turn resets the per-request count
    (``src/aios/db/queries/sessions.py:632-679``,
    ``src/aios/services/sessions.py:1328,1719-1734``). The failure mode (3
    consecutive idle turns → ``no_return``) is identical for a
    ``complete_goal``-tracked goal, since the guard fires on the underlying
    obligation regardless of which tool would close it — not a regression.
"""

from __future__ import annotations

from typing import Any

import pytest

from aios.tools.invoke_session import _validate_output
from aios.tools.workflow_completion import _validate_value

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def _rejects_value(value: Any, schema: dict[str, Any]) -> bool:
    """True iff the *general return* servicer-side gate rejects ``value``."""
    return _validate_value(value, schema) is not None


def _rejects_result(value: Any, schema: dict[str, Any]) -> bool:
    """True iff the *complete_goal* servicer-side gate rejects ``value`` — it
    returns an ``output_schema_violation`` ToolResult on a mismatch, ``None`` on a
    match (same shape as the call_* output_schema path)."""
    return _validate_output(value, schema) is not None


@pytest.mark.parametrize(
    "value",
    [
        {"answer": "hi"},  # conforms
        {"answer": 1},  # wrong leaf type
        {},  # missing required
        {"answer": "hi", "extra": 1},  # additionalProperties
        "not-an-object",  # wrong root type
        None,
    ],
)
def test_b_return_and_complete_goal_gates_agree(value: Any) -> None:
    """(b) parity: for any value, the general ``return`` schema gate and the
    ``complete_goal`` schema gate reach the SAME accept/reject verdict against the
    SAME persisted ``output_schema``. This is the invariant that lets child #2
    retire ``complete_goal``/``fail_goal`` without losing #1513's validation."""
    assert _rejects_value(value, _SCHEMA) == _rejects_result(value, _SCHEMA)


def test_b_both_gates_accept_conforming_value() -> None:
    """A schema-valid value is accepted by both servicer-side gates (no error →
    the response is written, the request closes)."""
    assert _validate_value({"answer": "ok"}, _SCHEMA) is None
    assert _validate_output({"answer": "ok"}, _SCHEMA) is None


def test_b_both_gates_reject_nonconforming_value() -> None:
    """A schema-invalid value is rejected by both servicer-side gates (an error →
    NO response written, so the self-goal/obligation stays open and the model
    retries) — i.e. retiring ``complete_goal`` loses none of the validation."""
    assert _validate_value({"answer": 1}, _SCHEMA) is not None
    assert _validate_output({"answer": 1}, _SCHEMA) is not None


def test_b_bare_scalar_schema_honored_by_both_gates() -> None:
    """``output_schema`` replaces ``value`` wholesale, so a bare-scalar contract is
    enforced identically by both gates (the self-goal close is not object-only)."""
    assert _validate_value(3, {"type": "number"}) is None
    assert _validate_output(3, {"type": "number"}) is None
    assert _validate_value("x", {"type": "number"}) is not None
    assert _validate_output("x", {"type": "number"}) is not None


def test_b_complete_goal_no_schema_is_a_passthrough() -> None:
    """``_validate_output(None-schema)`` is a no-op pass — the structural reason the
    general ``return`` gate (which resolves ``schema is None`` to "proceed" in
    ``_enforce_output_schema``) and ``complete_goal`` agree on the no-schema case.
    ``_validate_value`` is only ever called by ``return`` *after* a non-None schema
    is fetched, so its no-schema behavior is encoded in that caller, asserted
    here for completeness of the parity argument."""
    assert _validate_output({"anything": True}, None) is None
