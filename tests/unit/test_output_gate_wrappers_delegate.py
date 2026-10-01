"""#2178: the per-site output-schema validators delegate to the shared gate.

PR #2096 unified normalization + strict validation into one shared gate
(:func:`aios.tools.schema_errors.normalize_and_format_schema_violation`). Three
per-site validators — ``workflow_completion._validate_value``,
``invoke_session._validate_output`` and ``step._validate_output_against_schema``
— survived that PR as strict-only, zero-caller "attractive nuisances": wiring one
back in would silently reintroduce path-dependent divergence (the same
stringified-JSON payload accepted at one boundary, rejected at another).

They now ARE the production gate at their site (each call site routes through
its wrapper) and each delegates to the shared gate, returning the NORMALIZED
value alongside the verdict so a caller cannot validate one value and persist
another. These tests pin that: every wrapper repairs a double-encoded conforming
answer, preserves an already-conforming one, and rejects a genuinely bad one.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from aios.tools import schema_errors
from aios.tools.invoke_session import _validate_output
from aios.tools.registry import ToolResult
from aios.tools.workflow_completion import _validate_value
from aios.workflows.step import _validate_output_against_schema

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"n": {"type": "integer"}},
    "required": ["n"],
}


def _call_output(value: Any, schema: dict[str, Any]) -> tuple[Any, bool]:
    normalized, violation = _validate_output(value, schema)
    assert violation is None or isinstance(violation, ToolResult)
    return normalized, violation is not None


def _return_value(value: Any, schema: dict[str, Any]) -> tuple[Any, bool]:
    normalized, message = _validate_value(value, schema)
    return normalized, message is not None


def _run_output(value: Any, schema: dict[str, Any]) -> tuple[Any, bool]:
    normalized, message = _validate_output_against_schema(value, schema)
    return normalized, message is not None


_GATES = [
    pytest.param(_return_value, id="workflow_completion._validate_value"),
    pytest.param(_call_output, id="invoke_session._validate_output"),
    pytest.param(_run_output, id="step._validate_output_against_schema"),
]


@pytest.mark.parametrize("gate", _GATES)
def test_double_encoded_conforming_value_is_repaired(
    gate: Callable[[Any, dict[str, Any]], tuple[Any, bool]],
) -> None:
    """The divergence #2096 closed: a strict-only validator rejects this payload."""
    assert gate('{"n": 1}', _SCHEMA) == ({"n": 1}, False)


@pytest.mark.parametrize("gate", _GATES)
def test_conforming_value_passes_unchanged(
    gate: Callable[[Any, dict[str, Any]], tuple[Any, bool]],
) -> None:
    assert gate({"n": 1}, _SCHEMA) == ({"n": 1}, False)
    assert gate('"hello"', {"type": "string"}) == ('"hello"', False)


@pytest.mark.parametrize("gate", _GATES)
def test_nonconforming_value_is_rejected_unchanged(
    gate: Callable[[Any, dict[str, Any]], tuple[Any, bool]],
) -> None:
    assert gate('{"n": "x"}', _SCHEMA) == ('{"n": "x"}', True)
    assert gate({"n": "x"}, _SCHEMA) == ({"n": "x"}, True)


@pytest.mark.parametrize("gate", _GATES)
def test_wrappers_route_through_the_shared_gate(
    gate: Callable[[Any, dict[str, Any]], tuple[Any, bool]], monkeypatch: Any
) -> None:
    """Correctness by construction: each wrapper calls the ONE shared gate, so a
    future change to the repair policy cannot leave a site behind."""
    calls: list[str] = []
    real = schema_errors.normalize_and_format_schema_violation

    def spy(*args: Any, **kwargs: Any) -> tuple[Any, str | None]:
        calls.append(kwargs["site"])
        return real(*args, **kwargs)

    for module in (
        "aios.tools.schema_errors",
        "aios.tools.invoke_session",
        "aios.tools.workflow_completion",
        "aios.workflows.step",
    ):
        monkeypatch.setattr(f"{module}.normalize_and_format_schema_violation", spy)
    gate({"n": 1}, _SCHEMA)
    assert len(calls) == 1


def test_call_output_without_schema_is_passthrough() -> None:
    assert _validate_output('{"n": 1}', None) == ('{"n": 1}', None)
