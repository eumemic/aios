"""Regression guard for schema validators retired by #2178."""

from __future__ import annotations

from aios.tools import invoke_session, workflow_completion
from aios.workflows import step


def test_legacy_strict_schema_validators_are_not_callable() -> None:
    """Future callers must use the normalization-aware shared schema gate."""
    assert not hasattr(workflow_completion, "_validate_value")
    assert not hasattr(invoke_session, "_validate_output")
    assert not hasattr(step, "_validate_output_against_schema")
