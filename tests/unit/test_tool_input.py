"""Unit tests for the schema-first arg-parsing util ``tools.input.tool_input``.

Lifted verbatim from ``workflow_management._parse``; these pin the
model-visible-bail contract independently of any one caller:

* a populated arg model on valid input,
* a ``ToolBail`` (NOT a raw ``ValidationError``) on a type/required
  violation,
* a ``ToolBail`` on a ``field_validator`` failure (passes JSON schema,
  fails Pydantic) — proves the second-pass semantic layer survives the
  lift,
* ``extra="forbid"`` arg models advertise a closed schema
  (``additionalProperties is False``).
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict, field_validator

from aios.tools.input import tool_input
from aios.tools.invoke import ToolBail


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    count: int


class _ValidatedArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str

    @field_validator("name")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("name must not be blank")
        return v


def test_returns_populated_model() -> None:
    parsed = tool_input(_Args, {"name": "w", "count": 3})
    assert isinstance(parsed, _Args)
    assert parsed.name == "w"
    assert parsed.count == 3


def test_type_violation_bails() -> None:
    with pytest.raises(ToolBail, match="invalid arguments"):
        tool_input(_Args, {"name": "w", "count": "not-an-int"})


def test_missing_required_bails() -> None:
    with pytest.raises(ToolBail, match="invalid arguments"):
        tool_input(_Args, {"name": "w"})


def test_field_validator_failure_bails() -> None:
    # Passes JSON schema (a string) but fails the Pydantic field_validator —
    # the semantic layer the JSON Schema can't encode must still surface as a
    # clean model-visible bail, not a raw ValidationError that evicts.
    with pytest.raises(ToolBail, match="invalid arguments"):
        tool_input(_ValidatedArgs, {"name": "   "})


def test_extra_forbid_schema_closed() -> None:
    assert _Args.model_json_schema()["additionalProperties"] is False
