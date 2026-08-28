"""The generic model-facing schema transform (:mod:`aios.tools.schema_diet`, #2294).

``test_workflow_tool_schema_budget`` pins the *outcome* on the real workflow
tools; these pin the *mechanism*, including the two properties that make the
transform safe to point at any fat tool schema: it only ever loosens the
dispatch-time check, and it never touches ``additionalProperties: false``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from aios.tools.schema_diet import slim_tool_schema


class _Nested(BaseModel):
    """A developer-facing docstring that the model should never see."""

    model_config = ConfigDict(extra="forbid")

    transport: str = "stdio"
    permission: str | None = None


class _Body(BaseModel):
    """Another developer-facing docstring."""

    model_config = ConfigDict(extra="forbid")

    script: str = Field(description="A very long authoring manual." * 20)
    servers: list[_Nested] = Field(default_factory=list)
    keep_me: str | None = Field(default=None, description="Real model guidance.")


def _slim(**kwargs: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "opaque_arrays": {"servers": "Opaque."},
        "redescribe": {"script": "Short."},
    }
    return slim_tool_schema(_Body.model_json_schema(), **{**defaults, **kwargs})


def test_opaque_field_collapses_to_a_bare_array() -> None:
    slim = _slim()
    assert slim["properties"]["servers"] == {"type": "array", "description": "Opaque."}


def test_collapsing_the_opaque_field_prunes_its_defs() -> None:
    assert "$defs" in _Body.model_json_schema()
    assert "$defs" not in _slim()


def test_nullable_opaque_field_keeps_its_null_arm() -> None:
    class _Nullable(BaseModel):
        servers: list[_Nested] | None = None

    slim = slim_tool_schema(
        _Nullable.model_json_schema(), opaque_arrays={"servers": "Opaque."}, redescribe={}
    )
    assert slim["properties"]["servers"]["anyOf"] == [{"type": "array"}, {"type": "null"}]


def test_redescribe_replaces_only_the_description() -> None:
    slim = _slim()
    assert slim["properties"]["script"] == {"type": "string", "description": "Short."}


def test_field_descriptions_are_preserved() -> None:
    assert _slim()["properties"]["keep_me"]["description"] == "Real model guidance."


def test_model_docstrings_and_titles_are_dropped() -> None:
    slim = slim_tool_schema(
        _Body.model_json_schema(), opaque_arrays={}, redescribe={"script": "Short."}
    )
    assert "description" not in slim
    assert "title" not in slim
    assert "description" not in slim["$defs"]["_Nested"]
    assert "title" not in slim["properties"]["script"]


def test_null_defaults_dropped_but_real_defaults_kept() -> None:
    slim = slim_tool_schema(
        _Body.model_json_schema(), opaque_arrays={}, redescribe={"script": "Short."}
    )
    assert "default" not in slim["properties"]["keep_me"]
    assert slim["$defs"]["_Nested"]["properties"]["transport"]["default"] == "stdio"


def test_additional_properties_false_is_never_stripped() -> None:
    """The invariant that keeps trusted ids out of a tool schema."""
    slim = _slim()
    assert slim["additionalProperties"] is False


def test_transform_does_not_mutate_its_input() -> None:
    original = _Body.model_json_schema()
    before = dict(original)
    slim_tool_schema(original, opaque_arrays={"servers": "Opaque."}, redescribe={})
    assert original == before
    assert "$defs" in original


def test_nested_properties_are_reached_through_a_ref() -> None:
    """A body reachable only via ``$ref`` gets the same treatment (call_workflow)."""

    class _Outer(BaseModel):
        body: _Body | None = None

    slim = slim_tool_schema(
        _Outer.model_json_schema(),
        opaque_arrays={"servers": "Opaque."},
        redescribe={"script": "Short."},
    )
    nested = slim["$defs"]["_Body"]["properties"]
    assert nested["servers"] == {"type": "array", "description": "Opaque."}
    assert nested["script"]["description"] == "Short."
    assert "_Nested" not in slim["$defs"]
