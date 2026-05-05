"""JSON Schema generation from Python type hints.

Covers the ``_hint_to_schema`` helper end-to-end: primitive types,
parameterized generics, ``T | None`` unions (the headline case — a
connector parameter typed ``list[str] | None`` must surface the
array-of-strings shape, not an empty schema), plus an integration
test that runs the full ``@tool()`` decorator path so a regression
in the helper would also surface here.

Wire evidence motivating this:  ``mcp__telegram__telegram_send``'s
``attachments: list[str] | None = None`` parameter previously
published an empty ``{}`` schema, leaving the model to guess the
shape and learn from a tool error.
"""

from __future__ import annotations

import inspect
from typing import Annotated

from aios_connector import Connector, tool
from aios_connector.base import _TOOL_ATTR, ToolDescriptor, _hint_to_schema

# ── primitive shapes survive the swap ───────────────────────────────


def test_str_hint() -> None:
    assert _hint_to_schema(str) == {"type": "string"}


def test_int_hint() -> None:
    assert _hint_to_schema(int) == {"type": "integer"}


def test_float_hint() -> None:
    assert _hint_to_schema(float) == {"type": "number"}


def test_bool_hint() -> None:
    assert _hint_to_schema(bool) == {"type": "boolean"}


# ── parameterized generics now carry their item / value type ────────


def test_list_of_str_includes_items() -> None:
    """``list[str]`` must publish ``items`` so the model knows it's a
    list of strings, not an array of anything."""
    assert _hint_to_schema(list[str]) == {"type": "array", "items": {"type": "string"}}


def test_list_of_int_includes_items() -> None:
    assert _hint_to_schema(list[int]) == {"type": "array", "items": {"type": "integer"}}


def test_dict_str_int_includes_additional_properties() -> None:
    """``dict[str, int]`` must publish ``additionalProperties`` so the
    model knows values are integers, not arbitrary."""
    assert _hint_to_schema(dict[str, int]) == {
        "type": "object",
        "additionalProperties": {"type": "integer"},
    }


# ── T | None flattens to T (optional handled by required[] absence) ─


def test_optional_list_of_str_flattens_to_array() -> None:
    """The headline failing case.  ``list[str] | None`` previously
    produced ``{}``; pydantic emits an ``anyOf`` with a ``null`` arm
    that we strip — the parameter being optional is encoded by its
    absence from the parent object's ``required[]``, not by JSON
    ``null`` shape (per MCP / OpenAI / Anthropic tool-args convention).
    """
    assert _hint_to_schema(list[str] | None) == {
        "type": "array",
        "items": {"type": "string"},
    }


def test_optional_str_flattens_to_string() -> None:
    assert _hint_to_schema(str | None) == {"type": "string"}


def test_optional_int_flattens_to_integer() -> None:
    assert _hint_to_schema(int | None) == {"type": "integer"}


# ── empty / unannotated parameters keep the SDK's best-effort floor ─


def test_empty_hint_returns_empty_dict() -> None:
    """A parameter without an annotation — ``def f(self, x)`` — must
    not crash; the helper returns ``{}`` so schema generation stays
    best-effort."""
    assert _hint_to_schema(inspect.Parameter.empty) == {}


def test_none_hint_returns_empty_dict() -> None:
    assert _hint_to_schema(None) == {}


# ── Annotated[X, ...] passes through to the inner type ──────────────


def test_annotated_passes_through() -> None:
    """``Annotated[str, "description"]`` should still produce a string
    schema (pydantic treats Annotated metadata as descriptive, not
    type-changing)."""
    schema = _hint_to_schema(Annotated[str, "some metadata"])
    assert schema["type"] == "string"


# ── full @tool() integration: list[str] | None on a fixture method ──


class _SchemaFixtureConnector(Connector):
    name = "_schema_fixture"

    @tool()
    async def with_optional_list(
        self,
        text: str,
        attachments: list[str] | None = None,
    ) -> dict[str, object]:
        """Mirror of telegram_send's signature for end-to-end verification."""
        return {"text": text, "attachments": attachments}


def _descriptor(fn: object) -> ToolDescriptor:
    descriptor = getattr(fn, _TOOL_ATTR, None)
    assert isinstance(descriptor, ToolDescriptor)
    return descriptor


def test_decorator_publishes_full_array_schema_for_optional_list() -> None:
    """The @tool() decorator must publish the array-of-strings shape
    for ``attachments: list[str] | None``.  This is the exact wire
    failure the pydantic swap fixes — previously
    ``input_schema["properties"]["attachments"]`` was ``{}``.
    """
    descriptor = _descriptor(_SchemaFixtureConnector.with_optional_list)
    properties = descriptor.input_schema["properties"]
    assert properties["text"] == {"type": "string"}
    assert properties["attachments"] == {
        "type": "array",
        "items": {"type": "string"},
    }
    # `attachments` has a default → optional → not in required[].
    assert descriptor.input_schema["required"] == ["text"]
