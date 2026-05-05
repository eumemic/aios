"""JSON Schema generation from Python type hints.

Covers the ``_hint_to_schema`` helper end-to-end: primitive types,
parameterized generics, ``T | None`` unions (the headline case — a
connector parameter typed ``list[str] | None`` must surface the
array-of-strings shape, not an empty schema), plus an integration
test that runs the full ``@tool()`` decorator path so a regression
in the helper would also surface here.

Also covers :data:`aios_connector.SandboxPath`, the marker connector
authors annotate outbound-attachment parameters with: the schema must
publish ``string`` (not ``Path``) so the model passes path strings,
and the SDK's dispatch wrapper resolves them to host ``Path`` objects
before the tool body runs.

Wire evidence motivating this:  ``mcp__telegram__telegram_send``'s
``attachments: list[str] | None = None`` parameter previously
published an empty ``{}`` schema, leaving the model to guess the
shape and learn from a tool error.
"""

from __future__ import annotations

import inspect
from typing import Annotated

from aios_connector import Connector, SandboxPath, tool
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


# ── SandboxPath: model-facing schema is `string`, body sees `Path` ───


def test_sandbox_path_schema_is_string() -> None:
    """A bare :data:`SandboxPath` parameter must publish a string schema —
    the model passes in-sandbox path strings, the SDK auto-resolves to
    a host :class:`Path` before the tool body runs.
    """
    schema = _hint_to_schema(SandboxPath)
    assert schema["type"] == "string"
    assert "/workspace/" in schema["description"]


def test_list_sandbox_path_schema_is_array_of_strings() -> None:
    schema = _hint_to_schema(list[SandboxPath])
    assert schema == {
        "type": "array",
        "items": {
            "type": "string",
            "description": schema["items"]["description"],
        },
    }
    assert "/workspace/" in schema["items"]["description"]


def test_optional_sandbox_path_schema_is_string() -> None:
    schema = _hint_to_schema(SandboxPath | None)
    assert schema["type"] == "string"
    assert "/workspace/" in schema["description"]


def test_optional_list_sandbox_path_schema_is_array_of_strings() -> None:
    """The headline shape signal/telegram tools use:
    ``list[SandboxPath] | None = None``.
    """
    schema = _hint_to_schema(list[SandboxPath] | None)
    assert schema["type"] == "array"
    assert schema["items"]["type"] == "string"
    assert "/workspace/" in schema["items"]["description"]


class _SandboxFixtureConnector(Connector):
    name = "_sandbox_fixture"

    @tool()
    async def with_attachments(
        self,
        text: str,
        attachments: list[SandboxPath] | None = None,
    ) -> dict[str, object]:
        """End-to-end: list[SandboxPath] arrives resolved as list[Path]."""
        if attachments is None:
            return {"text": text}
        return {"text": text, "first_path": str(attachments[0])}

    @tool()
    async def with_one_attachment(
        self,
        path: SandboxPath,
    ) -> dict[str, object]:
        """End-to-end: scalar SandboxPath arrives resolved as Path."""
        return {"path": str(path)}


def test_sandbox_path_descriptor_records_list_kind() -> None:
    """The dispatch wrapper needs to know which params to resolve and how —
    ``sandbox_params`` is the structured record built at decoration time.
    """
    descriptor = _descriptor(_SandboxFixtureConnector.with_attachments)
    assert descriptor.sandbox_params == {"attachments": "list"}


def test_sandbox_path_descriptor_records_scalar_kind() -> None:
    descriptor = _descriptor(_SandboxFixtureConnector.with_one_attachment)
    assert descriptor.sandbox_params == {"path": "scalar"}


def test_sandbox_path_published_schema_is_string_array() -> None:
    """Integration: the @tool() decorator publishes the array-of-strings
    shape for ``attachments: list[SandboxPath] | None``."""
    descriptor = _descriptor(_SandboxFixtureConnector.with_attachments)
    properties = descriptor.input_schema["properties"]
    assert properties["attachments"]["type"] == "array"
    assert properties["attachments"]["items"]["type"] == "string"
    assert "/workspace/" in properties["attachments"]["items"]["description"]
    # `attachments` has a default → optional → not in required[].
    assert descriptor.input_schema["required"] == ["text"]


def test_scalar_sandbox_path_published_schema_is_string() -> None:
    descriptor = _descriptor(_SandboxFixtureConnector.with_one_attachment)
    properties = descriptor.input_schema["properties"]
    assert properties["path"]["type"] == "string"
    assert "/workspace/" in properties["path"]["description"]
