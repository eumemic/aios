"""Derive a ``ToolSpec`` JSON Schema from a ``@tool``-decorated method.

The connector author writes Python; the SDK derives the schema the
model sees.  Operators no longer hand-write a ``tools.json`` that
drifts from the source.

The output shape is one entry of a ``ConnectionSetTools`` body — a
dict the runner POSTs at startup so the connection's tools list
matches whatever the connector container is actually serving::

    {
      "type": "custom",
      "name": "<tool_name>",
      "description": "<docstring summary + paragraphs>",
      "input_schema": {
        "type": "object",
        "properties": {<param_name>: <json_schema>, ...},
        "required": [<undefaulted_params>],
      }
    }

Type mapping (covers what production connectors actually use; the
table grows organically when a new connector method needs a new
shape):

* ``str`` / ``int`` / ``float`` / ``bool`` → primitive JSON types.
* ``Literal["a", "b"]`` → ``{"type": "string", "enum": [...]}``.
* ``T | None`` → ``{"type": ["X", "null"]}`` — the type-array form
  Anthropic and OpenAI both accept.  Multi-element unions
  (``A | B``) are not supported and raise ``SchemaError`` at
  derivation time so the connector author sees the failure
  immediately, not the model.
* ``list[T]`` → ``{"type": "array", "items": <T>}``.
* ``SandboxPath`` (and ``list[SandboxPath]``) → ``string``-shaped —
  the model sends path *strings*; the SDK resolves them to host
  ``Path``s before dispatching.
* Anything else → ``SchemaError``.

Focal-channel-injected params (``account``, ``chat_id``) are
excluded from the schema entirely — the model never supplies them.

Docstrings parse Google-style: the description is everything before
``Args:``; the per-param descriptions come from the indented entries
inside ``Args:``.  ``Returns:`` and other section headers terminate
``Args:`` parsing.
"""

from __future__ import annotations

import inspect
import re
import textwrap
import types
from collections.abc import Callable
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from .sandbox import _SandboxPathMarker

_FOCAL_INJECTABLE: frozenset[str] = frozenset({"account", "chat_id"})


class SchemaError(ValueError):
    """A ``@tool`` method's signature can't be turned into JSON Schema.

    Raised at derivation time (connector startup) — surfaces as a
    container crash so the connector author fixes the signature
    rather than the model seeing a malformed tool.
    """


def derive_tool_spec(name: str, fn: Callable[..., Any]) -> dict[str, Any]:
    """Build a ToolSpec dict for one ``@tool``-decorated method."""
    sig = inspect.signature(fn)
    try:
        hints = get_type_hints(fn, include_extras=True)
    except Exception:
        hints = {}
    description, param_docs = _parse_docstring(inspect.getdoc(fn))
    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []
    for param_name, param in sig.parameters.items():
        if param_name == "self" or param_name in _FOCAL_INJECTABLE:
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        hint = hints.get(param_name, param.annotation)
        prop = _annotation_to_schema(hint, param=param_name)
        if param.default is inspect.Parameter.empty:
            required.append(param_name)
        else:
            prop["default"] = param.default
        if param_name in param_docs:
            prop["description"] = param_docs[param_name]
        properties[param_name] = prop
    input_schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        input_schema["required"] = required
    return {
        "type": "custom",
        "name": name,
        "description": description,
        "input_schema": input_schema,
    }


# ─── type mapping ─────────────────────────────────────────────────────


def _annotation_to_schema(hint: Any, *, param: str) -> dict[str, Any]:
    """Recursive type → JSON schema mapping.  See module docstring."""
    inner, nullable = _split_optional(hint)
    if _is_sandbox_path(inner):
        return _wrap_nullable({"type": "string"}, nullable)
    origin = get_origin(inner)
    if origin is Literal:
        values = list(get_args(inner))
        if not values or not all(isinstance(v, str) for v in values):
            raise SchemaError(
                f"{param!r}: only string-valued Literal[...] is supported; got {inner!r}"
            )
        return _wrap_nullable({"type": "string", "enum": values}, nullable)
    if origin in (list, tuple):
        elems = get_args(inner)
        item_schema = _annotation_to_schema(elems[0], param=param) if elems else {}
        return _wrap_nullable({"type": "array", "items": item_schema}, nullable)
    if origin in (dict,):
        return _wrap_nullable({"type": "object"}, nullable)
    if inner is str:
        return _wrap_nullable({"type": "string"}, nullable)
    if inner is bool:
        return _wrap_nullable({"type": "boolean"}, nullable)
    if inner is int:
        return _wrap_nullable({"type": "integer"}, nullable)
    if inner is float:
        return _wrap_nullable({"type": "number"}, nullable)
    raise SchemaError(f"{param!r}: unsupported type annotation {hint!r}")


def _wrap_nullable(schema: dict[str, Any], nullable: bool) -> dict[str, Any]:
    """Promote ``{"type": "X"}`` to ``{"type": ["X", "null"]}`` when nullable.

    Preserves any sibling keys (``enum``, ``items``, etc.) the caller
    set on ``schema``.  No-op when ``nullable`` is False.
    """
    if not nullable:
        return schema
    base_type = schema.get("type")
    if isinstance(base_type, str):
        schema = {**schema, "type": [base_type, "null"]}
    return schema


def _split_optional(hint: Any) -> tuple[Any, bool]:
    """``T | None`` → ``(T, True)``; everything else → ``(hint, False)``.

    Raises ``SchemaError`` for unions with two or more non-``None``
    arms — those don't have a single-type JSON-schema representation
    in this minimal mapping.
    """
    origin = get_origin(hint)
    if origin not in (Union, types.UnionType):
        return hint, False
    non_none = [a for a in get_args(hint) if a is not type(None)]
    if len(non_none) == 1:
        return non_none[0], True
    raise SchemaError(f"unsupported union type {hint!r} — only ``T | None`` is allowed")


def _is_sandbox_path(hint: Any) -> bool:
    """Detect ``Annotated[Path, _SandboxPathMarker]`` regardless of nesting."""
    return any(
        isinstance(meta, _SandboxPathMarker)
        for meta in getattr(hint, "__metadata__", ())
    )


# ─── docstring parsing ────────────────────────────────────────────────


_SECTION_HEADER_RE = re.compile(r"^(Args|Arguments|Parameters|Returns|Raises|Yields):\s*$")
_PARAM_LINE_RE = re.compile(r"^(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$")


def _parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Split a Google-style docstring into ``(description, param_docs)``.

    The description is the lines before the first section header
    (``Args:``, ``Returns:``, etc.), with leading/trailing blanks
    stripped and inter-paragraph blank lines preserved.

    The ``Args:`` section is parsed param-by-param: a line shaped
    ``<name>: <text>`` starts a new param's description; subsequent
    lines indented further (the Google convention) are appended as
    continuation.  ``Returns:`` (or any other recognised section
    header) terminates the ``Args:`` block.
    """
    if not doc:
        return "", {}
    lines = textwrap.dedent(doc).splitlines()
    # Description capture: everything until the first section header.
    desc_lines: list[str] = []
    cursor = len(lines)  # no section headers → entire doc is description
    for i, line in enumerate(lines):
        if _SECTION_HEADER_RE.match(line.rstrip()):
            cursor = i
            break
        desc_lines.append(line)
    description = "\n".join(desc_lines).strip()

    # Args section: only meaningful when we hit one explicitly.
    param_docs: dict[str, str] = {}
    if cursor < len(lines) and lines[cursor].rstrip().startswith(("Args", "Arguments", "Parameters")):
        cursor += 1  # skip the header itself
        param_docs = _parse_args_section(lines[cursor:])
    return description, param_docs


def _parse_args_section(lines: list[str]) -> dict[str, str]:
    """Walk the body of an ``Args:`` block, building per-param descriptions.

    Returns when another section header (``Returns:``, ``Raises:``,
    etc.) is encountered or the block ends.
    """
    out: dict[str, str] = {}
    current_name: str | None = None
    current_lines: list[str] = []

    def _flush() -> None:
        if current_name is not None:
            out[current_name] = " ".join(part.strip() for part in current_lines if part.strip())

    for raw in lines:
        stripped = raw.rstrip()
        if not stripped.strip():
            current_lines.append("")
            continue
        if _SECTION_HEADER_RE.match(stripped.lstrip()):
            break
        match = _PARAM_LINE_RE.match(stripped.strip())
        # Indented continuation: append to the current param's lines.
        if (raw.startswith(" ") or raw.startswith("\t")) and current_name is not None and not (
            match and len(stripped) - len(stripped.lstrip()) <= 4
        ):
            # Heuristic: a deeply-indented "name: ..." line in some
            # docstrings is just continuation prose.  Only treat
            # top-level "name: ..." as a new param.
            current_lines.append(stripped.strip())
            continue
        if match:
            _flush()
            current_name = match.group(1)
            current_lines = [match.group(2)]
        else:
            current_lines.append(stripped.strip())
    _flush()
    return {k: v for k, v in out.items() if v}
