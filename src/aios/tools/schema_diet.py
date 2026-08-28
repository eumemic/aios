"""Model-facing tool-schema slimming — progressive disclosure at the schema edge.

A pydantic request body is the right shape for the HTTP/SDK plane: precise,
exhaustively documented, fully expanded. It is the *wrong* shape for a
model-facing tool schema, because a tool schema ships in **every request** of
every agent that holds the tool, whether or not the turn touches it. The
workflow-authoring trio (``create_workflow`` / ``update_workflow`` /
``call_workflow``) rendered at ~69KB combined — the fattest schemas on the tool
surface — from two inlining decisions (#2294):

1. the whole script-authoring manual inlined as the ``script`` field
   description, duplicated per tool; and
2. the declared-surface config models (``ToolSpec`` / ``McpServerSpec`` /
   ``HttpServerSpec``) expanded into ``$defs``, per tool.

This module is the fix, applied *only* where the registry renders a tool for
the model. It is deliberately a post-render transform on the JSON Schema rather
than a second set of parallel pydantic models:

* the HTTP/SDK contract is untouched — ``openapi.json`` and the generated SDK
  still render the full ``WorkflowCreate``/``WorkflowUpdate``/``InlineScriptBody``;
* server-side validation is untouched — the handlers still parse the *real*
  models, so a malformed body still gets a field-precise 422/``ToolBail``, which
  is what actually teaches the model; and
* there is no parallel model to drift out of sync with the real one.

The trade the schema makes is explicit: the model is told less up front, and
learns the rest from a precise error or from the on-demand authoring contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

_REF_PREFIX = "#/$defs/"


def slim_tool_schema(
    schema: Mapping[str, Any],
    *,
    opaque_arrays: Mapping[str, str],
    redescribe: Mapping[str, str],
) -> dict[str, Any]:
    """Return a slimmed copy of a rendered JSON Schema.

    ``opaque_arrays`` maps a property name to the one-line description that
    replaces its fully-expanded subschema: the property becomes a bare
    ``{"type": "array"}`` (null-arm preserved when the original allowed null),
    which drops the whole ``$defs`` tree it referenced. A property that is not
    actually an array is left alone — see :func:`_opaque_array`.

    ``redescribe`` maps a property name to a replacement ``description``, with
    the rest of the property's subschema left exactly as rendered.

    Both maps are applied to **every** ``properties`` object in the schema, not
    just the root — that is how a nested body reached only through a ``$ref``
    (``call_workflow``'s ``inline``) is slimmed by the same call. Property names
    here are unambiguous within these schemas.

    Four kinds of pydantic boilerplate are dropped unconditionally, because they
    are bytes the model never benefits from: every ``title`` (mechanically derived
    from the field name it sits next to); every ``"default": null`` (a restatement
    of "absent from ``required``" — non-null defaults are kept, they carry real
    information); every ``"additionalProperties": true`` (the JSON Schema default
    — the load-bearing ``false``, which is what keeps trusted ids out of these
    schemas, is never touched); and every MODEL-level ``description`` — the root's
    and each ``$defs`` entry's — which pydantic sources from the class docstring.
    Those are prose written for developers ("Request body for ``POST /v1/workflows``",
    issue numbers, even the names of the trusted kwargs the model must never see);
    the tool's own ``description`` is what introduces the tool. Field-level
    ``description`` is untouched: that is the guidance the model actually reads.

    ``$defs`` entries that are no longer reachable are dropped.
    """
    out = deepcopy(dict(schema))
    for properties in _iter_properties(out):
        for name, description in opaque_arrays.items():
            prop = properties.get(name)
            if prop is not None:
                properties[name] = _opaque_array(prop, description)
        for name, description in redescribe.items():
            prop = properties.get(name)
            if isinstance(prop, dict):
                prop["description"] = description
    _prune_unreachable_defs(out)
    _strip_boilerplate(out)
    out.pop("description", None)
    for definition in out.get("$defs", {}).values():
        if isinstance(definition, dict):
            definition.pop("description", None)
    return out


#: Keywords whose value is a NAME → schema mapping (the names are field names,
#: never schema keywords — which is exactly why a blind ``dict`` walk is wrong).
_SCHEMA_MAPS = ("properties", "$defs", "patternProperties", "definitions")
#: Keywords whose value is a list of schemas.
_SCHEMA_LISTS = ("anyOf", "oneOf", "allOf", "prefixItems")
#: Keywords whose value is a single schema.
_SCHEMA_VALUES = ("items", "not", "contains", "additionalProperties", "propertyNames")


def _walk_schemas(root: Any) -> list[dict[str, Any]]:
    """Every genuine schema object reachable from ``root``, root first.

    Structural, not a blind ``dict`` walk: it descends only through JSON Schema
    keywords. That distinction is load-bearing — inside ``properties`` the keys
    are FIELD names, so a field named ``title`` (or ``default``, or
    ``additionalProperties``) would otherwise be mistaken for the keyword of the
    same name and stripped, leaving a property that ``required`` still demands.
    """
    found: list[dict[str, Any]] = []
    stack: list[Any] = [root]
    while stack:
        current = stack.pop()
        if not isinstance(current, dict):
            continue
        found.append(current)
        for keyword in _SCHEMA_MAPS:
            group = current.get(keyword)
            if isinstance(group, dict):
                stack.extend(group.values())
        for keyword in _SCHEMA_LISTS:
            arms = current.get(keyword)
            if isinstance(arms, list):
                stack.extend(arms)
        for keyword in _SCHEMA_VALUES:
            sub = current.get(keyword)
            if isinstance(sub, dict):
                stack.append(sub)
    return found


def _iter_properties(root: Any) -> list[dict[str, Any]]:
    """Every ``properties`` mapping anywhere in the schema."""
    return [
        node["properties"]
        for node in _walk_schemas(root)
        if isinstance(node.get("properties"), dict)
    ]


def _strip_boilerplate(root: Any) -> None:
    """Remove titles, null defaults, and permissive ``additionalProperties``, in place."""
    for node in _walk_schemas(root):
        node.pop("title", None)
        if node.get("default", ...) is None:
            del node["default"]
        if node.get("additionalProperties") is True:
            del node["additionalProperties"]


def _opaque_array(prop: Any, description: str) -> dict[str, Any]:
    """Collapse a rendered list-of-model property to a bare array.

    Wholesale replacement, so everything the rendered property carried goes —
    including a ``default: []``, which says nothing the field's absence from
    ``required`` doesn't already say. Returns ``prop`` unchanged if it isn't
    actually an array: collapsing a non-array to ``{"type": "array"}`` would be
    the one way this module could TIGHTEN the dispatch-time check, so a
    name-only match is not enough to fire.
    """
    if isinstance(prop, dict) and not _is_array(prop):
        return prop
    array: dict[str, Any] = {"type": "array"}
    if _allows_null(prop):
        return {"anyOf": [array, {"type": "null"}], "description": description}
    return {**array, "description": description}


def _arms(prop: Any) -> list[Any]:
    """The property's type arms — its ``anyOf`` branches, or itself."""
    if not isinstance(prop, dict):
        return []
    arms = prop.get("anyOf")
    return arms if isinstance(arms, list) else [prop]


def _is_array(prop: Any) -> bool:
    """True iff every non-null arm of ``prop`` is an array."""
    non_null = [
        arm for arm in _arms(prop) if not (isinstance(arm, dict) and arm.get("type") == "null")
    ]
    return bool(non_null) and all(
        isinstance(arm, dict) and arm.get("type") == "array" for arm in non_null
    )


def _allows_null(prop: Any) -> bool:
    return any(isinstance(arm, dict) and arm.get("type") == "null" for arm in _arms(prop))


def _prune_unreachable_defs(schema: dict[str, Any]) -> None:
    """Drop ``$defs`` entries no longer referenced from the schema body."""
    defs = schema.get("$defs")
    if not isinstance(defs, dict):
        return
    body = {k: v for k, v in schema.items() if k != "$defs"}
    reachable: set[str] = set()
    frontier = _refs_in(body)
    while frontier:
        name = frontier.pop()
        if name in reachable or name not in defs:
            continue
        reachable.add(name)
        frontier |= _refs_in(defs[name])
    for name in set(defs) - reachable:
        del defs[name]
    if not defs:
        del schema["$defs"]


def _refs_in(node: Any) -> set[str]:
    """Every ``#/$defs/<name>`` target referenced anywhere under ``node``."""
    names: set[str] = set()
    for schema in _walk_schemas(node):
        ref = schema.get("$ref")
        if isinstance(ref, str) and ref.startswith(_REF_PREFIX):
            names.add(ref[len(_REF_PREFIX) :])
    return names
