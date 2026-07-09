"""Shared JSON-Schema-violation formatter (#1769 spec v2).

Three call sites format a jsonschema validation failure into a model-facing
error string: :func:`aios.tools.invoke.validate_arguments` (every tool-call's
arguments), :func:`aios.tools.workflow_completion._validate_value` (a
``return``-tool answer against its request's ``output_schema``, plus
:func:`aios.workflows.step._validate_output_against_schema` for a workflow
run's terminal output), and :func:`aios.tools.invoke_session._validate_output`
(a ``call_session``/``call_workflow`` caller checking a peer/run's answer).
Before this module each site rolled its own message by reusing jsonschema's
stock ``err.message``, which embeds a full ``repr()`` of the failing instance
— and *also* echoed the whole argument/value payload a second time via
``json.dumps`` up front. On a multi-KB payload (the incident this closes: a
dev-implement session's ``return`` ``value`` was a ~3KB stringified-JSON blob)
that is two multi-KB echoes bracketing a ~20-char type mismatch, and the
replay eval (issue #1769 comment thread, "FINAL EVAL VERDICT") found the
echo-heavy wording rescues 0/120 samples — so the mandate is architectural,
not cosmetic: build every line from ``err.absolute_path`` / ``err.validator``
/ ``err.validator_value`` — never ``err.message`` (which embeds the instance
repr) — and never echo the full instance.

Rules (ratified spec v2, issue #1769 comment 4918753010):

1. No instance echo, ever. A scalar instance (str/int/float/bool/None) whose
   ``json.dumps()`` is short (<= 100 chars) may appear verbatim where it IS the
   diagnosis (``enum``/``const`` mismatches show the offending value). Anything
   longer, or a container (dict/list), is described by its JSON type name only.
2. Every ``type``-mismatch line reads ``expected <X>, got <json-type>`` — the
   JSON type of the failing instance via a 7-branch isinstance ladder (null,
   boolean, integer, number, string, array, object — boolean checked BEFORE
   integer, since Python ``bool`` is an ``int`` subclass).
3. The full schema is included when small (<= ~2KB serialized); otherwise only
   the subschema at the failing path (``err.schema``) — still no instance data.
4. A targeted hint fires when the failing top-level value is itself a JSON
   string that would parse+validate cleanly against the schema: the caller
   double-encoded its answer. This is detection + a short imperative hint
   only — NOT silent coercion (the coercion shim was evaluated and dropped;
   see #1769's final verdict comment — re-offering the tool schema fixed the
   incident, not the message).
"""

from __future__ import annotations

import json
from typing import Any

import jsonschema

from aios.logging import get_logger

log = get_logger(__name__)

_SCALAR_ECHO_MAX = 100
_SCHEMA_ECHO_MAX = 2000


def json_type_name(value: Any) -> str:
    """The JSON Schema type name of ``value`` (the 7-branch ladder).

    ``bool`` MUST be checked before ``int`` — Python's ``bool`` is an ``int``
    subclass, so ``isinstance(True, int)`` is ``True``.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__  # pragma: no cover - jsonschema instances are JSON-typed


def _describe_instance(instance: Any) -> str:
    """A short, safe description of a failing instance — never a full echo.

    A short scalar is shown verbatim (it usually IS the diagnosis, e.g. an
    ``enum``/``const`` mismatch); anything longer, or any container, collapses
    to its JSON type name.
    """
    if instance is None or isinstance(instance, (str, int, float, bool)):
        text = json.dumps(instance)
        if len(text) <= _SCALAR_ECHO_MAX:
            return text
    return f"<{json_type_name(instance)}>"


def _at(path: str, root: str) -> str:
    """The reported location of an error: ``root``-scoped when ``root`` is set
    (e.g. ``"value.answer"``, or bare ``"value"`` at the value's own root),
    otherwise the bare path or ``"<root>"`` when the path itself is empty.
    """
    if root:
        return f"{root}.{path}" if path else root
    return path or "<root>"


def _format_error_line(err: jsonschema.exceptions.ValidationError, *, root: str) -> str:
    path = ".".join(str(p) for p in err.absolute_path)
    at = _at(path, root)
    if err.validator == "type":
        expected = err.validator_value
        expected_str = " or ".join(expected) if isinstance(expected, list) else str(expected)
        return f"  - at {at}: expected {expected_str}, got {json_type_name(err.instance)}"
    if err.validator in ("enum", "const", "required", "additionalProperties"):
        # These jsonschema stock messages are already instance-free (they name
        # the offending keys/allowed values, not a repr of the whole instance).
        return f"  - at {at}: {err.message}"
    # Generic fallback (minLength/maxLength/pattern/minimum/maximum/etc.):
    # describe the failing instance safely rather than trust jsonschema's stock
    # message, which embeds a full repr of the instance.
    return f"  - at {at}: fails `{err.validator}: {err.validator_value}` (got {_describe_instance(err.instance)})"


def _schema_snippet(err: jsonschema.exceptions.ValidationError, schema: dict[str, Any]) -> str:
    """The schema to show: the full schema if small, else just the failing subschema."""
    full = json.dumps(schema)
    if len(full) <= _SCHEMA_ECHO_MAX:
        return full
    return json.dumps(err.schema)


def _stringified_json_hint(instance: Any, schema: dict[str, Any], *, site: str) -> str | None:
    """Detect the double-encoding quirk: ``instance`` is a JSON string that itself
    parses to something conforming to ``schema``. Returns a short imperative hint,
    or ``None`` if the quirk isn't present. Detection only — never coerces.
    """
    if not isinstance(instance, str):
        return None
    try:
        parsed = json.loads(instance)
    except (json.JSONDecodeError, ValueError):
        return None
    if not jsonschema.Draft202012Validator(schema).is_valid(parsed):
        return None
    log.info("return_value_stringified_json", site=site)
    return (
        "the string you sent is itself JSON that parses to a conforming value — "
        "pass that value directly, not wrapped in a string."
    )


def format_schema_violation(
    instance: Any,
    schema: dict[str, Any],
    *,
    root: str,
    intro: str,
    retry_hint: str | None,
    site: str,
) -> str | None:
    """Validate ``instance`` against ``schema``; ``None`` on success.

    On failure, returns a model-facing message: ``intro``, one no-echo line per
    error (``root``-scoped path + expected/got), the schema (full if small, else
    just the failing subschema), the stringified-JSON hint when detected, and
    ``retry_hint`` as the closing instruction (omitted when ``None`` — a
    fail-loud caller that does not bounce-and-retry). Never echoes the full
    ``instance``. ``site`` tags the stringified-JSON telemetry marker so its
    per-call-site frequency is distinguishable.
    """
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(instance), key=lambda e: list(e.absolute_path))
    if not errors:
        return None

    lines = [intro, "Errors:"]
    for err in errors:
        lines.append(_format_error_line(err, root=root))
    lines.append(f"The required schema is: {_schema_snippet(errors[0], schema)}")

    hint = _stringified_json_hint(instance, schema, site=site)
    if hint is not None:
        lines.append(hint)

    if retry_hint is not None:
        lines.append(retry_hint)
    return "\n".join(lines)
