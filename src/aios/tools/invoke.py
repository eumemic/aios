"""Pure built-in tool invocation: parse → lookup → validate → call.

Both the model dispatch path (``harness/tool_dispatch.py``) and the
sandbox CLI broker (``sandbox/tool_broker.py``) drive built-in tools
through :func:`invoke_builtin`. The model path wraps it with the
event-append + sweep + ``_tool_lifecycle`` context manager; the broker
wraps it with HTTP response serialisation. The pure core never appends
events, never triggers the sweep, never touches sandbox state.

:class:`ToolBail` is the typed exception for *expected* invocation
failures — bad JSON arguments, unknown tool, schema mismatch. Handler
exceptions propagate unchanged; the caller's transport-specific wrapper
decides what to do with them (log + sandbox eviction on the model path;
500 response on the broker).
"""

from __future__ import annotations

import json
from typing import Any

import jsonschema

from aios.tools.registry import ToolNotFoundError, ToolResult, registry


class ToolBail(Exception):
    """Clean model-visible bail from a tool invocation.

    Raised for failures the model can read and self-correct from: bad
    JSON arguments, unknown tool name, schema validation mismatch.
    Distinct from a handler ``Exception`` (which signals an internal
    failure inside the tool body and triggers sandbox eviction on the
    model path).
    """


def parse_arguments(raw_args: Any) -> dict[str, Any] | None:
    """Parse tool arguments from a JSON string or dict.

    Returns ``None`` on malformed input. Callers raise :class:`ToolBail`
    so the model sees a clear error and retries.
    """
    if isinstance(raw_args, dict):
        return raw_args
    try:
        parsed = json.loads(raw_args) if raw_args else {}
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def validate_arguments(arguments: dict[str, Any], schema: dict[str, Any]) -> str | None:
    """Validate ``arguments`` against a tool's JSON Schema.

    Returns ``None`` on success, or a human-readable error string that
    enumerates every validation failure (missing required keys,
    unexpected extra keys, wrong types). The string lands in the
    tool_result's ``error`` body, so the model sees every issue at once
    and can self-correct without iterating one-at-a-time.

    The schema is the same dict registered with the tool and sent to the
    model as the tool's ``parameters``, so a mismatch genuinely
    indicates the model didn't follow the contract — not a framework
    bug. Surfacing specific paths (e.g. ``foo.bar[2]``) and
    passed-value previews keeps the feedback actionable.
    """
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(arguments), key=lambda e: list(e.absolute_path))
    if not errors:
        return None
    lines = [
        f"Arguments failed schema validation. You sent: {json.dumps(arguments)}",
        "Errors:",
    ]
    for err in errors:
        path = ".".join(str(p) for p in err.absolute_path) or "<root>"
        lines.append(f"  - at {path}: {err.message}")
    lines.append("Look at the tool's `parameters` schema for the correct shape and retry.")
    return "\n".join(lines)


async def invoke_builtin(
    session_id: str,
    tool_name: str,
    raw_arguments: Any,
    tool_call_id: str | None = None,
) -> ToolResult | dict[str, Any]:
    """Run a built-in tool: parse args, look up the handler, validate, call.

    Returns the handler's raw result (``ToolResult`` or plain ``dict``).

    Raises :class:`ToolBail` for expected failures (bad JSON, unknown
    tool name, schema mismatch). Handler exceptions propagate unchanged
    — the caller's transport wrapper decides what to do with them.

    Does NOT append events, trigger the sweep, or touch sandbox state.
    The model path wraps this with ``_tool_lifecycle`` + event append +
    sweep; the CLI broker wraps with HTTP response serialisation.

    ``tool_call_id`` (#1414) is the originating assistant ``tool_calls[*].id``,
    threaded through from the model dispatch path so a handler that needs a
    **deterministic, dispatch-stable** key (``set_goal`` derives its
    ``request_id`` from ``(session_id, tool_call_id)`` so a crash-retried
    dispatch re-keys the *same* goal edge — exactly-once) can read it.
    Handlers whose signature does not accept ``tool_call_id`` are called the
    legacy 2-arg way, so the threading is purely additive. The CLI broker
    passes ``None`` (agent_tool-only tools — ``set_goal`` included — never
    reach the broker; plumbing only).
    """
    arguments = parse_arguments(raw_arguments)
    if arguments is None:
        raise ToolBail("arguments were not valid JSON")
    try:
        tool = registry.get(tool_name)
    except ToolNotFoundError as err:
        raise ToolBail(err.message) from err
    schema_error = validate_arguments(arguments, tool.parameters_schema)
    if schema_error is not None:
        raise ToolBail(schema_error)
    if tool.wants_tool_call_id:
        return await tool.handler(session_id, arguments, tool_call_id)  # type: ignore[call-arg]
    return await tool.handler(session_id, arguments)
