"""Pure built-in tool execution: parse → lookup → validate → call.

Both the model dispatch path (``harness/tool_dispatch.py``) and the
sandbox CLI broker (``sandbox/tool_broker.py``) drive built-in tools
through :func:`invoke_builtin`. The model path wraps it with the
event-append + sweep + ``_tool_lifecycle`` context manager; the broker
wraps it with HTTP response serialisation. The pure core never appends
events, never triggers the sweep, never touches sandbox state.

:class:`ToolBail` is the typed exception for *expected* tool-call
failures — bad JSON arguments, unknown tool, schema mismatch. Handler
exceptions propagate unchanged; the caller's transport-specific wrapper
decides what to do with them (log + sandbox eviction on the model path;
500 response on the broker).
"""

from __future__ import annotations

import contextvars
import json
from typing import Any

from aios.tools.registry import ToolNotFoundError, ToolResult, registry
from aios.tools.schema_errors import format_schema_violation

# The id of the tool_call the running handler is servicing, scoped per tool task.
# Set by :func:`invoke_builtin` around the handler call on the model dispatch path
# (the sandbox CLI broker and any caller that omits ``tool_call_id`` leave it ``None``).
# The parking ``call_*`` handlers read it via :func:`current_tool_call_id` to stamp their
# own ``tool_call_id`` onto the servicer edge, so a parked task can be re-derived and
# re-parked after a worker restart (#1431) — a builtin's only honest channel to its call id,
# which the ``(session_id, arguments)`` handler signature deliberately doesn't carry.
_CURRENT_TOOL_CALL_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_tool_call_id", default=None
)


def current_tool_call_id() -> str | None:
    """The id of the tool_call the current handler is servicing, or ``None`` outside one."""
    return _CURRENT_TOOL_CALL_ID.get()


class ToolBail(Exception):
    """Clean model-visible bail from a tool call.

    Raised for failures the model can read and self-correct from: bad
    JSON arguments, unknown tool name, schema validation mismatch, and
    (post-#1680) every *expected* failure a built-in handler used to
    encode as a bare ``{"error": ...}`` dict — a web_fetch non-2xx, an
    http_request policy denial, an edit/write memory precondition, a
    query timeout, and so on.  Distinct from a handler ``Exception``
    (which signals an internal failure inside the tool body and triggers
    sandbox eviction on the model path).

    ``detail`` carries the extra structured keys some handlers attach
    alongside the message — ``edit``/``write`` keep their ``{path, detail,
    matches}`` context this way.  The single event writer
    (``_tool_lifecycle`` → ``_append_tool_result``) merges these keys into
    the tool-result event's ``{"error": msg}`` content, so the
    prompt-surface content shape is preserved across the migration from
    the old dict returns (which json-encoded those same keys).
    """

    def __init__(self, message: str, *, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or {}


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
    unexpected extra keys, wrong types) built by the shared no-echo
    formatter (:func:`aios.tools.schema_errors.format_schema_violation`) — it
    never echoes the full ``arguments`` payload, only per-error
    expected/got lines plus the schema. The string lands in the
    tool_result's ``error`` body, so the model sees every issue at once
    and can self-correct without iterating one-at-a-time.

    The schema is the same dict registered with the tool and sent to the
    model as the tool's ``parameters``, so a mismatch genuinely
    indicates the model didn't follow the contract — not a framework
    bug.
    """
    return format_schema_violation(
        arguments,
        schema,
        root="",
        intro="Arguments failed schema validation.",
        retry_hint="Look at the tool's `parameters` schema for the correct shape and retry.",
        site="invoke.validate_arguments",
    )


async def invoke_builtin(
    session_id: str,
    tool_name: str,
    raw_arguments: Any,
    *,
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

    ``tool_call_id`` (model dispatch path) is exposed to the handler for the
    duration of the call via :func:`current_tool_call_id` — the parking ``call_*``
    handlers read it to stamp the servicer edge for crash-resume (#1431).
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
    token = _CURRENT_TOOL_CALL_ID.set(tool_call_id)
    try:
        return await tool.handler(session_id, arguments)
    finally:
        _CURRENT_TOOL_CALL_ID.reset(token)
