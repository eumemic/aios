"""The **response edge** of `invoke_session` — a workflow child answering the
request it was invoked with (a response is NEVER inferred from idle).

A child is spawned with a **request** (its first user message, stamped
``metadata.request``; see ``create_child_session``). Exactly one response is ever
captured for it, by whichever of these writes first:

* the ``return`` / ``error`` tools (the model's own answer — injected ONLY into a
  workflow child, ``origin='background'`` with a ``parent_run_id``; see
  ``compute_step_prelude``) and the harness erroring path
  (``fail_all_open_requests``, when the model fails past its retry budget) both go
  through :func:`respond_to_request`,
* the run's totality backstop (``services.sessions`` quiescence guard) writes
  inside its own open transaction — it can't acquire its own connection.

The common, load-bearing seam for an external writer is
``sessions_service.write_child_response`` — exactly-once first-writer-wins (via
``write_response_if_absent``) PLUS the atomic ``child_done`` marker the
needs-step sweep's recall depends on (a writer that bypassed it would silently
re-open the lost-wake stall the marker closes). A ``return``+``error`` batch, a
model double-call, a model-failure racing a late ``return``, or the backstop
racing any of them all collapse to one response there.
:func:`respond_to_request` adds the caller-wake on top for the pool-level callers.

It does **not** archive or terminate the child. Responding resumes the caller
regardless of the child's subsequent fate; the child carries on (a fresh
``agent()`` child has nothing else to do, so it quiesces, and run-end reclaim
archives it — off the correctness path). The response is the durable record the
caller's harvest reads; the periodic ``wf_runs`` sweep is the lost-wake backstop.
"""

from __future__ import annotations

import json
from typing import Any

import jsonschema

from aios.db import queries
from aios.harness import runtime
from aios.jobs.app import defer_run_wake
from aios.models.sessions import Err, Ok, Outcome
from aios.services import sessions as sessions_service
from aios.tools.registry import ToolResult, openai_tool_entry, registry

RETURN_TOOL_NAME = "return"
ERROR_TOOL_NAME = "error"

RETURN_DESCRIPTION = (
    "Answer a request you were given with a successful result. `request_id` is the "
    "id shown with the request you're answering; `value` is the result the caller "
    "receives. Answer each open request exactly once."
)
ERROR_DESCRIPTION = (
    "Answer a request you were given with a failure. `request_id` is the id shown "
    "with the request you're answering; `message` explains why you couldn't "
    "complete it. Answer each open request exactly once."
)

_REQUEST_ID_PROP = {"type": "string", "description": "the id of the request you're answering"}
_RETURN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "request_id": _REQUEST_ID_PROP,
        "value": {"description": "the result returned to the caller (any JSON)"},
    },
    "required": ["request_id", "value"],
    "additionalProperties": False,
}
_ERROR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "request_id": _REQUEST_ID_PROP,
        "message": {"type": "string", "description": "why the request failed"},
    },
    "required": ["request_id", "message"],
    "additionalProperties": False,
}

_NOT_A_CHILD = ToolResult(
    content="return/error is only available to a workflow agent child", is_error=True
)
_UNKNOWN_REQUEST = ToolResult(
    content="no open request with that request_id — answer a request using the request_id "
    "shown in its message",
    is_error=True,
)


async def respond_to_request(
    pool: Any,
    session_id: str,
    *,
    request_id: str,
    outcome: Outcome,
) -> str:
    """Write one request's response and wake its caller — the shared core behind
    every `invoke_session` response (``return``/``error``, the harness erroring
    path, the no_return backstop).

    Responding does **not** archive or terminate the target; it resumes the caller
    regardless of the target's fate. The response is captured **exactly once per
    request** (``write_response_if_absent`` — first-writer-wins); the caller is woken
    only when this call actually wrote (a duplicate is a no-op — the first response
    already woke it; the periodic ``wf_runs`` sweep is the lost-wake backstop).

    The caller is read off the trusted ``request_opened`` edge (#1127), so this is no
    longer run-only: a request whose ``caller.kind`` is ``run`` keeps the fused
    ``child_done`` marker + run wake; a ``session`` caller writes the plain response
    edge and the caller session is woken; an ``api`` caller writes the edge and nobody
    is woken (the ephemeral HTTP awaiter long-polls). Ownership — not child-ness — is
    the gate.

    Returns one of ``responded`` | ``duplicate`` | ``not_a_child`` |
    ``unknown_request`` (the ``request_id`` isn't an open request of this session)
    so callers can shape their own result. ``not_a_child`` is retained for the
    legacy run path: a request whose edge names a run caller but whose session lost
    its ``parent_run_id`` fails closed rather than signal a NULL run.
    """
    # All routing lives in the one conn-level writer (sessions.respond_to_request_conn);
    # here we just provide the transaction and fire the returned wakes post-commit.
    async with pool.acquire() as conn, conn.transaction():
        write = await sessions_service.respond_to_request_conn(
            conn,
            session_id,
            request_id=request_id,
            outcome=outcome,
        )
    if write.wake_run_id is not None:
        # batch: child completions are the high-frequency wake source — a fan-out's
        # burst coalesces into one re-drive when the window setting is on.
        await defer_run_wake(write.wake_run_id, batch=True)
    elif write.wake_session_id is not None and write.account_id is not None:
        # Wake the caller session so its parked invoke() tool task harvests the answer
        # (its await_session is also self-subscribed to this channel — wake or NOTIFY,
        # whichever lands first, drives the harvest).
        from aios.jobs.app import defer_wake

        await defer_wake(
            pool, write.wake_session_id, cause="invoke_response", account_id=write.account_id
        )
    return write.outcome


async def fail_all_open_requests(
    pool: Any, session_id: str, *, account_id: str, error: dict[str, Any]
) -> None:
    """Error **every** still-open request on a session that can no longer answer
    them — the harness erroring path, where the model failed past its retry budget,
    so a dead child must not leave its callers hung.

    Each request is funnelled through :func:`respond_to_request` (the pool-level
    response edge), so the child-ness / fail-closed / wake logic lives in exactly
    one place. A no-op for any session that owes nothing — its open set is empty,
    including every non-child session.
    """
    async with pool.acquire() as conn:
        open_ids = await queries.get_open_request_ids(conn, session_id, account_id=account_id)
    for request_id in open_ids:
        await respond_to_request(pool, session_id, request_id=request_id, outcome=Err(error=error))


async def _finish(
    session_id: str, *, request_id: Any, outcome: Outcome
) -> dict[str, Any] | ToolResult:
    # ``request_id`` is model-supplied (possibly missing or wrong-typed); a value
    # that isn't an open request resolves to ``unknown_request`` → a tool error the
    # model self-corrects from.
    status = await respond_to_request(
        runtime.require_pool(),
        session_id,
        request_id=request_id,
        outcome=outcome,
    )
    if status == "not_a_child":
        return _NOT_A_CHILD
    if status == "unknown_request":
        return _UNKNOWN_REQUEST
    # responded | duplicate — either way the request now has exactly one response.
    return {"status": "errored" if isinstance(outcome, Err) else "returned"}


def _validate_value(value: Any, schema: dict[str, Any]) -> str | None:
    """Validate a ``return`` ``value`` against the request's ``output_schema``.

    ``None`` on success; otherwise a model-facing ``output_schema_violation`` error
    enumerating every failure (mirrors :func:`aios.tools.invoke.validate_arguments`'
    formatting) so the child self-corrects and calls ``return`` again through the
    normal tool-error loop. This is the single servicer-side schema gate every
    obligation answered with ``return`` passes — self-goals (opened by
    ``create_goal``) included, since their persisted ``output_schema`` is read off
    the same ``request_opened`` edge.
    """
    errors = sorted(
        jsonschema.Draft202012Validator(schema).iter_errors(value),
        key=lambda e: list(e.absolute_path),
    )
    if not errors:
        return None
    lines = [
        "output_schema_violation: `value` does not match the request's required "
        f"output_schema. You sent: {json.dumps(value)}",
        "Errors:",
    ]
    for err in errors:
        path = ".".join(str(p) for p in err.absolute_path)
        lines.append(f"  - at {'value.' + path if path else 'value'}: {err.message}")
    lines.append("Fix `value` to match the schema shown with the request and call return again.")
    return "\n".join(lines)


async def _enforce_output_schema(session_id: str, request_id: Any, value: Any) -> str | None:
    """Validate ``value`` against the schema this request demands, if any.

    Returns a model-facing error string to bounce back (the child retries), or
    ``None`` to proceed. A non-str ``request_id`` (or one matching no request) and a
    non-child session resolve to no schema, leaving the rejection to
    :func:`respond_to_request` (``unknown_request`` / ``not_a_child``); a request with
    no ``output_schema`` (the common case) also passes.
    """
    if not isinstance(request_id, str):
        return None
    async with runtime.require_pool().acquire() as conn:
        schema = await queries.get_request_output_schema(conn, session_id, request_id=request_id)
    return None if schema is None else _validate_value(value, schema)


def _closed_request_message(outcome: Outcome, closed_at: Any) -> str:
    """The terminal stop message for a `return`/`error` call that targets an
    ALREADY-answered request (#1773 defect 1).

    The ``deadline timeout`` wording is the exact string the incident's replay
    eval (Arm D, 24/24 real trap points → 24/24 clean stops) validated as
    actually stopping the model — use it verbatim for that case. Any other
    closing (a duplicate self-answer, `no_return`, the child going away) still
    gets a truthful, equally terminal message; it just isn't the specific
    wording the eval pinned, since claiming "deadline timeout" for a close the
    request itself caused would be false.
    """
    ts = closed_at.isoformat() if hasattr(closed_at, "isoformat") else str(closed_at)
    is_timeout = isinstance(outcome, Err) and outcome.error.get("kind") == "timeout"
    qualifier = f"deadline timeout at {ts}" if is_timeout else f"at {ts}"
    return (
        f"this request was already answered ({qualifier}); "
        "do not call return again — end your turn."
    )


async def _closed_request_error(session_id: str, request_id: Any) -> ToolResult | None:
    """If ``request_id`` names a request that's ALREADY closed, the terminal
    :class:`ToolResult` error `return`/`error` must surface BEFORE anything else
    (schema validation included) — the #1773 liveness-first fix. ``None`` means
    "proceed normally": the request is still open, or ``request_id`` doesn't
    resolve to any request at all (the existing ``unknown_request`` path downstream
    handles that untouched).

    Checking liveness first (rather than after schema validation, as before) means
    a child blindly re-answering a request that closed out from under it — e.g. the
    caller's await deadline wrote a timeout response before the child's first
    `return` landed — gets ONE clear stop signal instead of an endless
    ``output_schema_violation`` bounce that never mentions the request is dead.
    """
    if not isinstance(request_id, str):
        return None
    async with runtime.require_pool().acquire() as conn:
        closed = await queries.get_closed_request(conn, session_id, request_id=request_id)
    if closed is None:
        return None
    outcome, closed_at = closed
    return ToolResult(content=_closed_request_message(outcome, closed_at), is_error=True)


async def return_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any] | ToolResult:
    request_id = arguments.get("request_id")
    # Liveness FIRST (#1773 defect 1): a closed request short-circuits here, before
    # schema validation ever runs — otherwise a child answering a dead request that
    # happens to send a schema-invalid value loops forever on the schema bounce and
    # never learns the request is closed.
    closed_error = await _closed_request_error(session_id, request_id)
    if closed_error is not None:
        return closed_error
    # Structured output: when the request demanded an output_schema, the value must
    # match it. A mismatch is a tool error the child retries on (no response written),
    # exactly like a malformed tool arg — so the workflow only ever harvests a
    # schema-valid value. error() is unconstrained (a child that can't conform bails).
    schema_error = await _enforce_output_schema(session_id, request_id, arguments.get("value"))
    if schema_error is not None:
        return ToolResult(content=schema_error, is_error=True)
    return await _finish(
        session_id,
        request_id=request_id,
        outcome=Ok(result=arguments.get("value")),
    )


async def error_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any] | ToolResult:
    request_id = arguments.get("request_id")
    # Same liveness-first short-circuit as return_handler — see #1773 defect 1.
    closed_error = await _closed_request_error(session_id, request_id)
    if closed_error is not None:
        return closed_error
    return await _finish(
        session_id,
        request_id=request_id,
        outcome=Err(error={"message": arguments.get("message")}),
    )


def workflow_completion_tool_specs() -> list[dict[str, Any]]:
    """The chat-completions tool entries for ``return``/``error`` — injected into
    a workflow child's tool list by ``compute_step_prelude``."""
    return [openai_tool_entry(registry.get(name)) for name in (RETURN_TOOL_NAME, ERROR_TOOL_NAME)]


def _register() -> None:
    registry.register(
        name=RETURN_TOOL_NAME,
        description=RETURN_DESCRIPTION,
        parameters_schema=_RETURN_SCHEMA,
        handler=return_handler,
        transport="agent_tool",
    )
    registry.register(
        name=ERROR_TOOL_NAME,
        description=ERROR_DESCRIPTION,
        parameters_schema=_ERROR_SCHEMA,
        handler=error_handler,
        transport="agent_tool",
    )


_register()
