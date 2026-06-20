"""The **session caller surface** (#1127) — model-only ``call_*`` builtins.

The building block is **"call a session"**: write a trusted request edge
(#1123 ``request_opened`` with ``caller={kind:"session", id:<this session>}``)
into an **existing same-account session** and **park** until it answers
``{ok | error}``. Calling an **agent** or a **workflow** is thin porcelain on
top — *create the servicer, then call it*:

* ``call_session(session_id, input, output_schema)`` — the primitive: inject the
  edge into an existing same-account session, park, resolve via the kind-agnostic
  resolver (#1126, ``derive_response``) to a schema-conforming ``{ok | error}``.
* ``call_agent(agent_id, input, output_schema)`` — ``create_session`` from the
  agent **then** call it (the two steps stay fused in ``service.invoke``).
* ``call_workflow(workflow_id, input, output_schema)`` — ``create_run`` then
  await it (a run is the servicer; single-shot by nature).

All three return a **single-shot handle** (one ``request_id``, one resolution;
re-asking = a fresh invoke) and **park as an implicit-async tool task** — the
caller session stays responsive to user messages while parked (never #772's
blocking long-poll). The park reuses the existing await primitives
(``await_session`` for a session servicer, ``await_run`` for a run) re-polled in
a loop; the response-write NOTIFYs the servicer's channel the park subscribes to.

**Authorization = same-account.** A cross-account ``session_id``/``agent_id``/
``workflow_id`` 404s (``NotFoundError``) before any edge is written — the
account-scoped resolve/create constructors enforce it. That is the only boundary.

**Identity is load-bearing:** the trusted **caller** id is the harness-supplied
executing ``session_id`` (``invoke_builtin(session_id, …)``), NEVER model input.
The ``session_id``/``agent_id``/``workflow_id`` argument is the *target* resource
(scoped by same-account auth), not the caller. Every arg model is
``additionalProperties: false`` so an injected ``caller``/``account_id`` is
rejected before the handler runs.

All register ``transport="agent_tool"`` (model-only; the CLI broker refuses them).

**At-least-once on worker crash.** Like the API caller (#1128, a retried POST), an
``invoke*`` call is not idempotent across a worker restart mid-park: the fire-and-forget
tool task outlives the step body, so a crash before its ``tool_result`` lands re-dispatches
the handler and writes a *second* request edge into the target. Single-shot is the
per-call contract, not a crash-exactly-once guarantee — the same stance as every other
non-deterministic builtin. Deterministic request-id keying (call-key dedup, as the
workflow ``agent()`` path does) is a future hardening, not v1 scope.
"""

from __future__ import annotations

from typing import Any

import jsonschema
from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from aios.config import get_settings
from aios.harness import runtime
from aios.ids import REQUEST, make_id
from aios.models.sessions import SessionAwaitResponse
from aios.services import sessions as sessions_service
from aios.services import workflows as wf_service
from aios.tools.invoke import ToolBail
from aios.tools.registry import ToolResult, registry

# Per-park await budget. The tool task is fire-and-forget (implicit-async), so a
# long park never blocks the caller's other turns; we re-poll in a loop so a
# single LISTEN drop can't strand the resolution. Bounded so an unbounded park
# can't pin a connection forever between polls.
_AWAIT_POLL_SECONDS = 300.0


# ─── argument models ─────────────────────────────────────────────────────────


class _CallSessionArgs(BaseModel):
    """``invoke`` arguments — the **target** session id plus the request payload.

    ``extra="forbid"``: the trusted *caller* id is the executing session the
    harness supplies, never a field here — an injected ``caller``/``account_id``
    is rejected before the handler runs.
    """

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(description="The id of the same-account session to invoke.")
    input: Any = Field(
        default=None, description="The request payload delivered to the session (JSON or a string)."
    )
    output_schema: dict[str, Any] | None = Field(
        default=None,
        description="Optional JSON Schema the answer's value must satisfy (validated fail-loud).",
    )


class _CallAgentArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str = Field(description="The id of the same-account agent to spawn and invoke.")
    input: Any = Field(default=None, description="The request payload (JSON or a string).")
    output_schema: dict[str, Any] | None = Field(
        default=None,
        description="Optional JSON Schema the answer's value must satisfy (validated fail-loud).",
    )


class _CallWorkflowArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workflow_id: str = Field(description="The id of the same-account workflow to run and await.")
    input: Any = Field(default=None, description="The run input (JSON or a string).")
    output_schema: dict[str, Any] | None = Field(
        default=None,
        description="Optional JSON Schema the run output must satisfy (validated fail-loud).",
    )


def _parse[M: BaseModel](model: type[M], arguments: dict[str, Any]) -> M:
    try:
        return model.model_validate(arguments)
    except PydanticValidationError as exc:
        raise ToolBail(f"invalid arguments: {exc}") from exc


# ─── shared result shaping ───────────────────────────────────────────────────


def _validate_output(value: Any, schema: dict[str, Any] | None) -> ToolResult | None:
    """Validate the resolved ``value`` against ``output_schema`` (fail-loud).

    ``None`` on success (or no schema); otherwise a model-visible error ToolResult
    (``output_schema_violation``) so the caller sees a non-conforming answer as an
    error rather than silently accepting it — mirrors ``workflow_completion``'s
    ``return`` enforcement, but on the *caller* side for a run/peer answer that
    bypassed the servicer's own ``return`` schema gate.
    """
    if schema is None:
        return None
    errors = sorted(
        jsonschema.Draft202012Validator(schema).iter_errors(value),
        key=lambda e: list(e.absolute_path),
    )
    if not errors:
        return None
    detail = "; ".join(
        f"at {'.'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}" for e in errors
    )
    return ToolResult(
        content=f"output_schema_violation: the answer does not match output_schema ({detail})",
        is_error=True,
    )


def _ok_result(result: Any) -> dict[str, Any]:
    return {"ok": result}


def _error_result(error: dict[str, Any] | None) -> ToolResult:
    return ToolResult(content={"error": error}, is_error=True)


# ─── park loops ──────────────────────────────────────────────────────────────


async def _park_on_session(
    pool: Any, *, session_id: str, account_id: str, request_id: str
) -> SessionAwaitResponse:
    """Park (implicit-async) until the servicer session answers ``request_id``.

    Re-polls the existing ``await_session`` primitive (request_id mode → resolves
    via ``derive_response``, #1126) so a single LISTEN drop can't strand us. As a
    fire-and-forget tool task the caller stays responsive to user messages while
    parked. No new blocking long-poll is introduced — this rides the shipped await.
    """
    db_url = get_settings().db_url
    while True:
        resp = await sessions_service.await_session(
            pool,
            db_url,
            session_id,
            account_id=account_id,
            request_id=request_id,
            watermark=None,
            timeout_seconds=_AWAIT_POLL_SECONDS,
        )
        if resp.done:
            return resp


async def _park_on_run(pool: Any, *, run_id: str, account_id: str) -> Any:
    """Park (implicit-async) until the run reaches a terminal state."""
    db_url = get_settings().db_url
    while True:
        resp = await wf_service.await_run(
            pool, db_url, run_id, account_id=account_id, timeout_seconds=_AWAIT_POLL_SECONDS
        )
        if resp.done:
            return resp


# ─── handlers ────────────────────────────────────────────────────────────────


async def call_session_handler(
    session_id: str, arguments: dict[str, Any]
) -> dict[str, Any] | ToolResult:
    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    args = _parse(_CallSessionArgs, arguments)
    # Write the trusted edge into the EXISTING same-account session (404s a foreign
    # target before any edge is written). caller names THIS session.
    handle = await sessions_service.invoke(
        pool,
        account_id=account_id,
        target_kind="session",
        target=args.session_id,
        input=args.input,
        output_schema=args.output_schema,
        caller={"kind": "session", "id": session_id},
    )
    resp = await _park_on_session(
        pool, session_id=handle.servicer_id, account_id=account_id, request_id=handle.request_id
    )
    if resp.is_error:
        return _error_result(resp.error)
    violation = _validate_output(resp.result, args.output_schema)
    return violation if violation is not None else _ok_result(resp.result)


async def call_agent_handler(
    session_id: str, arguments: dict[str, Any]
) -> dict[str, Any] | ToolResult:
    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    args = _parse(_CallAgentArgs, arguments)
    # Porcelain = create_session(from the agent) + invoke that fresh session. The
    # child inherits THIS session's environment (a caller-chosen env id would be a
    # cross-tenant attack surface — same stance as create_run).
    session = await sessions_service.get_session_basic(pool, session_id, account_id=account_id)
    handle = await sessions_service.invoke(
        pool,
        account_id=account_id,
        target_kind="agent",
        target=args.agent_id,
        input=args.input,
        output_schema=args.output_schema,
        environment_id=session.environment_id,
        crypto_box=runtime.require_crypto_box(),
        caller={"kind": "session", "id": session_id},
    )
    resp = await _park_on_session(
        pool, session_id=handle.servicer_id, account_id=account_id, request_id=handle.request_id
    )
    if resp.is_error:
        return _error_result(resp.error)
    violation = _validate_output(resp.result, args.output_schema)
    return violation if violation is not None else _ok_result(resp.result)


async def call_workflow_handler(
    session_id: str, arguments: dict[str, Any]
) -> dict[str, Any] | ToolResult:
    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    args = _parse(_CallWorkflowArgs, arguments)
    # Porcelain = create_run + await it (the run is the servicer; single-shot). The
    # run inherits THIS session's environment + lineage and is launched by it.
    session = await sessions_service.get_session_basic(pool, session_id, account_id=account_id)
    run = await wf_service.create_run(
        pool,
        account_id=account_id,
        workflow_id=args.workflow_id,
        environment_id=session.environment_id,
        input=args.input,
        launcher_session_id=session_id,
        parent_run_id=session.parent_run_id,
        request_id=make_id(REQUEST),
        caller={"kind": "session", "id": session_id, "awaited": True},
        request_output_schema=args.output_schema,
    )
    resp = await _park_on_run(pool, run_id=run.id, account_id=account_id)
    if resp.is_error:
        return _error_result(resp.error)
    violation = _validate_output(resp.output, args.output_schema)
    return violation if violation is not None else _ok_result(resp.output)


# ─── descriptions + registration ─────────────────────────────────────────────

CALL_SESSION_DESCRIPTION = (
    "Call an existing same-account session: deliver `input` to it as a trusted "
    "request and wait for its single answer ({ok: value} on success, an error "
    "otherwise). The request is invisible to any human chatting in that session. "
    "Optionally pass `output_schema` (JSON Schema) the answer's value must satisfy. "
    "Single-shot: each call is a fresh request. You stay responsive while waiting."
)
CALL_AGENT_DESCRIPTION = (
    "Spawn a fresh session from one of your agents and call it with `input`, "
    "waiting for its single answer ({ok: value} or an error). The new session runs "
    "in your own environment. Optionally constrain the answer with `output_schema`. "
    "Single-shot; you stay responsive while waiting."
)
CALL_WORKFLOW_DESCRIPTION = (
    "Launch a run of one of your workflows with `input` and wait for its result "
    "({ok: output} on completion, an error if it errored/was cancelled). The run "
    "uses your own environment. Optionally constrain the output with `output_schema` "
    "(a non-conforming output is reported as an error). Single-shot per call."
)


def _register() -> None:
    registry.register(
        name="call_session",
        description=CALL_SESSION_DESCRIPTION,
        parameters_schema=_CallSessionArgs.model_json_schema(),
        handler=call_session_handler,
        transport="agent_tool",
    )
    registry.register(
        name="call_agent",
        description=CALL_AGENT_DESCRIPTION,
        parameters_schema=_CallAgentArgs.model_json_schema(),
        handler=call_agent_handler,
        transport="agent_tool",
    )
    registry.register(
        name="call_workflow",
        description=CALL_WORKFLOW_DESCRIPTION,
        parameters_schema=_CallWorkflowArgs.model_json_schema(),
        handler=call_workflow_handler,
        transport="agent_tool",
    )


_register()
