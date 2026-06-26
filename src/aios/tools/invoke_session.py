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
blocking long-poll). The park rides the one awaiter (``await_task``,
dispatching session vs run) re-polled in a loop; the response-write NOTIFYs the
servicer's channel the park subscribes to.

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

**Crash recovery = re-park, not error (#1431).** The request edge is written *before* the
park and carries the launching ``tool_call_id`` on its ``caller`` (via :func:`_caller`), so
the caller↔servicer link is durable; only the fire-and-forget park task is lost on a worker
crash. The harness does NOT re-run the handler (no builtin re-dispatch path) — instead the
ghost-repair sweep re-derives the servicer from that edge
(``queries.find_parked_servicer``) and **re-parks** the lost ``tool_call_id`` via
:func:`_park_and_resolve`: a pure read of durable state that re-attaches to the servicer's
exactly-once response and lands the original tool result. If the launch crashed *before* its
edge was durable, no servicer is found and the call resolves to a retryable error instead. A
*second* request edge appears only if the **model** retries that error — never from a harness
re-dispatch. Single-shot is the per-call contract.
"""

from __future__ import annotations

from typing import Any

import jsonschema
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic import ValidationError as PydanticValidationError

from aios.config import get_settings
from aios.harness import runtime
from aios.models.tasks import AwaitResponse
from aios.models.workflows import InlineScriptBody
from aios.services import sessions as sessions_service
from aios.services import tasks as tasks_service
from aios.services import workflows as wf_service
from aios.tools.invoke import ToolBail, current_tool_call_id
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

    workflow_id: str | None = Field(
        default=None,
        description=(
            "The id of the same-account workflow to run and await. Supply EITHER this or "
            "`inline` (exactly one). Omit to launch an inline one-shot run."
        ),
    )
    inline: InlineScriptBody | None = Field(
        default=None,
        description=(
            "Inline-script body for an anonymous one-shot run (T5): `{script, schemas, "
            "surface}`. NO workflow is registered; the run snapshots the script and clamps "
            "the declared surface to your own. Supply EITHER this or `workflow_id`."
        ),
    )
    input: Any = Field(default=None, description="The run input (JSON or a string).")
    output_schema: dict[str, Any] | None = Field(
        default=None,
        description="Optional JSON Schema the run output must satisfy (validated fail-loud).",
    )
    vault_ids: list[str] = Field(
        default_factory=list,
        description="Vault ids to bind to the run — a subset of the vaults bound to you.",
    )
    budget_usd: float | None = Field(
        default=None, gt=0, description="Optional shared USD spend ceiling for the run."
    )

    @model_validator(mode="after")
    def _validate_source_arm(self) -> _CallWorkflowArgs:
        if (self.workflow_id is None) == (self.inline is None):
            got = "both" if self.inline is not None else "neither"
            raise ValueError(f"exactly one of workflow_id or inline must be provided (got {got})")
        return self


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


async def _park_on_task(
    pool: Any,
    *,
    servicer_kind: tasks_service.ServicerKind,
    servicer_id: str,
    request_id: str | None,
    account_id: str,
) -> AwaitResponse:
    """Park (implicit-async) until the servicer answers, via the one awaiter.

    Re-polls :func:`await_task` (session → ``derive_response`` on
    ``request_id``; run → terminal row) so a single LISTEN drop can't strand us. As
    a fire-and-forget tool task the caller stays responsive to user messages while
    parked. ``outcome`` is non-None exactly when the task is terminal.
    """
    db_url = get_settings().db_url
    while True:
        resp = await tasks_service.await_task(
            pool,
            db_url,
            servicer_kind=servicer_kind,
            servicer_id=servicer_id,
            request_id=request_id,
            account_id=account_id,
            timeout_seconds=_AWAIT_POLL_SECONDS,
        )
        if resp.outcome is not None:
            return resp


async def _park_and_resolve(
    pool: Any,
    *,
    servicer_kind: tasks_service.ServicerKind,
    servicer_id: str,
    request_id: str | None,
    account_id: str,
    output_schema: dict[str, Any] | None,
) -> dict[str, Any] | ToolResult:
    """Park on the servicer, then resolve to a model-visible ``{ok | error}``.

    The shared tail of every ``call_*`` handler AND the crash-resume path
    (:func:`aios.harness.tool_dispatch.relaunch_parked_task`). It is a **pure read**
    of durable state — re-entrant, so re-parking after a worker restart re-reads the same
    servicer edge with zero side effects; the resolved tool result is the only write it
    drives. ``output_schema`` is validated caller-side (a run/peer answer can bypass the
    servicer's own ``return`` gate).
    """
    resp = await _park_on_task(
        pool,
        servicer_kind=servicer_kind,
        servicer_id=servicer_id,
        request_id=request_id,
        account_id=account_id,
    )
    if resp.outcome != "ok":
        return _error_result(resp.error)
    violation = _validate_output(resp.result, output_schema)
    return violation if violation is not None else _ok_result(resp.result)


def _caller(session_id: str) -> dict[str, Any]:
    """Trusted caller provenance for a model-launched task: THIS session's id plus
    the launching ``tool_call_id`` (#1431).

    The ``tool_call_id`` rides onto the servicer's edge ``caller`` so a parked task
    can be re-derived from durable state and re-parked after a worker restart
    (``queries.find_parked_servicer``). Omitted (not written as ``null``) when no tool
    context is set — e.g. a non-dispatch caller — keeping the edge clean.
    """
    caller: dict[str, Any] = {"kind": "session", "id": session_id}
    tool_call_id = current_tool_call_id()
    if tool_call_id is not None:
        caller["tool_call_id"] = tool_call_id
    return caller


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
        caller=_caller(session_id),
    )
    return await _park_and_resolve(
        pool,
        servicer_kind="session",
        servicer_id=handle.servicer_id,
        request_id=handle.request_id,
        account_id=account_id,
        output_schema=args.output_schema,
    )


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
        caller=_caller(session_id),
    )
    return await _park_and_resolve(
        pool,
        servicer_kind="session",
        servicer_id=handle.servicer_id,
        request_id=handle.request_id,
        account_id=account_id,
        output_schema=args.output_schema,
    )


async def call_workflow_handler(
    session_id: str, arguments: dict[str, Any]
) -> dict[str, Any] | ToolResult:
    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    args = _parse(_CallWorkflowArgs, arguments)
    # Porcelain = launch the run as an awaited servicer + park on it (single-shot). The
    # run inherits THIS session's environment + lineage and is launched by it.
    session = await sessions_service.get_session_basic(pool, session_id, account_id=account_id)
    inline = (
        wf_service.InlineScript(
            script=args.inline.script,
            input_schema=args.inline.input_schema,
            output_schema=args.inline.output_schema,
            tools=args.inline.tools,
            mcp_servers=args.inline.mcp_servers,
            http_servers=args.inline.http_servers,
        )
        if args.inline is not None
        else None
    )
    run, _request_id = await wf_service.launch_awaited_run(
        pool,
        account_id=account_id,
        workflow_id=args.workflow_id,
        inline=inline,
        environment_id=session.environment_id,
        input=args.input,
        caller=_caller(session_id),
        output_schema=args.output_schema,
        launcher_session_id=session_id,
        parent_run_id=session.parent_run_id,
        vault_ids=args.vault_ids,
        budget_usd=args.budget_usd,
    )
    return await _park_and_resolve(
        pool,
        servicer_kind="run",
        servicer_id=run.id,
        request_id=None,
        account_id=account_id,
        output_schema=args.output_schema,
    )


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
    "Launch a run and wait for its result ({ok: output} on completion, an error if it "
    "errored/was cancelled). Run EITHER a registered workflow (`workflow_id`) OR an "
    "inline one-shot script (`inline`: {script, schemas, surface}) with no workflow "
    "registered — supply exactly one; inline is the ergonomic default for one-shot work. "
    "An inline run's declared surface is clamped to your own. The run uses your own "
    "environment. Optionally attach `vault_ids` (a subset of your own vaults), set a "
    "shared `budget_usd` spend ceiling, and constrain the output with `output_schema` (a "
    "non-conforming output is reported as an error). Single-shot per call."
)


def _register() -> None:
    # ``resumable=True``: each handler parks (:func:`_park_and_resolve`) on a durable
    # servicer edge — a pure-await, safe for the ghost-repair sweep to re-park rather
    # than error-repair on crash recovery (#1431). The sweep derives its discriminant
    # from this flag; no separate name list to keep in lockstep.
    registry.register(
        name="call_session",
        description=CALL_SESSION_DESCRIPTION,
        parameters_schema=_CallSessionArgs.model_json_schema(),
        handler=call_session_handler,
        transport="agent_tool",
        resumable=True,
    )
    registry.register(
        name="call_agent",
        description=CALL_AGENT_DESCRIPTION,
        parameters_schema=_CallAgentArgs.model_json_schema(),
        handler=call_agent_handler,
        transport="agent_tool",
        resumable=True,
    )
    registry.register(
        name="call_workflow",
        description=CALL_WORKFLOW_DESCRIPTION,
        parameters_schema=_CallWorkflowArgs.model_json_schema(),
        handler=call_workflow_handler,
        transport="agent_tool",
        resumable=True,
    )


_register()
