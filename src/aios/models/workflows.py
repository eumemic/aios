"""Workflows: read models for the durable runtime core (Block 1).

A *workflow* is a deterministic-Python orchestrator (the dual of an agent):
``workflows`` are versioned definitions (updated in place, agent-style); a ``WfRun``
is a durable execution instance whose state lives entirely in its append-only journal
(``WfRunEvent``) and which pins its workflow's script + declared surface at launch;
``WfRunSignal`` is the side-marker an external resume writes so the journal keeps a
single writer.

The read views below carry ``account_id`` (internal); the ``*Create`` / resume
request models at the bottom back the public HTTP surface (Block 3). Responses
reuse the read views directly, the way ``Agent``/``Session`` do.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aios.models.agents import HttpServerSpec, McpServerSpec, ToolSpec, validate_http_servers

WfRunStatus = Literal["pending", "running", "suspended", "completed", "errored", "cancelled"]
WfRunEventType = Literal[
    "run_started",
    "call_started",
    "call_result",
    "run_completed",
    "annotation",
    "frontier_deferred",
]
WfRunSignalKind = Literal["gate_resume", "child_done", "cancel", "tool_result"]

# The terminal run statuses — monotonic: once here, a run never leaves. The one source for
# every "is this run done?" check (the step loop's early-out, the SSE stream's close, the await
# predicate). ``cancelled`` is terminal too (a user cancel finalizes the run).
TERMINAL_RUN_STATUSES: frozenset[str] = frozenset({"completed", "errored", "cancelled"})


class Workflow(BaseModel):
    """A versioned workflow definition (updated in place; ``version`` bumps per change)."""

    id: str
    account_id: str
    name: str
    version: int
    script: str
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    description: str | None = None  # optional human blurb (the agent ``description`` analog)
    # The declared tool surface — the verbatim agent envelope. A run reaches these
    # (authed MCP / http_request / builtins) directly via ``tool()`` (a later slice);
    # an agent authoring a workflow may only declare a subset of its own surface.
    tools: list[ToolSpec] = Field(default_factory=list)
    mcp_servers: list[McpServerSpec] = Field(default_factory=list)
    http_servers: list[HttpServerSpec] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None = None


class WfRun(BaseModel):
    """A durable workflow execution instance.

    ``script`` is the run's own immutable snapshot of the workflow source at
    creation time (``script_sha`` is its hash); every wake execs exactly this.
    ``tools``/``mcp_servers``/``http_servers`` are the matching snapshot of the
    declared tool surface — pinned at launch like ``script``, so a later
    ``update_workflow`` never shifts an in-flight run's authority.
    ``status`` is persisted (unlike sessions): the run loop writes
    ``suspended``/``completed``/``errored``.
    """

    id: str
    workflow_id: str
    account_id: str
    environment_id: str  # the run binds to an environment; agent() children inherit it
    # Lineage + the vertical depth cap's walk key. Set by nested workflow()
    # launches AND by trigger fires (#819): a run_completion fire threads the
    # completing run's id, a timer fire threads the owner session's own parent
    # run — so reactive cascades and self-fire loops are depth-bounded.
    parent_run_id: str | None = None
    # The agent session that launched this run (None = operator/HTTP). Lineage, plus
    # the per-launcher fan-out cap's count key.
    launcher_session_id: str | None = None
    script: str
    script_sha: str
    tools: list[ToolSpec] = Field(default_factory=list)
    mcp_servers: list[McpServerSpec] = Field(default_factory=list)
    http_servers: list[HttpServerSpec] = Field(default_factory=list)
    status: WfRunStatus
    input: Any = None  # arbitrary JSON: a workflow's input need not be an object
    output: Any = None  # arbitrary JSON: the script's return value
    last_event_seq: int
    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None = None


class WfRunEvent(BaseModel):
    """One row of a run's append-only journal (the replay-with-memo source).

    ``call_key`` is set for ``call_started``/``call_result`` (the memo key) and for
    ``annotation`` (the branch-local dedup key that makes ``log()``/``phase()``
    emit-once across replays); it is ``None`` for the ``run_started``/``run_completed``
    bookends. An ``annotation`` is a journaled progress marker (``payload`` =
    ``{"kind": "log" | "phase", "text": ...}``), not a capability call.
    """

    id: str
    run_id: str
    seq: int
    type: WfRunEventType
    call_key: str | None = None
    payload: dict[str, Any]
    created_at: datetime


class WfRunSignal(BaseModel):
    """A side-marker for an external resume (gate) or, later, child completion.

    The run step's pre-replay harvest reads these and journals the matching
    ``call_result``; the signal row itself is never the durable result.
    """

    run_id: str
    call_key: str
    kind: WfRunSignalKind
    result: Any = None  # arbitrary JSON: the externally-delivered resume value
    delivered_at: datetime


class WfRunWaitResponse(BaseModel):
    """Response for ``GET /v1/runs/{run_id}/wait`` — the run's completion record, or its
    current (non-terminal) state if the wait timed out.

    Deliberately mirrors the ``{result, is_error, error}`` shape of a request response
    (``derive_response``) so the ``await`` primitive's two backings (run-terminal and, later,
    session-request) share one envelope. Poll until ``done``: a still-running run returns
    ``done=False`` with its live ``run_status`` (``running``/``suspended``/…); call again to
    keep blocking.
    """

    run_status: WfRunStatus
    done: bool  # run_status in TERMINAL_RUN_STATUSES (completed/errored/cancelled)
    output: Any = None  # the run's return value (on completed; None otherwise)
    is_error: bool = False  # run_status == errored
    error: dict[str, Any] | None = None  # the run_completed event's {kind} (on errored)


# ─── request models (the public HTTP surface) ────────────────────────────────

WORKFLOW_SCRIPT_CONTRACT = """Workflow script contract:
- Entry point: define `async def main(input)`. A run's output is the value returned by
  `main`.
- Injected capability API, available without imports:
  - `agent(agent_id, input, output_schema=None)`: invoke an agent and await its result.
  - `tool(name, input)`: invoke a declared tool; tool errors are returned, not raised.
  - `gate()`: suspend until an external resume delivers a value.
  - `parallel(thunks)`: run zero-argument callables concurrently (for example,
    `lambda: agent(...)`). A failed agent branch yields `None` at the barrier instead
    of raising. Fan-out width is capped by `MAX_PARALLEL_FANOUT` (currently 1000).
  - `pipeline(items, *stages)`: run each item through staged transforms concurrently.
  - `log(msg)`: record progress on the run journal.
- Shell execution: `tool('bash', {"command": str, "timeout_s": float | None})` runs the
  command in a per-run sandbox (provisioned lazily on first use, in the run's
  environment). `bash` must be a member of the workflow's declared tools or the call
  resolves to a `{"error": ...}` value. Result: `{exit_code, stdout, stderr, timed_out,
  truncated}` — a nonzero exit or in-command timeout is a successful result to branch
  on, not an error.
- Crash semantics: at-least-once at the call boundary. A capability call interrupted by
  a crash re-runs on resume; completed calls never re-run. The sandbox filesystem is
  ephemeral scratch — write re-run-tolerant commands (e.g. `rm -rf dir && git clone ...`).
- Irreversible external effects (a POST that charges, sends, or publishes): the bash
  environment exposes `$AIOS_IDEMPOTENCY_KEY` (stable across crash re-runs of the same
  call, distinct per call) — pass it to the external service as an idempotency key so
  the service drops a re-fired duplicate, or knowingly accept at-least-once.
- Partition rule: put re-run-tolerant mechanical work in `tool('bash')`; put work whose
  uncertain completion needs judgment to resolve inside `agent(...)`.
- Environment: deterministic, credential-free, isolated child process. Imports are
  restricted to a curated stdlib allowlist. No network or filesystem side channels are
  available beyond the capability API.

Minimal example:
```python
async def main(input):
    result = await agent(
        input["agent_id"],
        {"task": input["task"]},
        None,
    )
    return result
```
"""


class WorkflowCreate(BaseModel):
    """Request body for ``POST /v1/workflows`` — a new workflow definition at v1."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=128)
    script: str = Field(description=WORKFLOW_SCRIPT_CONTRACT)
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    description: str | None = None
    # The declared tool surface (verbatim agent envelope). When an agent authors a
    # workflow, these must be a subset of the creating agent's own surface; the HTTP
    # path is unattenuated operator authority.
    tools: list[ToolSpec] = Field(default_factory=list)
    mcp_servers: list[McpServerSpec] = Field(default_factory=list)
    http_servers: list[HttpServerSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_http_servers(self) -> WorkflowCreate:
        validate_http_servers(self.http_servers)
        return self


class WorkflowUpdate(BaseModel):
    """Request body for ``PUT /v1/workflows/{id}`` — update in place, bumping ``version``.

    ``version`` is the optimistic-concurrency token: it must match the workflow's
    current version or the update 409s (re-fetch and retry). Omitted fields are
    preserved — nullable fields (``input_schema``/``output_schema``/``description``)
    can therefore be replaced but never cleared back to null, as on ``AgentUpdate``.
    An identical update is a no-op (no bump). There is no version-snapshot table —
    a run pins ``script`` + the declared surface onto itself at launch, so in-flight
    runs never observe an update. (The ``AgentUpdate`` shape, minus history.)
    """

    model_config = ConfigDict(extra="forbid")

    version: int
    name: str | None = Field(default=None, min_length=1, max_length=128)
    script: str | None = Field(default=None, description=WORKFLOW_SCRIPT_CONTRACT)
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    description: str | None = None
    tools: list[ToolSpec] | None = None
    mcp_servers: list[McpServerSpec] | None = None
    http_servers: list[HttpServerSpec] | None = None

    @model_validator(mode="after")
    def _validate_http_servers(self) -> WorkflowUpdate:
        if self.http_servers is not None:
            validate_http_servers(self.http_servers)
        return self


class WfRunCreate(BaseModel):
    """Request body for ``POST /v1/runs`` — launch a run of a workflow.

    ``input`` is arbitrary JSON (a workflow's input need not be an object). The run
    binds to ``environment_id`` (like a session), into which its ``agent()`` children
    spawn. (``launcher_session_id`` is deliberately NOT a field — trusted ids never
    ride in request bodies; the HTTP path is always an operator launch.)
    """

    model_config = ConfigDict(extra="forbid")

    workflow_id: str
    environment_id: str
    input: Any = None
    vault_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Vault ids to bind to the run for credential resolution. When an agent "
            "launches the run, these must be a subset of the launcher's own vaults; "
            "the HTTP path is unattenuated operator authority."
        ),
    )


class GateResume(BaseModel):
    """Request body for ``POST /v1/runs/{run_id}/resume`` — deliver a gate's value.

    Keyed by ``gate_nonce`` (the unguessable capability token minted into the gate's
    ``call_started`` event), not the internal ``call_key``. ``result`` is the
    externally-delivered resume value (arbitrary JSON).
    """

    model_config = ConfigDict(extra="forbid")

    gate_nonce: str
    result: Any = None
