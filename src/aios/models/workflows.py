"""Workflows: read models for the durable runtime core (Block 1).

A *workflow* is a deterministic-Python orchestrator (the dual of an agent):
``workflows`` are immutable versioned definitions; a ``WfRun`` is a durable
execution instance whose state lives entirely in its append-only journal
(``WfRunEvent``); ``WfRunSignal`` is the side-marker an external resume writes
so the journal keeps a single writer.

The read views below carry ``account_id`` (internal); the ``*Create`` / resume
request models at the bottom back the public HTTP surface (Block 3). Responses
reuse the read views directly, the way ``Agent``/``Session`` do.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from aios.models.agents import HttpServerSpec, McpServerSpec, ToolSpec

WfRunStatus = Literal["pending", "running", "suspended", "completed", "errored", "cancelled"]
WfRunEventType = Literal["run_started", "call_started", "call_result", "run_completed"]
WfRunSignalKind = Literal["gate_resume", "child_done", "cancel"]

# The terminal run statuses — monotonic: once here, a run never leaves. The one source for
# every "is this run done?" check (the step loop's early-out, the SSE stream's close, the await
# predicate). ``cancelled`` is terminal too (a user cancel finalizes the run).
TERMINAL_RUN_STATUSES: frozenset[str] = frozenset({"completed", "errored", "cancelled"})


class Workflow(BaseModel):
    """An immutable, versioned workflow definition."""

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


class WfRun(BaseModel):
    """A durable workflow execution instance.

    ``script`` is the run's own immutable snapshot of the workflow source at
    creation time (``script_sha`` is its hash); every wake execs exactly this.
    ``status`` is persisted (unlike sessions): the run loop writes
    ``suspended``/``completed``/``errored``.
    """

    id: str
    workflow_id: str
    account_id: str
    environment_id: str  # the run binds to an environment; agent() children inherit it
    parent_run_id: str | None = None
    script: str
    script_sha: str
    status: WfRunStatus
    input: Any = None  # arbitrary JSON: a workflow's input need not be an object
    output: Any = None  # arbitrary JSON: the script's return value
    last_event_seq: int
    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None = None


class WfRunEvent(BaseModel):
    """One row of a run's append-only journal (the replay-with-memo source).

    ``call_key`` is set for ``call_started``/``call_result`` (the memo key) and
    ``None`` for the ``run_started``/``run_completed`` bookends.
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
    done: bool  # run_status in {completed, errored} — terminal, never reverts
    output: Any = None  # the run's return value (on completed; None on error)
    is_error: bool = False  # run_status == errored
    error: dict[str, Any] | None = None  # the run_completed event's {kind} (on errored)


# ─── request models (the public HTTP surface) ────────────────────────────────


class WorkflowCreate(BaseModel):
    """Request body for ``POST /v1/workflows`` — a new workflow definition at v1."""

    name: str
    script: str
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    description: str | None = None
    # The declared tool surface (verbatim agent envelope). When an agent authors a
    # workflow, these must be a subset of the creating agent's own surface; the HTTP
    # path is unattenuated operator authority.
    tools: list[ToolSpec] = Field(default_factory=list)
    mcp_servers: list[McpServerSpec] = Field(default_factory=list)
    http_servers: list[HttpServerSpec] = Field(default_factory=list)


class WfRunCreate(BaseModel):
    """Request body for ``POST /v1/runs`` — launch a run of a workflow.

    ``input`` is arbitrary JSON (a workflow's input need not be an object). The run
    binds to ``environment_id`` (like a session), into which its ``agent()`` children
    spawn.
    """

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

    gate_nonce: str
    result: Any = None
