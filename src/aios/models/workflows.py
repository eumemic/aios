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

from pydantic import BaseModel

WfRunStatus = Literal["pending", "running", "suspended", "completed", "errored", "cancelled"]
WfRunEventType = Literal["run_started", "call_started", "call_result", "run_completed"]
WfRunSignalKind = Literal["gate_resume", "child_done", "cancel"]


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


# ─── request models (the public HTTP surface) ────────────────────────────────


class WorkflowCreate(BaseModel):
    """Request body for ``POST /v1/workflows`` — a new workflow definition at v1."""

    name: str
    script: str
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    description: str | None = None


class WfRunCreate(BaseModel):
    """Request body for ``POST /v1/runs`` — launch a run of a workflow.

    ``input`` is arbitrary JSON (a workflow's input need not be an object). The run
    binds to ``environment_id`` (like a session), into which its ``agent()`` children
    spawn.
    """

    workflow_id: str
    environment_id: str
    input: Any = None


class GateResume(BaseModel):
    """Request body for ``POST /v1/runs/{run_id}/resume`` — deliver a gate's value.

    Keyed by ``gate_nonce`` (the unguessable capability token minted into the gate's
    ``call_started`` event), not the internal ``call_key``. ``result`` is the
    externally-delivered resume value (arbitrary JSON).
    """

    gate_nonce: str
    result: Any = None
