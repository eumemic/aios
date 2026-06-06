"""Workflows: read models for the durable runtime core (Block 1).

A *workflow* is a deterministic-Python orchestrator (the dual of an agent):
``workflows`` are immutable versioned definitions; a ``WfRun`` is a durable
execution instance whose state lives entirely in its append-only journal
(``WfRunEvent``); ``WfRunSignal`` is the side-marker an external resume writes
so the journal keeps a single writer.

These are internal read views (they carry ``account_id``); request/echo models
and the public HTTP surface arrive in a later block. The runtime is driven
internally via ``services.workflows`` + integration tests for now.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel

WfRunStatus = Literal["pending", "running", "suspended", "completed", "errored"]
WfRunEventType = Literal["run_started", "call_started", "call_result", "run_completed"]
WfRunSignalKind = Literal["gate_resume", "child_done"]


class Workflow(BaseModel):
    """An immutable, versioned workflow definition."""

    id: str
    account_id: str
    name: str
    version: int
    script: str
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
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
    parent_run_id: str | None = None
    script: str
    script_sha: str
    status: WfRunStatus
    input: dict[str, Any] | None = None
    output: dict[str, Any] | None = None
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
    result: dict[str, Any] | None = None
    delivered_at: datetime
