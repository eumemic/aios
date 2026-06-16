"""Trace resource: a one-call linear projection of a workflow-run / session tree
(#1149).

A *trace* is **not** a new subsystem — it is a read-projection over the
invocation-edge tree the invoke-edge epic (#1122) already records. Three moves:
**walk** the parent→child edge tree from the root, **normalize** each node's
outcome into a small terminal-state enum (+ a raw ``error_kind`` passthrough),
and **interleave** the nodes' existing journals (``events`` for sessions,
``wf_run_events`` for runs).

The wire shape is a **flat list** (no recursion — it fits codegen, and no
recursive response models exist in this codebase). Causal **DFS pre-order** over
the edge tree is canonical: a child's subtree is contiguous under its parent, so
CLI indentation by ``depth`` is trivially correct. ``timestamp`` is exposed as a
per-entry column for chronology; a ``--chronological`` client re-sort by
``timestamp`` is best-effort to transaction granularity (the two journals share
no global sequence — only the causal parent→child edge is exact).

Scope = the invocation-edge tree (``agent()`` / ``invoke_workflow`` /
api-invoke). ``wake_session`` peer-pokes are **out of scope** (a peer stimulus
carrying ``wake_depth``, not a spawn — it opens no ``request_opened`` edge).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

# The normalized terminal state of a node — the small enum every caller reads at
# a glance (the centerpiece of #1149 / #1140's shared normalizer). ``suspended``
# is load-bearing: a pipeline parked at a ``gate()`` awaiting the chairman is
# ``suspended`` and that is its dominant *healthy* state — it must be
# distinguishable from actively-``running`` and from wedged.
TerminalState = Literal["ok", "errored", "cancelled", "suspended", "running"]

# A node entry is a session or run; a frame entry is one journal line under a
# node. The ``agent_call`` / ``invoke_workflow`` run-journal frames are MERGED
# into their child node (never double-listed) — a standalone ``agent_call``
# entry appears only for an orphan (a child that was never spawned / rejected
# pre-spawn).
TraceEntryKind = Literal[
    # node kinds (carry terminal_state / error_kind)
    "run",
    "session",
    # frame kinds (live under a node; parent_id = the node)
    "request",
    "response",
    "tool_call",
    "gate",
    "annotation",
    "error",
    "message",
    "agent_call",  # orphan only — a call_started with no spawned child
]


class TraceEntry(BaseModel):
    """One line of a flattened trace.

    ``depth`` is the node's DFS depth (0 = root); a CLI renders it as
    indentation. ``parent_id`` is the enclosing node's id (``None`` for the
    root). For **node** entries ``terminal_state`` / ``error_kind`` carry the
    normalized outcome; for **frame** entries they are ``None`` and ``summary``
    carries the per-kind digest. ``timestamp`` is the entry's ``created_at``
    (``transaction_timestamp()`` for journal frames) — exposed as a column for
    chronology, NOT the canonical order (see the module docstring).
    """

    timestamp: datetime | None = None
    depth: int
    parent_id: str | None = None
    kind: TraceEntryKind
    id: str
    summary: str = ""
    terminal_state: TerminalState | None = None
    error_kind: str | None = None


class TraceTruncated(BaseModel):
    """The typed marker that the walk hit its node-count ceiling.

    ``#1124``'s depth counter bounds path length (≤10 by construction), but it
    does NOT bound the node count: ``workflow_max_agent_calls`` defaults to 1000
    lifetime ``agent()`` children per run. So the walk carries an explicit
    config-tunable node-count ceiling; when it trips, ``at_nodes`` records how
    many nodes were emitted before the frontier was cut. The response stays
    well-formed (a partial-but-honest tree), never a silent truncation.
    """

    at_nodes: int = Field(description="Number of nodes emitted before the ceiling cut the walk.")


class TraceResponse(BaseModel):
    """A one-call flat trace for a run-root or session-root.

    ``root_kind`` / ``root_id`` name the tree's root. ``entries`` is the flat
    DFS-pre-order list (nodes interleaved with their journal frames).
    ``truncated`` is non-``None`` iff the node-count ceiling cut the walk.

    Ordering caveat (documented per #1149): cross-subtree time-ordering is
    best-effort to **transaction granularity** — ``created_at`` is
    ``transaction_timestamp()`` (constant for a whole run-step's journal; it can
    tie or invert across concurrent transactions on separate pooled
    connections), and the two journals share no global sequence. Only the causal
    parent→child edge is exact, so DFS pre-order is canonical; chronological is a
    client-side re-sort.

    Scope caveat: ``wake_session`` peer-pokes are out of scope (a peer stimulus,
    not a spawn — no ``request_opened`` edge); they do not appear as nodes.
    """

    root_kind: Literal["run", "session"]
    root_id: str
    entries: list[TraceEntry] = Field(default_factory=list)
    truncated: TraceTruncated | None = None


def node_summary_for_session(
    *,
    label: str | None,
    title: str | None,
    agent_id: str | None,
) -> str:
    """The session node's ``summary`` — ``label ‖ title ‖ agent_id ‖ 'generic agent'``.

    The useful label lives on the **parent** half of the ``agent()`` edge (the
    run-journal ``call_started.payload.label``), so the same join that de-dups
    the ``agent_call`` frame also supplies the summary. Falls back through the
    session's own ``title``, then its ``agent_id``, then a generic placeholder.
    """
    return label or title or agent_id or "generic agent"


def summarize_tool_call(payload: dict[str, Any]) -> str:
    """``tool_name`` + a truncated first line of the call input.

    Handles the spill-reference shape (a large input stored out-of-band as
    ``{"$spill": ...}``) by naming the reference rather than dumping it.
    """
    name = payload.get("tool_name") or payload.get("name") or "tool"
    raw_input = payload.get("input")
    if isinstance(raw_input, dict) and "$spill" in raw_input:
        return f"{name} (spilled input)"
    text = "" if raw_input is None else str(raw_input)
    first = text.splitlines()[0] if text else ""
    if len(first) > 80:
        first = first[:77] + "..."
    return f"{name}: {first}" if first else name
