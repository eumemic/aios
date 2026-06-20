"""The trace service (#1149): walk → normalize → interleave.

A *trace* is a read-projection over the invocation-edge tree the invoke-edge
epic already records — no new subsystem, no trace table, no spans. This module:

1. **Walks** the parent→child edge tree from the root (``db.queries.trace`` does
   the reverse "children-of by caller" lookup, union of the trusted edge and the
   still-live FK columns).
2. **Normalizes** each node's outcome (``services.trace_normalizer``, reusing
   #1126's ``derive_response`` / ``derive_run_response``).
3. **Interleaves** the nodes' existing journals into a flat DFS-pre-order list.

The whole walk runs in **one ``REPEATABLE READ`` transaction on a single
connection**, so every read sees one MVCC snapshot — a live tree that mutates
mid-walk can't produce a torn trace. A **node-count ceiling** bounds ``|V|`` (the
depth cap bounds only path length); when it trips the response carries a typed
``truncated`` marker.

The pure traversal/assembly (``build_entries``) is split from the I/O so the
DB-mocked unit tier can pin its behavior (DFS order, the ceiling, the
``agent_call``→child merge) without a live Postgres.
"""

from __future__ import annotations

from typing import Any

import asyncpg

from aios.config import get_settings
from aios.db import queries
from aios.db.queries import trace as trace_q
from aios.db.queries import workflows as wf_queries
from aios.models.trace import (
    TraceEntry,
    TraceResponse,
    TraceTruncated,
    node_summary_for_session,
    summarize_tool_call,
)
from aios.services import trace_normalizer

# Abbreviated (``verbose=false``) keeps only the load-bearing frames per node:
# the request/response/turn lifecycle, gates, and any ``is_error`` frame (so the
# proximate cause sits one line from the terminal outcome). The full firehose is
# ``verbose=true`` (or the per-resource ``…/events`` endpoints).
_ABBREVIATED_LIFECYCLE_EVENTS = frozenset({"request_opened", "request_response", "turn_ended"})
# Run-journal frame types kept in the abbreviated view (the run journal is
# already sparse; gates + the request/run bookends + errors carry the signal).
_ABBREVIATED_RUN_TYPES = frozenset({"run_started", "run_completed", "annotation"})


class _Node:
    """A resolved node in the walk, with its merge-from-parent context."""

    __slots__ = ("children", "depth", "id", "kind", "label", "parent_id", "request_id")

    def __init__(
        self,
        *,
        kind: str,
        id: str,
        parent_id: str | None,
        depth: int,
        label: str | None,
        request_id: str | None,
    ) -> None:
        self.kind = kind
        self.id = id
        self.parent_id = parent_id
        self.depth = depth
        self.label = label
        self.request_id = request_id
        self.children: list[_Node] = []


async def get_trace(
    pool: asyncpg.Pool[Any],
    *,
    root_kind: str,
    root_id: str,
    account_id: str,
    verbose: bool = False,
    max_nodes: int | None = None,
) -> TraceResponse:
    """Build the flat trace for a run-root or session-root, in one snapshot.

    The caller (router) has already account-scoped the root via ``get_run`` /
    ``get_session`` (404 cross-tenant). Here we re-open the work under a single
    ``REPEATABLE READ`` connection so the walk is coherent.
    """
    ceiling = max_nodes if max_nodes is not None else get_settings().trace_max_nodes
    async with pool.acquire() as conn, conn.transaction(isolation="repeatable_read", readonly=True):
        nodes, truncated = await _walk(
            conn, root_kind=root_kind, root_id=root_id, account_id=account_id, ceiling=ceiling
        )
        # Batch the journal + metadata reads over the collected id set (no N+1).
        session_ids = [n.id for n in nodes if n.kind == "session"]
        run_ids = [n.id for n in nodes if n.kind == "run"]
        session_meta = await trace_q.read_session_meta_batched(
            conn, session_ids, account_id=account_id
        )
        run_meta = await trace_q.read_run_meta_batched(conn, run_ids, account_id=account_id)
        session_journals = await trace_q.read_session_journal_batched(
            conn, session_ids, account_id=account_id
        )
        run_journals = await trace_q.read_run_journal_batched(conn, run_ids, account_id=account_id)
        # Resolve each servicer node's caller's-eye response under the SAME
        # snapshot (reusing the #1126 resolvers).
        responses: dict[str, dict[str, Any] | None] = {}
        for n in nodes:
            if n.request_id is None:
                continue
            if n.kind == "session":
                responses[n.id] = await queries.derive_response(
                    conn, n.id, account_id=account_id, request_id=n.request_id
                )
            else:
                responses[n.id] = await wf_queries.derive_run_response(
                    conn, n.id, account_id=account_id
                )

    entries = build_entries(
        nodes,
        session_meta=session_meta,
        run_meta=run_meta,
        session_journals=session_journals,
        run_journals=run_journals,
        responses=responses,
        verbose=verbose,
        truncated=truncated,
    )
    return TraceResponse(
        root_kind=root_kind,
        root_id=root_id,
        entries=entries,
        truncated=TraceTruncated(at_nodes=len(nodes)) if truncated else None,
    )


async def _walk(
    conn: asyncpg.Connection[Any],
    *,
    root_kind: str,
    root_id: str,
    account_id: str,
    ceiling: int,
) -> tuple[list[_Node], bool]:
    """DFS the edge tree from the root, building the node list in pre-order.

    Returns ``(nodes, truncated)``. ``truncated`` is ``True`` iff the node
    ceiling cut the frontier (the typed ``truncated`` marker). A visited set
    keyed on ``(kind, id)`` makes the walk robust to any accidental cycle (the
    depth cap already makes the real tree acyclic).
    """
    root = _Node(kind=root_kind, id=root_id, parent_id=None, depth=0, label=None, request_id=None)
    nodes: list[_Node] = []
    visited: set[tuple[str, str]] = set()
    truncated = False

    # Explicit stack of nodes to expand, preserving DFS pre-order: we emit a node
    # when popped, then push its children in REVERSE so the first child is
    # expanded next.
    stack: list[_Node] = [root]
    while stack:
        node = stack.pop()
        key = (node.kind, node.id)
        if key in visited:
            continue
        if len(nodes) >= ceiling:
            truncated = True
            break
        visited.add(key)
        nodes.append(node)

        children = await trace_q.children_of(
            conn, caller_kind=node.kind, caller_id=node.id, account_id=account_id
        )
        # The api-edge caller id is the account_id; a session→session / run→child
        # edge stores the node id. ``children_of`` already keys on (kind,id), so
        # the only extra concern is that a node may also be reachable as an
        # ``api``-caller's child (root only). We expand children spawned by THIS
        # node's (kind,id).
        child_nodes: list[_Node] = [
            _Node(
                kind=c.kind,
                id=c.id,
                parent_id=node.id,
                depth=node.depth + 1,
                label=c.label,
                request_id=c.request_id,
            )
            for c in children
        ]
        node.children = child_nodes
        for child in reversed(child_nodes):
            stack.append(child)

    return nodes, truncated


def build_entries(
    nodes: list[_Node],
    *,
    session_meta: dict[str, dict[str, Any]],
    run_meta: dict[str, dict[str, Any]],
    session_journals: dict[str, list[dict[str, Any]]],
    run_journals: dict[str, list[dict[str, Any]]],
    responses: dict[str, dict[str, Any] | None],
    verbose: bool,
    truncated: bool = False,
) -> list[TraceEntry]:
    """Assemble the flat DFS-pre-order entry list — pure, no I/O.

    For each node (already in DFS pre-order) emit its node entry (normalized
    ``terminal_state`` / ``error_kind``) followed by its abbreviated-or-full
    journal frames. The ``agent_call`` / ``invoke_workflow`` run-journal frames
    are NOT emitted as frames — they are merged into their child node (the
    child's ``label`` already came from that frame), so a spawned child is never
    double-listed.

    ``truncated`` is the walk's ceiling flag. It MUST be threaded through:
    ``walked_ids`` is derived post-truncation, so a child cut by the ceiling is
    absent from it and would otherwise be mislabelled a never-spawned "orphan".
    When the walk truncated we suppress that orphan emission — an "orphan" must
    mean *provably never spawned*, not *merely cut from this view*.
    """
    entries: list[TraceEntry] = []
    # The set of child ids each run spawned via a journal frame — so the merge
    # can drop the frame and detect orphans (a call_started whose child id is
    # not in the walked node set).
    walked_ids = {n.id for n in nodes}

    for node in nodes:
        if node.kind == "session":
            entries.append(_session_node_entry(node, session_meta, responses))
            entries.extend(
                _session_frames(node, session_journals.get(node.id, []), verbose=verbose)
            )
        else:
            entries.append(_run_node_entry(node, run_meta, responses))
            entries.extend(
                _run_frames(
                    node,
                    run_journals.get(node.id, []),
                    walked_ids=walked_ids,
                    truncated=truncated,
                    verbose=verbose,
                )
            )
    return entries


def _session_node_entry(
    node: _Node,
    session_meta: dict[str, dict[str, Any]],
    responses: dict[str, dict[str, Any] | None],
) -> TraceEntry:
    meta = session_meta.get(node.id, {})
    if node.request_id is not None:
        # A servicer session — caller's-eye outcome.
        state, error_kind = trace_normalizer.normalize_response(responses.get(node.id))
    else:
        # A root session — derive from stop_reason.
        open_ids = meta.get("open_request_ids") or []
        state, error_kind = trace_normalizer.normalize_session_root(
            meta.get("stop_reason"),
            owes_open_request=bool(open_ids),
            owed_request_response=meta.get("owed_request_response"),
            is_archived=meta.get("archived_at") is not None,
        )
    summary = node_summary_for_session(
        label=node.label, title=meta.get("title"), agent_id=meta.get("agent_id")
    )
    return TraceEntry(
        timestamp=meta.get("created_at"),
        depth=node.depth,
        parent_id=node.parent_id,
        kind="session",
        id=node.id,
        summary=summary,
        terminal_state=state,
        error_kind=error_kind,
    )


def _run_node_entry(
    node: _Node,
    run_meta: dict[str, dict[str, Any]],
    responses: dict[str, dict[str, Any] | None],
) -> TraceEntry:
    meta = run_meta.get(node.id, {})
    if node.request_id is not None:
        state, error_kind = trace_normalizer.normalize_response(responses.get(node.id))
    else:
        completed = meta.get("run_completed") or {}
        state, error_kind = trace_normalizer.normalize_run_root(
            status=meta.get("status", "running"),
            run_completed_error=completed.get("error"),
            run_completed_is_error=bool(completed.get("is_error")),
        )
    summary = meta.get("workflow_id") or "workflow run"
    return TraceEntry(
        timestamp=meta.get("created_at"),
        depth=node.depth,
        parent_id=node.parent_id,
        kind="run",
        id=node.id,
        summary=node.label or summary,
        terminal_state=state,
        error_kind=error_kind,
    )


def _session_frames(
    node: _Node, journal: list[dict[str, Any]], *, verbose: bool
) -> list[TraceEntry]:
    out: list[TraceEntry] = []
    for ev in journal:
        kind = ev["kind"]
        data = ev["data"] or {}
        is_error = bool(data.get("is_error")) or kind == "interrupt"
        if not verbose and not _keep_session_frame(kind, data, is_error):
            continue
        entry_kind, summary = _classify_session_frame(kind, data)
        out.append(
            TraceEntry(
                timestamp=ev["created_at"],
                depth=node.depth + 1,
                parent_id=node.id,
                kind=entry_kind,
                id=f"{node.id}#{ev['seq']}",
                summary=summary,
                error_kind=(data.get("error") or {}).get("kind") if is_error else None,
            )
        )
    return out


def _keep_session_frame(kind: str, data: dict[str, Any], is_error: bool) -> bool:
    if is_error:
        return True
    return kind == "lifecycle" and data.get("event") in _ABBREVIATED_LIFECYCLE_EVENTS


def _classify_session_frame(kind: str, data: dict[str, Any]) -> tuple[str, str]:
    if kind == "lifecycle":
        event = data.get("event")
        if event == "request_opened":
            return "request", f"request {data.get('request_id', '')}".strip()
        if event == "request_response":
            err = (data.get("error") or {}).get("kind")
            verb = f"error: {err}" if data.get("is_error") else "ok"
            return "response", f"response {verb}"
        return "annotation", str(event or "lifecycle")
    if kind == "interrupt":
        return "error", "interrupt"
    if kind == "span" and (data.get("error") or data.get("is_error")):
        return "error", _frame_error_summary(data)
    if kind == "message":
        role = data.get("role", "message")
        return "message", str(role)
    if data.get("is_error"):
        return "error", _frame_error_summary(data)
    return "message", kind


def _frame_error_summary(data: dict[str, Any]) -> str:
    err = data.get("error") or {}
    kind = err.get("kind") if isinstance(err, dict) else None
    msg = err.get("message") if isinstance(err, dict) else None
    parts = [p for p in (kind, msg) if p]
    return " · ".join(str(p) for p in parts) or "error"


def _run_frames(
    node: _Node,
    journal: list[dict[str, Any]],
    *,
    walked_ids: set[str],
    truncated: bool = False,
    verbose: bool,
) -> list[TraceEntry]:
    out: list[TraceEntry] = []
    for ev in journal:
        etype = ev["type"]
        payload = ev["payload"] or {}
        # Merge: an agent()/invoke_workflow call_started whose child IS in the
        # walked node set is folded into that child node — drop the frame. A
        # call_started whose child was never spawned (orphan) is kept.
        if etype == "call_started":
            cap = payload.get("capability")
            child_id = payload.get("child_session_id") or payload.get("child_run_id")
            if cap in ("agent", "invoke_workflow"):
                if child_id in walked_ids:
                    continue  # merged into the child node
                # A child absent from ``walked_ids`` is an orphan ONLY when the
                # walk was complete. If the ceiling truncated the walk, the child
                # may simply have been cut from this view — emitting "orphan {cap}"
                # would falsely assert it was never spawned. Suppress it; the
                # response carries the ``truncated`` marker for the cut frontier.
                if truncated:
                    continue
                # Orphan: child never spawned / rejected pre-spawn.
                out.append(
                    TraceEntry(
                        timestamp=ev["created_at"],
                        depth=node.depth + 1,
                        parent_id=node.id,
                        kind="agent_call",
                        id=f"{node.id}#{ev['seq']}",
                        summary=f"orphan {cap}: {payload.get('label', '')}".strip(),
                    )
                )
                continue
        is_error = etype == "run_completed" and bool(payload.get("is_error"))
        if not verbose and not _keep_run_frame(etype, payload, is_error):
            continue
        entry_kind, summary = _classify_run_frame(etype, payload)
        out.append(
            TraceEntry(
                timestamp=ev["created_at"],
                depth=node.depth + 1,
                parent_id=node.id,
                kind=entry_kind,
                id=f"{node.id}#{ev['seq']}",
                summary=summary,
                error_kind=(payload.get("error") or {}).get("kind") if is_error else None,
            )
        )
    return out


def _keep_run_frame(etype: str, payload: dict[str, Any], is_error: bool) -> bool:
    if is_error:
        return True
    if etype in _ABBREVIATED_RUN_TYPES:
        return True
    # Gates are load-bearing for the suspended-on-gate story.
    return etype in ("call_started", "call_result") and payload.get("capability") == "gate"


def _classify_run_frame(etype: str, payload: dict[str, Any]) -> tuple[str, str]:
    cap = payload.get("capability")
    if etype == "call_started":
        if cap == "gate":
            return "gate", f"gate {payload.get('gate_nonce', '')}".strip()
        if cap == "tool":
            return "tool_call", summarize_tool_call(payload)
        return "tool_call", str(cap or "call")
    if etype == "call_result":
        if cap == "gate":
            return "gate", "gate resumed"
        return "tool_call", f"{cap or 'call'} result"
    if etype == "annotation":
        return "annotation", str(payload.get("text") or payload.get("kind") or "annotation")
    if etype == "run_completed":
        if payload.get("is_error"):
            err = (payload.get("error") or {}).get("kind")
            return "error", f"run errored: {err}"
        return "annotation", "run completed"
    if etype == "run_started":
        return "annotation", "run started"
    return "annotation", etype
