"""Pure traversal/assembly of the trace service (#1149), DB mocked.

Pins the structural decisions independent of Postgres:
* ``build_entries`` emits **DFS pre-order** — a child's subtree is contiguous
  under its parent (so CLI indentation by ``depth`` is correct).
* the ``agent()`` / ``invoke_workflow`` ``call_started`` frame is **merged**
  into its spawned child node (never double-listed); an *orphan* call (child not
  in the walked set) is kept as a standalone ``agent_call`` entry.
* the abbreviated (``verbose=false``) filter drops chatter but keeps the
  request/response/turn lifecycle, gates, and any error frame; ``verbose=true``
  lifts the filter.
* ``_walk``'s node-count **ceiling** cuts the frontier and reports ``truncated``.
"""

from __future__ import annotations

import asyncio
from typing import Any

from aios.db.queries.trace import ChildNode
from aios.services import trace as svc


def _node(
    kind: str,
    id: str,
    *,
    parent: str | None = None,
    depth: int = 0,
    label: str | None = None,
    request_id: str | None = None,
) -> svc._Node:
    return svc._Node(
        kind=kind, id=id, parent_id=parent, depth=depth, label=label, request_id=request_id
    )


# ─── build_entries: DFS pre-order + the agent_call merge ─────────────────────


def test_build_entries_dfs_preorder_and_depth() -> None:
    # run(root) → session A → run B ; plus session C under root.
    nodes = [
        _node("run", "wfr_root", depth=0),
        _node("session", "sess_a", parent="wfr_root", depth=1, request_id="req_a"),
        _node("run", "wfr_b", parent="sess_a", depth=2, request_id="req_b"),
        _node("session", "sess_c", parent="wfr_root", depth=1, request_id="req_c"),
    ]
    entries = svc.build_entries(
        nodes,
        session_meta={"sess_a": {}, "sess_c": {}},
        run_meta={"wfr_root": {"status": "running"}, "wfr_b": {}},
        session_journals={},
        run_journals={},
        responses={
            "sess_a": {"is_error": False},
            "wfr_b": {"is_error": False},
            "sess_c": {"is_error": False},
        },
        verbose=False,
    )
    # Node order is exactly the DFS pre-order we passed.
    node_ids = [(e.kind, e.id) for e in entries if e.kind in ("run", "session")]
    assert node_ids == [
        ("run", "wfr_root"),
        ("session", "sess_a"),
        ("run", "wfr_b"),
        ("session", "sess_c"),
    ]
    depths = {e.id: e.depth for e in entries if e.kind in ("run", "session")}
    assert depths == {"wfr_root": 0, "sess_a": 1, "wfr_b": 2, "sess_c": 1}


def test_agent_call_frame_merged_into_child_not_double_listed() -> None:
    nodes = [
        _node("run", "wfr_root", depth=0),
        _node(
            "session", "sess_child", parent="wfr_root", depth=1, label="reviewer", request_id="req1"
        ),
    ]
    run_journals = {
        "wfr_root": [
            {
                "seq": 1,
                "type": "call_started",
                "created_at": None,
                "payload": {
                    "capability": "agent",
                    "child_session_id": "sess_child",
                    "label": "reviewer",
                },
            }
        ]
    }
    entries = svc.build_entries(
        nodes,
        session_meta={"sess_child": {"title": "Reviewer"}},
        run_meta={"wfr_root": {"status": "running"}},
        session_journals={},
        run_journals=run_journals,
        responses={"sess_child": {"is_error": False}},
        verbose=True,
    )
    # The call_started for the spawned child is NOT emitted as a frame.
    assert not any(e.kind == "agent_call" for e in entries)
    # The child node appears exactly once, with the parent's label as summary.
    child = [e for e in entries if e.id == "sess_child"]
    assert len(child) == 1
    assert child[0].summary == "reviewer"


def test_orphan_agent_call_is_kept() -> None:
    nodes = [_node("run", "wfr_root", depth=0)]
    run_journals = {
        "wfr_root": [
            {
                "seq": 7,
                "type": "call_started",
                "created_at": None,
                "payload": {
                    "capability": "agent",
                    "child_session_id": "sess_never",
                    "label": "ghost",
                },
            }
        ]
    }
    entries = svc.build_entries(
        nodes,
        session_meta={},
        run_meta={"wfr_root": {"status": "running"}},
        session_journals={},
        run_journals=run_journals,
        responses={},
        verbose=False,
    )
    orphans = [e for e in entries if e.kind == "agent_call"]
    assert len(orphans) == 1
    assert "ghost" in orphans[0].summary


def test_truncated_walk_does_not_mislabel_cut_child_as_orphan() -> None:
    # The walk truncated, so a child cut by the ceiling is absent from the node
    # set — but its parent's call_started frame remains. That child must NOT be
    # emitted as a never-spawned "orphan" (it WAS spawned; it was merely cut).
    nodes = [_node("run", "wfr_root", depth=0)]
    run_journals = {
        "wfr_root": [
            {
                "seq": 7,
                "type": "call_started",
                "created_at": None,
                "payload": {
                    "capability": "agent",
                    "child_session_id": "sess_cut",  # spawned, but cut by ceiling
                    "label": "cut-child",
                },
            }
        ]
    }
    entries = svc.build_entries(
        nodes,
        session_meta={},
        run_meta={"wfr_root": {"status": "running"}},
        session_journals={},
        run_journals=run_journals,
        responses={},
        verbose=False,
        truncated=True,
    )
    assert not any(e.kind == "agent_call" for e in entries)


# ─── abbreviated vs verbose filtering ────────────────────────────────────────


def _session_journal() -> list[dict[str, Any]]:
    return [
        {
            "seq": 1,
            "kind": "lifecycle",
            "created_at": None,
            "data": {"event": "request_opened", "request_id": "req1"},
        },
        {"seq": 2, "kind": "message", "created_at": None, "data": {"role": "assistant"}},
        {
            "seq": 3,
            "kind": "span",
            "created_at": None,
            "data": {"is_error": True, "error": {"kind": "tool_failed"}},
        },
    ]


def test_abbreviated_drops_chatter_keeps_error_and_lifecycle() -> None:
    nodes = [_node("session", "sess_a", depth=0)]
    entries = svc.build_entries(
        nodes,
        session_meta={"sess_a": {}},
        run_meta={},
        session_journals={"sess_a": _session_journal()},
        run_journals={},
        responses={},
        verbose=False,
    )
    frame_kinds = [e.kind for e in entries if e.kind not in ("session", "run")]
    # The message chatter frame is dropped; the request + error frames survive.
    assert "message" not in frame_kinds
    assert "request" in frame_kinds
    assert "error" in frame_kinds
    err = next(e for e in entries if e.kind == "error")
    assert err.error_kind == "tool_failed"


def test_verbose_keeps_everything() -> None:
    nodes = [_node("session", "sess_a", depth=0)]
    entries = svc.build_entries(
        nodes,
        session_meta={"sess_a": {}},
        run_meta={},
        session_journals={"sess_a": _session_journal()},
        run_journals={},
        responses={},
        verbose=True,
    )
    frame_kinds = [e.kind for e in entries if e.kind not in ("session", "run")]
    assert "message" in frame_kinds  # chatter retained under verbose


# ─── _walk: the node-count ceiling ───────────────────────────────────────────


class _FakeConn:
    """A conn whose ``children_of`` is driven by an in-memory adjacency map."""

    def __init__(self, adjacency: dict[tuple[str, str], list[ChildNode]]) -> None:
        self._adj = adjacency


async def _fake_children_of(
    conn: _FakeConn, *, caller_kind: str, caller_id: str, account_id: str
) -> list[ChildNode]:
    return conn._adj.get((caller_kind, caller_id), [])


def test_walk_ceiling_truncates(monkeypatch: Any) -> None:
    # A root run with 5 child sessions; ceiling=3 ⇒ emit root + 2, truncated.
    adjacency = {
        ("run", "wfr_root"): [
            ChildNode(kind="session", id=f"sess_{i}", label=None, request_id=f"r{i}")
            for i in range(5)
        ]
    }
    conn = _FakeConn(adjacency)
    monkeypatch.setattr("aios.services.trace.trace_q.children_of", _fake_children_of)

    nodes, truncated = asyncio.run(
        svc._walk(conn, root_kind="run", root_id="wfr_root", account_id="acct", ceiling=3)
    )
    assert truncated is True
    assert len(nodes) == 3  # root + 2 children before the ceiling cut


def test_walk_no_truncation_under_ceiling(monkeypatch: Any) -> None:
    adjacency = {
        ("run", "wfr_root"): [ChildNode(kind="session", id="sess_x", label=None, request_id="r")]
    }
    conn = _FakeConn(adjacency)
    monkeypatch.setattr("aios.services.trace.trace_q.children_of", _fake_children_of)
    nodes, truncated = asyncio.run(
        svc._walk(conn, root_kind="run", root_id="wfr_root", account_id="acct", ceiling=50)
    )
    assert truncated is False
    assert [(n.kind, n.id) for n in nodes] == [("run", "wfr_root"), ("session", "sess_x")]


def test_walk_is_cycle_safe(monkeypatch: Any) -> None:
    # An accidental A→B→A cycle must not loop forever (visited set).
    adjacency = {
        ("run", "wfr_a"): [ChildNode(kind="run", id="wfr_b", label=None, request_id="r1")],
        ("run", "wfr_b"): [ChildNode(kind="run", id="wfr_a", label=None, request_id="r2")],
    }
    conn = _FakeConn(adjacency)
    monkeypatch.setattr("aios.services.trace.trace_q.children_of", _fake_children_of)
    nodes, truncated = asyncio.run(
        svc._walk(conn, root_kind="run", root_id="wfr_a", account_id="acct", ceiling=50)
    )
    assert [(n.kind, n.id) for n in nodes] == [("run", "wfr_a"), ("run", "wfr_b")]
    assert truncated is False
