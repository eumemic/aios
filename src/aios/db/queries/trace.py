"""Read queries backing the trace walk (#1149).

Two responsibilities, both pure reads:

* **the reverse "children-of by caller" lookup** — for a node ``(kind, id)``,
  the set of child nodes it invoked. It is the **union of the trusted edge and
  the still-live structural FK columns** (``parent_run_id`` /
  ``launcher_session_id``), because in the current dual-write/dual-read era
  neither alone is complete. When a future contract migration retires the FK
  columns the FK half simply drops out and the edge covers everything.
* **batched journal reads** — once the node-id set is resolved, the per-node
  event streams are read with ``= ANY($1)`` set-membership (no N+1).

Every lookup re-applies ``account_id = $root_account`` as a SQL predicate (a
tenant invariant) — never derive the next account from edge JSONB. Sessions key
on ``events.account_id`` directly; runs key via a join to ``wf_runs``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import asyncpg

from aios.db.queries import open_request_anti_join

NodeKind = Literal["run", "session"]


@dataclass(frozen=True)
class ChildNode:
    """A child node discovered by the reverse walk.

    ``spawn_at`` orders siblings under their parent (the child's spawn
    ``created_at``); ``label`` is the parent-half ``call_started.label`` for an
    ``agent()`` / ``invoke_workflow`` spawn (``None`` for FK-only / api / session
    callers). ``request_id`` is the edge's request id when the child was invoked
    *in service of a request* (``None`` for a root-style FK child). ``awaited`` is
    the edge's ``Ask|Tell`` bit (Ask ⇒ owned/dies-with-parent; Tell ⇒ detached) —
    defaults ``True`` for legacy/FK rows that carry no bit (the conservative cut).
    """

    kind: NodeKind
    id: str
    spawn_at: Any | None = None
    label: str | None = None
    request_id: str | None = None
    awaited: bool = True


async def children_of(
    conn: asyncpg.Connection[Any],
    *,
    caller_kind: str,
    caller_id: str,
    account_id: str,
) -> list[ChildNode]:
    """The child nodes ``(caller_kind, caller_id)`` invoked — edge union live FK.

    Keyed on ``(caller.kind, caller.id)`` — **never ``id`` alone** — because
    ``api`` edges store ``caller.id = account_id`` (so an id-only key would fan
    a whole tenant's sessions under one api node). Dedups by ``(kind, id)`` so a
    node visible via *both* the edge and its FK column is returned once (the edge
    half is preferred — it carries the ``label`` + ``request_id``).
    """
    children: dict[tuple[str, str], ChildNode] = {}

    # ── trusted-edge half (preferred) ────────────────────────────────────────
    # run/session → child session: a ``request_opened`` whose caller matches.
    sess_rows = await conn.fetch(
        "SELECT e.session_id AS id, e.created_at AS spawn_at, "
        "       e.data->>'request_id' AS request_id, "
        "       COALESCE((e.data->>'awaited')::bool, true) AS awaited "
        "FROM events e "
        "WHERE e.account_id = $1 AND e.kind = 'lifecycle' "
        "AND e.data->>'event' = 'request_opened' "
        "AND e.data->'caller'->>'kind' = $2 AND e.data->'caller'->>'id' = $3",
        account_id,
        caller_kind,
        caller_id,
    )
    for r in sess_rows:
        key = ("session", r["id"])
        children.setdefault(
            key,
            ChildNode(
                kind="session",
                id=r["id"],
                spawn_at=r["spawn_at"],
                request_id=r["request_id"],
                awaited=r["awaited"],
            ),
        )

    # run → sub-run: a ``wf_runs.caller`` matching (#1129). The spawn order is
    # the sub-run's own ``created_at``.
    subrun_rows = await conn.fetch(
        "SELECT r.id AS id, r.created_at AS spawn_at, r.request_id AS request_id, "
        "       COALESCE((r.caller->>'awaited')::bool, true) AS awaited "
        "FROM wf_runs r "
        "WHERE r.account_id = $1 "
        "AND r.caller->>'kind' = $2 AND r.caller->>'id' = $3",
        account_id,
        caller_kind,
        caller_id,
    )
    for r in subrun_rows:
        key = ("run", r["id"])
        children.setdefault(
            key,
            ChildNode(
                kind="run",
                id=r["id"],
                spawn_at=r["spawn_at"],
                request_id=r["request_id"],
                awaited=r["awaited"],
            ),
        )

    # ── live FK half (until the FK columns are retired) ──────────────────────
    # These cover pre-#1123/#1129 subtrees and the *detached* session→run launches
    # (the ``create_run`` tool, the trigger ``workflow`` action) that write no
    # caller edge — the awaited ``call_workflow``/``invoke_workflow`` path now does
    # (above). They only ADD nodes the edge missed — ``setdefault`` keeps the
    # richer edge row (with ``request_id``/``awaited``) when both are present.
    if caller_kind == "run":
        fk_sessions = await conn.fetch(
            "SELECT s.id AS id, s.created_at AS spawn_at FROM sessions s "
            "WHERE s.account_id = $1 AND s.parent_run_id = $2",
            account_id,
            caller_id,
        )
        for r in fk_sessions:
            children.setdefault(
                ("session", r["id"]),
                ChildNode(kind="session", id=r["id"], spawn_at=r["spawn_at"]),
            )
        fk_subruns = await conn.fetch(
            "SELECT r.id AS id, r.created_at AS spawn_at FROM wf_runs r "
            "WHERE r.account_id = $1 AND r.parent_run_id = $2",
            account_id,
            caller_id,
        )
        for r in fk_subruns:
            children.setdefault(
                ("run", r["id"]),
                ChildNode(kind="run", id=r["id"], spawn_at=r["spawn_at"]),
            )
    elif caller_kind == "session":
        # session → run: the awaited ``call_workflow`` launch writes the caller
        # edge (above); the launcher FK still covers detached launches (the
        # ``create_run`` tool, the trigger ``workflow`` action) that write none.
        fk_launched = await conn.fetch(
            "SELECT r.id AS id, r.created_at AS spawn_at FROM wf_runs r "
            "WHERE r.account_id = $1 AND r.launcher_session_id = $2",
            account_id,
            caller_id,
        )
        for r in fk_launched:
            children.setdefault(
                ("run", r["id"]),
                ChildNode(kind="run", id=r["id"], spawn_at=r["spawn_at"]),
            )

    # Deterministic sibling order: spawn time, then id (a stable tiebreak when
    # two siblings share a transaction_timestamp). ``None`` spawn_at sorts last.
    return sorted(
        children.values(),
        key=lambda c: (c.spawn_at is None, c.spawn_at, c.id),
    )


async def find_parked_servicer(
    conn: asyncpg.Connection[Any],
    *,
    caller_session_id: str,
    tool_call_id: str,
    account_id: str,
) -> tuple[NodeKind, str, str | None, dict[str, Any] | None] | None:
    """Resolve the servicer a session's parked ``call_*`` task was awaiting (#1431).

    The durable counterpart of the in-memory park handle: a ``call_*`` handler stamps its
    own ``tool_call_id`` onto the servicer's edge ``caller`` (``request_opened.data.caller``
    for a session servicer, ``wf_runs.caller`` for a run), so crash-recovery can re-derive
    everything ``_park_and_resolve`` needs — ``(servicer_kind, servicer_id, request_id,
    output_schema)`` — from the edge alone. This is the same trusted "children-of by caller"
    direction :func:`children_of` walks, narrowed to one ``tool_call_id`` and projecting the
    per-request ``output_schema`` so the resume re-validates the answer exactly as the live
    handler would. Returns ``None`` when no such edge exists (the launch crashed before the
    servicer/edge was durable → recovery errors the call as retryable rather than re-parking
    on nothing). ``request_id`` is ``None`` for a run servicer (it parks on its terminal row,
    not a request edge).

    Both lookups key on the 0103 reverse caller indexes
    (``events_request_opened_caller_idx`` / ``wf_runs_caller_idx``: ``(account_id,
    caller.kind, caller.id)``) and add the ``tool_call_id`` filter on top; the caller is
    always ``kind='session'`` here (only the model-facing ``call_*`` handlers stamp a
    ``tool_call_id``). A session servicer is preferred when — impossibly — both match.
    """
    sess = await conn.fetchrow(
        "SELECT e.session_id AS id, e.data->>'request_id' AS request_id, "
        "e.data->'output_schema' AS output_schema FROM events e "
        "WHERE e.account_id = $1 AND e.kind = 'lifecycle' "
        "AND e.data->>'event' = 'request_opened' "
        "AND e.data->'caller'->>'kind' = 'session' AND e.data->'caller'->>'id' = $2 "
        "AND e.data->'caller'->>'tool_call_id' = $3 LIMIT 1",
        account_id,
        caller_session_id,
        tool_call_id,
    )
    if sess is not None:
        schema = sess["output_schema"]
        return ("session", sess["id"], sess["request_id"], schema if schema else None)

    run = await conn.fetchrow(
        "SELECT r.id AS id, r.request_output_schema AS output_schema FROM wf_runs r "
        "WHERE r.account_id = $1 "
        "AND r.caller->>'kind' = 'session' AND r.caller->>'id' = $2 "
        "AND r.caller->>'tool_call_id' = $3 LIMIT 1",
        account_id,
        caller_session_id,
        tool_call_id,
    )
    if run is not None:
        schema = run["output_schema"]
        return ("run", run["id"], None, schema if schema else None)

    return None


@dataclass(frozen=True)
class CallerTask:
    """One ``call_*`` task a session launched, located by its caller edge (#1428).

    The set-returning sibling of :func:`find_parked_servicer`: a servicer edge whose ``caller``
    names this session and carries a ``tool_call_id`` (so it was a model-launched ``call_*``
    park). ``request_id`` is the session-servicer's request id (``None`` for a run, which parks
    on its terminal row). Liveness — open vs answered — is NOT decided here; the service layer
    resolves it under one MVCC snapshot via the #1126 derive resolvers. ``opened_at`` is the
    edge's ``created_at``.
    """

    tool_call_id: str
    servicer_kind: NodeKind
    servicer_id: str
    request_id: str | None
    opened_at: Any


async def list_caller_tasks(
    conn: asyncpg.Connection[Any],
    *,
    caller_session_id: str,
    account_id: str,
) -> list[CallerTask]:
    """Every ``call_*`` task a session launched — its outbound roster (#1428).

    The set-returning sibling of :func:`find_parked_servicer`, on the same 0103 reverse-caller
    indexes (``events_request_opened_caller_idx`` / ``wf_runs_caller_idx``: ``(account_id,
    caller.kind, caller.id)``). One row per servicer edge whose ``caller`` is
    ``{kind:'session', id:<this session>}`` AND carries a ``tool_call_id`` — both servicer
    kinds (events ``request_opened`` + ``wf_runs.caller``).

    The ``tool_call_id`` filter is the SAME discriminant :func:`find_parked_servicer` keys on,
    so the point and set locators agree by construction. It also subsumes an ``awaited`` filter:
    only the awaited ``call_*`` parks stamp a ``tool_call_id`` (the detached
    ``create_run``/``Tell`` launches write none), so an unawaited edge can never appear here.
    Edge-only by construction — never the broader ``children_of`` FK union, which would surface
    detached launches that carry no park to list or stop.

    This is the pure locator: it returns BOTH open and already-answered edges (events are
    append-only). The caller (``services.tasks.list_open_tasks``) resolves liveness
    under one snapshot and keeps only the open ones. Ordered oldest-first (``opened_at`` then
    ``tool_call_id`` as a stable tiebreak when two share a transaction timestamp).
    """
    sess_rows = await conn.fetch(
        "SELECT e.data->'caller'->>'tool_call_id' AS tool_call_id, "
        "e.session_id AS servicer_id, e.data->>'request_id' AS request_id, "
        "e.created_at AS opened_at FROM events e "
        "WHERE e.account_id = $1 AND e.kind = 'lifecycle' "
        "AND e.data->>'event' = 'request_opened' "
        "AND e.data->'caller'->>'kind' = 'session' AND e.data->'caller'->>'id' = $2 "
        "AND e.data->'caller'->>'tool_call_id' IS NOT NULL",
        account_id,
        caller_session_id,
    )
    out: list[CallerTask] = [
        CallerTask(
            tool_call_id=r["tool_call_id"],
            servicer_kind="session",
            servicer_id=r["servicer_id"],
            request_id=r["request_id"],
            opened_at=r["opened_at"],
        )
        for r in sess_rows
    ]

    run_rows = await conn.fetch(
        "SELECT r.caller->>'tool_call_id' AS tool_call_id, r.id AS servicer_id, "
        "r.created_at AS opened_at FROM wf_runs r "
        "WHERE r.account_id = $1 "
        "AND r.caller->>'kind' = 'session' AND r.caller->>'id' = $2 "
        "AND r.caller->>'tool_call_id' IS NOT NULL",
        account_id,
        caller_session_id,
    )
    out.extend(
        CallerTask(
            tool_call_id=r["tool_call_id"],
            servicer_kind="run",
            servicer_id=r["servicer_id"],
            request_id=None,
            opened_at=r["opened_at"],
        )
        for r in run_rows
    )
    out.sort(key=lambda c: (c.opened_at, c.tool_call_id))
    return out


async def read_session_meta_batched(
    conn: asyncpg.Connection[Any],
    session_ids: list[str],
    *,
    account_id: str,
) -> dict[str, dict[str, Any]]:
    """Per-session metadata the normalizer needs, keyed by id (no N+1).

    ``{stop_reason, archived_at, title, agent_id, created_at}`` plus the derived
    open-request set (``open_request_ids``) so a root session's ``end_turn`` /
    archived branch can tell "owes a request" from "done." Account-scoped.

    For an **archived** session that owes an open request we also resolve that
    request's terminal outcome (``owed_request_response``) under the same snapshot,
    following :func:`aios.db.queries.sessions.derive_response` semantics: a written
    ``request_response`` wins, else — because an archived session can never answer —
    the request resolves to ``child_gone``. The normalizer's archived branch needs
    this to flag a green-washed failed root (an archived session owing an errored
    request) as ``errored`` instead of silently ``ok``. ``None`` for a live session
    or one that owes nothing.
    """
    if not session_ids:
        return {}
    rows = await conn.fetch(
        "SELECT s.id, s.stop_reason, s.archived_at, s.title, s.agent_id, s.created_at, "
        # The "owes a request" liveness for archived-root green-washing must
        # match get_open_request_ids: it owes a *response*, which an unawaited
        # ``Tell(NewSession)`` edge does not — so this composes the shared
        # open_request_anti_join with awaited_only=True (genuinely mirroring
        # get_open_request_ids via the one fragment).
        "  (SELECT array_agg(req.data->>'request_id') FROM events req WHERE "
        + open_request_anti_join(sid="s.id", acct="$2", awaited_only=True)
        + ") AS open_request_ids, "
        # The oldest open request's written response, if any (deterministic
        # oldest-first, mirroring get_open_request_ids). For an archived session
        # this is NULL (the request is open precisely because nothing answered it);
        # the derive_response child_gone fallback is applied in Python below.
        "  (SELECT resp.data FROM events req "
        "   JOIN events resp ON resp.session_id = req.session_id "
        "       AND resp.account_id = req.account_id "
        "       AND resp.kind = 'lifecycle' AND resp.data->>'event' = 'request_response' "
        "       AND resp.data->>'request_id' = req.data->>'request_id' "
        "   WHERE req.session_id = s.id AND req.account_id = $2 "
        "   AND req.kind = 'lifecycle' AND req.data->>'event' = 'request_opened' "
        "   AND req.data->>'request_id' IS NOT NULL "
        "   ORDER BY req.seq ASC LIMIT 1) AS owed_request_response "
        "FROM sessions s WHERE s.id = ANY($1) AND s.account_id = $2",
        session_ids,
        account_id,
    )
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        open_ids = list(r["open_request_ids"] or [])
        archived = r["archived_at"] is not None
        written = r["owed_request_response"] if r["owed_request_response"] is not None else None
        out[r["id"]] = {
            "stop_reason": r["stop_reason"] if r["stop_reason"] is not None else None,
            "archived_at": r["archived_at"],
            "title": r["title"],
            "agent_id": r["agent_id"],
            "created_at": r["created_at"],
            "open_request_ids": open_ids,
            "owed_request_response": _resolve_owed_response(
                written, open_ids=open_ids, archived=archived
            ),
        }
    return out


def _resolve_owed_response(
    written: dict[str, Any] | None,
    *,
    open_ids: list[str],
    archived: bool,
) -> dict[str, Any] | None:
    """Resolve an archived session's owed-request outcome — derive_response in Python.

    Mirrors :func:`aios.db.queries.sessions.derive_response`'s terminal rule for the
    owed request, restricted to what the trace's archived-root branch consumes:

    * a **written** ``request_response`` → that outcome (normalized to
      ``{result, is_error, error}``);
    * else, if the session is **archived** and still owes an open request (so it can
      never answer) → a Failed ``child_gone`` outcome;
    * else → ``None`` (nothing owed, or a live session — handled elsewhere).
    """
    if written is not None:
        return {
            "result": written.get("result"),
            "is_error": bool(written.get("is_error")),
            "error": written.get("error"),
        }
    if archived and open_ids:
        return {"result": None, "is_error": True, "error": {"kind": "child_gone"}}
    return None


async def read_run_meta_batched(
    conn: asyncpg.Connection[Any],
    run_ids: list[str],
    *,
    account_id: str,
) -> dict[str, dict[str, Any]]:
    """Per-run metadata the normalizer needs, keyed by id (no N+1).

    ``{status, workflow_id, caller, request_id, created_at}`` plus the
    ``run_completed`` payload (``{is_error, error}``) so a root run's terminal
    state and ``error_kind`` resolve from the journal without a per-node fetch.
    Account-scoped.
    """
    if not run_ids:
        return {}
    rows = await conn.fetch(
        "SELECT r.id, r.status, r.workflow_id, r.caller, r.request_id, r.created_at, "
        "  (SELECT e.payload FROM wf_run_events e "
        "   WHERE e.run_id = r.id AND e.type = 'run_completed' LIMIT 1) AS completed "
        "FROM wf_runs r WHERE r.id = ANY($1) AND r.account_id = $2",
        run_ids,
        account_id,
    )
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        completed = r["completed"] if r["completed"] is not None else None
        out[r["id"]] = {
            "status": r["status"],
            "workflow_id": r["workflow_id"],
            "caller": r["caller"] if r["caller"] is not None else None,
            "request_id": r["request_id"],
            "created_at": r["created_at"],
            "run_completed": completed,
        }
    return out


async def read_run_journal_batched(
    conn: asyncpg.Connection[Any],
    run_ids: list[str],
    *,
    account_id: str,
) -> dict[str, list[dict[str, Any]]]:
    """The journal frames for a SET of runs, keyed by ``run_id`` (no N+1).

    One ``run_id = ANY($1)`` read (account-scoped via the ``wf_runs`` join),
    grouped in Python. Each frame is ``{seq, type, payload, created_at}``.
    """
    if not run_ids:
        return {}
    rows = await conn.fetch(
        "SELECT e.run_id, e.seq, e.type, e.payload, e.created_at "
        "FROM wf_run_events e JOIN wf_runs r ON r.id = e.run_id "
        "WHERE e.run_id = ANY($1) AND r.account_id = $2 "
        "ORDER BY e.run_id, e.seq ASC",
        run_ids,
        account_id,
    )
    out: dict[str, list[dict[str, Any]]] = {rid: [] for rid in run_ids}
    for r in rows:
        out[r["run_id"]].append(
            {
                "seq": r["seq"],
                "type": r["type"],
                "payload": r["payload"],
                "created_at": r["created_at"],
            }
        )
    return out


async def read_session_journal_batched(
    conn: asyncpg.Connection[Any],
    session_ids: list[str],
    *,
    account_id: str,
) -> dict[str, list[dict[str, Any]]]:
    """The journal frames for a SET of sessions, keyed by ``session_id`` (no N+1).

    One ``session_id = ANY($1)`` read (account-scoped on ``events.account_id``),
    grouped in Python. Each frame is ``{seq, kind, data, created_at}``.
    """
    if not session_ids:
        return {}
    rows = await conn.fetch(
        "SELECT e.session_id, e.seq, e.kind, e.data, e.created_at "
        "FROM events e "
        "WHERE e.session_id = ANY($1) AND e.account_id = $2 "
        "ORDER BY e.session_id, e.seq ASC",
        session_ids,
        account_id,
    )
    out: dict[str, list[dict[str, Any]]] = {sid: [] for sid in session_ids}
    for r in rows:
        out[r["session_id"]].append(
            {
                "seq": r["seq"],
                "kind": r["kind"],
                "data": r["data"],
                "created_at": r["created_at"],
            }
        )
    return out
