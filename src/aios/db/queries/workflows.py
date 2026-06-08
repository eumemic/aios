"""Database queries for the workflows durable runtime (Block 1).

The workflows subsystem module of the ``aios.db.queries`` package. (The legacy
monolith lives in ``queries/__init__.py`` for now; subsystems are being carved
out into their own modules incrementally — workflows is the first.) Raw SQL
against asyncpg, same conventions as the rest of the package.

The load-bearing function is :func:`append_run_event` — the **single writer** to
``wf_run_events`` (always called under the procrastinate ``lock=run_id`` of a
``run_workflow_step``). It allocates a **gapless** seq and is idempotent on
``(run_id, call_key, type)``: it inserts *first* (computing ``seq =
last_event_seq + 1`` in the same statement) and bumps ``last_event_seq`` **only
when a row is actually inserted** — so an idempotent conflict (the memo dedup) or
a terminal/archived run consumes no seq and leaves no gap.
"""

from __future__ import annotations

import json
from typing import Any

import asyncpg

from aios.db.queries import _get_scoped, _list_scoped, parse_jsonb
from aios.errors import ConflictError, NotFoundError
from aios.ids import WORKFLOW, WORKFLOW_EVENT, WORKFLOW_RUN, make_id
from aios.models.workflows import (
    WfRun,
    WfRunEvent,
    WfRunEventType,
    WfRunSignal,
    WfRunSignalKind,
    Workflow,
)


def _row_to_workflow(row: asyncpg.Record) -> Workflow:
    return Workflow(
        id=row["id"],
        account_id=row["account_id"],
        name=row["name"],
        version=row["version"],
        script=row["script"],
        input_schema=parse_jsonb(row["input_schema"]),
        output_schema=parse_jsonb(row["output_schema"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_wf_run(row: asyncpg.Record) -> WfRun:
    return WfRun(
        id=row["id"],
        workflow_id=row["workflow_id"],
        account_id=row["account_id"],
        environment_id=row["environment_id"],
        parent_run_id=row["parent_run_id"],
        script=row["script"],
        script_sha=row["script_sha"],
        status=row["status"],
        input=parse_jsonb(row["input"]),
        output=parse_jsonb(row["output"]),
        last_event_seq=row["last_event_seq"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        archived_at=row["archived_at"],
    )


def _row_to_wf_run_event(row: asyncpg.Record) -> WfRunEvent:
    return WfRunEvent(
        id=row["id"],
        run_id=row["run_id"],
        seq=row["seq"],
        type=row["type"],
        call_key=row["call_key"],
        payload=parse_jsonb(row["payload"]),
        created_at=row["created_at"],
    )


def _row_to_wf_run_signal(row: asyncpg.Record) -> WfRunSignal:
    return WfRunSignal(
        run_id=row["run_id"],
        call_key=row["call_key"],
        kind=row["kind"],
        result=parse_jsonb(row["result"]),
        delivered_at=row["delivered_at"],
    )


# ─── workflows (definitions) ─────────────────────────────────────────────────


async def insert_workflow(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    name: str,
    script: str,
    version: int = 1,
    input_schema: dict[str, Any] | None = None,
    output_schema: dict[str, Any] | None = None,
) -> Workflow:
    """Insert an immutable workflow definition. Raises ``ConflictError`` on a
    duplicate ``(account_id, name, version)``."""
    new_id = make_id(WORKFLOW)
    try:
        row = await conn.fetchrow(
            """
            INSERT INTO workflows
                (id, account_id, name, version, script, input_schema, output_schema)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb)
            RETURNING *
            """,
            new_id,
            account_id,
            name,
            version,
            script,
            json.dumps(input_schema) if input_schema is not None else None,
            json.dumps(output_schema) if output_schema is not None else None,
        )
    except asyncpg.UniqueViolationError as exc:
        raise ConflictError(
            f"workflow {name!r} v{version} already exists",
            detail={"name": name, "version": version},
        ) from exc
    assert row is not None
    return _row_to_workflow(row)


async def get_workflow(
    conn: asyncpg.Connection[Any], workflow_id: str, *, account_id: str
) -> Workflow:
    return await _get_scoped(
        conn,
        table="workflows",
        id_=workflow_id,
        account_id=account_id,
        row=_row_to_workflow,
        noun="workflow",
    )


async def list_workflows(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    limit: int = 50,
    after: str | None = None,
    name: str | None = None,
) -> list[Workflow]:
    """Keyset-paginated list of an account's workflows, newest first.

    Hand-written (not ``_list_scoped``) because the ``workflows`` table has no
    ``archived_at`` column — workflow definitions are immutable and never archived.
    Each returned row is one ``(name, version)``; versioning/update is deferred, so
    today there is one row per name.
    """
    args: list[Any] = [account_id]
    where = ["account_id = $1"]
    if name is not None:
        args.append(name)
        where.append(f"name = ${len(args)}")
    if after is not None:
        args.append(after)
        where.append(f"id < ${len(args)}")
    args.append(limit)
    sql = f"SELECT * FROM workflows WHERE {' AND '.join(where)} ORDER BY id DESC LIMIT ${len(args)}"
    return [_row_to_workflow(r) for r in await conn.fetch(sql, *args)]


# ─── wf_runs (execution instances) ───────────────────────────────────────────


async def get_wf_run(conn: asyncpg.Connection[Any], run_id: str, *, account_id: str) -> WfRun:
    """Account-scoped run read (the public surface). Raises ``NotFoundError`` on a
    miss — including a cross-tenant id, so it never leaks another account's run.
    (The unscoped :func:`get_run_for_step` is the internal/trusted variant.)"""
    return await _get_scoped(
        conn,
        table="wf_runs",
        id_=run_id,
        account_id=account_id,
        row=_row_to_wf_run,
        noun="workflow run",
    )


async def list_wf_runs(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    limit: int = 50,
    after: str | None = None,
    workflow_id: str | None = None,
    status: str | None = None,
) -> list[WfRun]:
    """Keyset-paginated list of an account's runs (non-archived), newest first."""
    return await _list_scoped(
        conn,
        table="wf_runs",
        account_id=account_id,
        row=_row_to_wf_run,
        limit=limit,
        after=after,
        filters=[("workflow_id", workflow_id), ("status", status)],
    )


async def insert_wf_run(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    workflow_id: str,
    environment_id: str,
    script: str,
    script_sha: str,
    input: Any = None,
    parent_run_id: str | None = None,
) -> WfRun:
    """Insert a fresh ``pending`` run that snapshots ``script`` (+ ``script_sha``)."""
    new_id = make_id(WORKFLOW_RUN)
    try:
        row = await conn.fetchrow(
            """
            INSERT INTO wf_runs
                (id, workflow_id, account_id, environment_id, parent_run_id,
                 script, script_sha, status, input)
            VALUES ($1, $2, $3, $4, $5, $6, $7, 'pending', $8::jsonb)
            RETURNING *
            """,
            new_id,
            workflow_id,
            account_id,
            environment_id,
            parent_run_id,
            script,
            script_sha,
            json.dumps(input) if input is not None else None,
        )
    except asyncpg.ForeignKeyViolationError as exc:
        raise NotFoundError(
            f"workflow {workflow_id} or environment {environment_id} not found",
            detail={"workflow_id": workflow_id, "environment_id": environment_id},
        ) from exc
    assert row is not None
    return _row_to_wf_run(row)


async def get_run_for_step(conn: asyncpg.Connection[Any], run_id: str) -> WfRun | None:
    """Load a run by id (no account scoping — the step is internal/trusted).

    Returns ``None`` if the run no longer exists, so a stray wake is a no-op.
    """
    row = await conn.fetchrow("SELECT * FROM wf_runs WHERE id = $1", run_id)
    return _row_to_wf_run(row) if row is not None else None


async def set_run_status(
    conn: asyncpg.Connection[Any], run_id: str, status: str, *, account_id: str
) -> None:
    await conn.execute(
        "UPDATE wf_runs SET status = $3, updated_at = now() WHERE id = $1 AND account_id = $2",
        run_id,
        account_id,
        status,
    )


async def set_run_terminal(
    conn: asyncpg.Connection[Any],
    run_id: str,
    *,
    status: str,
    output: Any,
    account_id: str,
) -> None:
    """Flip a run to its terminal ``status`` and store ``output`` in one UPDATE.

    Called from ``_complete_run`` inside the same txn as the ``run_completed``
    append. ``output`` is the script's return value on success and ``None`` on
    error (the error detail lives in the ``run_completed`` payload).
    """
    await conn.execute(
        "UPDATE wf_runs SET status = $3, output = $4::jsonb, updated_at = now() "
        "WHERE id = $1 AND account_id = $2",
        run_id,
        account_id,
        status,
        json.dumps(output) if output is not None else None,
    )


async def list_active_run_ids(conn: asyncpg.Connection[Any]) -> list[str]:
    """``id`` for every non-terminal, live run — the sweep predicate. (No
    ``account_id``: ``defer_run_wake`` needs none and appends no journal span.)"""
    rows = await conn.fetch(
        "SELECT id FROM wf_runs "
        "WHERE archived_at IS NULL AND status IN ('pending','running','suspended')"
    )
    return [r["id"] for r in rows]


# ─── wf_run_events (the journal — single writer, gapless, idempotent) ─────────


async def append_run_event(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    run_id: str,
    type: WfRunEventType,
    payload: dict[str, Any],
    call_key: str | None = None,
) -> WfRunEvent | None:
    """Append a journal event with a gapless seq.

    Returns the inserted event, or ``None`` when nothing was inserted — either
    the run is terminal/archived (the WHERE filters it) or an idempotent conflict
    on the memo key ``(run_id, call_key, type)``. In both cases no seq is
    consumed, so the sequence stays gapless. The insert + the ``last_event_seq``
    bump are atomic (own savepoint when the caller is already in a transaction —
    e.g. the ``run_completed`` + ``set_run_terminal`` unit).

    The memo (``UNIQUE NULLS NOT DISTINCT (run_id, call_key, type)``) dedups
    *every* event type uniformly — including the ``run_started``/``run_completed``
    bookends, whose ``call_key IS NULL`` (``NULLS NOT DISTINCT`` makes those NULLs
    collide instead of counting as distinct). So a replayed append is idempotent
    by construction; the status guard above is the orthogonal "no writes to a
    terminal/archived run" filter.
    """
    new_id = make_id(WORKFLOW_EVENT)
    async with conn.transaction():
        row = await conn.fetchrow(
            """
            INSERT INTO wf_run_events (id, run_id, seq, type, call_key, payload)
            SELECT $1, r.id, r.last_event_seq + 1, $4, $5, $6::jsonb
            FROM wf_runs r
            WHERE r.id = $2 AND r.account_id = $3 AND r.archived_at IS NULL
              AND r.status NOT IN ('completed', 'errored')
            ON CONFLICT (run_id, call_key, type) DO NOTHING
            RETURNING *
            """,
            new_id,
            run_id,
            account_id,
            type,
            call_key,
            json.dumps(payload),
        )
        if row is None:
            return None
        await conn.execute(
            "UPDATE wf_runs SET last_event_seq = $2, updated_at = now() WHERE id = $1",
            run_id,
            row["seq"],
        )
    # NOTIFY *after* the insert/bump txn (never inside it — a subscriber must not
    # see a payload for an uncommitted row). pg_notify is transactional, so when an
    # outer txn wraps this call (``_complete_run``'s run_completed + set_terminal
    # unit), delivery is deferred to that commit; in the common autocommit path it
    # fires immediately on the already-committed row. ``pg_notify`` function form,
    # not literal NOTIFY: run-event ids are mixed-case prefixed-ULIDs (Postgres
    # case-folds unquoted identifiers). Mirrors session ``append_event``. The
    # ``/runs/{id}/stream`` backfill + seq-dedup recovers any missed tick.
    await conn.execute("SELECT pg_notify($1, $2)", f"wf_run_events_{run_id}", row["id"])
    return _row_to_wf_run_event(row)


async def list_run_events(conn: asyncpg.Connection[Any], run_id: str) -> list[WfRunEvent]:
    """Unscoped, unpaginated journal read — the internal harvest path
    (``run_workflow_step``). For the public surface use :func:`list_run_events_scoped`."""
    rows = await conn.fetch("SELECT * FROM wf_run_events WHERE run_id = $1 ORDER BY seq", run_id)
    return [_row_to_wf_run_event(r) for r in rows]


async def list_run_events_scoped(
    conn: asyncpg.Connection[Any],
    run_id: str,
    *,
    account_id: str,
    after_seq: int = 0,
    limit: int = 200,
) -> list[WfRunEvent]:
    """Account-scoped, forward-only (``seq > after_seq``) journal page.

    ``wf_run_events`` has no ``account_id`` column, so scope via a join to
    ``wf_runs`` — defense in depth even though the router pre-checks the run with
    :func:`get_wf_run`. Ascending seq, like the session events read path."""
    rows = await conn.fetch(
        "SELECT e.* FROM wf_run_events e JOIN wf_runs r ON r.id = e.run_id "
        "WHERE e.run_id = $1 AND r.account_id = $2 AND e.seq > $3 "
        "ORDER BY e.seq ASC LIMIT $4",
        run_id,
        account_id,
        after_seq,
        limit,
    )
    return [_row_to_wf_run_event(r) for r in rows]


async def find_open_gate_call_key(
    conn: asyncpg.Connection[Any], run_id: str, *, gate_nonce: str
) -> str | None:
    """Resolve a gate's capability ``nonce`` to the ``call_key`` of its OPEN gate.

    A gate is open when its ``call_started`` has no matching ``call_result`` yet; an
    already-resolved nonce (or one that never existed) returns ``None``. The earliest
    such gate wins (``ORDER BY seq``), though a nonce is unique per run in practice.

    Unscoped — the caller (:func:`aios.services.workflows.resume_gate_by_nonce`)
    account-checks the run via :func:`get_wf_run` first. Pushing the predicate into
    SQL (vs. materializing the whole journal and scanning in Python) keeps the cost a
    single indexed row instead of growing with journal length — the gate resolver,
    like the session ``lookup_tool_name_by_call_id`` it mirrors, lives in the query
    layer.
    """
    call_key: str | None = await conn.fetchval(
        """
        SELECT e.call_key
        FROM wf_run_events e
        WHERE e.run_id = $1
          AND e.type = 'call_started'
          AND e.payload->>'capability' = 'gate'
          AND e.payload->>'gate_nonce' = $2
          AND e.call_key IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM wf_run_events r
              WHERE r.run_id = e.run_id
                AND r.type = 'call_result'
                AND r.call_key = e.call_key
          )
        ORDER BY e.seq ASC
        LIMIT 1
        """,
        run_id,
        gate_nonce,
    )
    return call_key


# ─── wf_run_signals (external-resume side markers — idempotent) ───────────────


async def insert_run_signal(
    conn: asyncpg.Connection[Any],
    *,
    run_id: str,
    call_key: str,
    kind: WfRunSignalKind,
    result: Any = None,
) -> WfRunSignal:
    """Idempotently record a resume/completion signal for ``(run_id, call_key)``.

    A second delivery (double-click, at-least-once webhook) returns the existing
    row rather than erroring — the journal harvest is what makes it durable.
    """
    # A no-op ``DO UPDATE`` (vs. ``DO NOTHING``) makes the conflicting row
    # "matched" so ``RETURNING`` yields it in a single round-trip — the existing
    # row's ``kind``/``result`` are left untouched (first delivery wins).
    row = await conn.fetchrow(
        """
        INSERT INTO wf_run_signals (run_id, call_key, kind, result)
        VALUES ($1, $2, $3, $4::jsonb)
        ON CONFLICT (run_id, call_key)
            DO UPDATE SET delivered_at = wf_run_signals.delivered_at
        RETURNING *
        """,
        run_id,
        call_key,
        kind,
        json.dumps(result) if result is not None else None,
    )
    assert row is not None
    return _row_to_wf_run_signal(row)


async def list_run_signals(conn: asyncpg.Connection[Any], run_id: str) -> list[WfRunSignal]:
    rows = await conn.fetch("SELECT * FROM wf_run_signals WHERE run_id = $1", run_id)
    return [_row_to_wf_run_signal(r) for r in rows]
