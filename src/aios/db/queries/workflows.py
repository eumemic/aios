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

from aios.db.queries import parse_jsonb
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
    row = await conn.fetchrow(
        "SELECT * FROM workflows WHERE id = $1 AND account_id = $2",
        workflow_id,
        account_id,
    )
    if row is None:
        raise NotFoundError(f"workflow {workflow_id} not found", detail={"id": workflow_id})
    return _row_to_workflow(row)


# ─── wf_runs (execution instances) ───────────────────────────────────────────


async def insert_wf_run(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    workflow_id: str,
    script: str,
    script_sha: str,
    input: dict[str, Any] | None = None,
    parent_run_id: str | None = None,
) -> WfRun:
    """Insert a fresh ``pending`` run that snapshots ``script`` (+ ``script_sha``)."""
    new_id = make_id(WORKFLOW_RUN)
    try:
        row = await conn.fetchrow(
            """
            INSERT INTO wf_runs
                (id, workflow_id, account_id, parent_run_id, script, script_sha, status, input)
            VALUES ($1, $2, $3, $4, $5, $6, 'pending', $7::jsonb)
            RETURNING *
            """,
            new_id,
            workflow_id,
            account_id,
            parent_run_id,
            script,
            script_sha,
            json.dumps(input) if input is not None else None,
        )
    except asyncpg.ForeignKeyViolationError as exc:
        raise NotFoundError(
            f"workflow {workflow_id} not found", detail={"workflow_id": workflow_id}
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


async def set_run_output(
    conn: asyncpg.Connection[Any], run_id: str, output: Any, *, account_id: str
) -> None:
    await conn.execute(
        "UPDATE wf_runs SET output = $3::jsonb WHERE id = $1 AND account_id = $2",
        run_id,
        account_id,
        json.dumps(output),
    )


async def list_active_run_ids(conn: asyncpg.Connection[Any]) -> list[tuple[str, str]]:
    """``(id, account_id)`` for every non-terminal, live run — the sweep predicate."""
    rows = await conn.fetch(
        "SELECT id, account_id FROM wf_runs "
        "WHERE archived_at IS NULL AND status IN ('pending','running','suspended')"
    )
    return [(r["id"], r["account_id"]) for r in rows]


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
    """Append a journal event with a gapless seq; idempotent on
    ``(run_id, call_key, type)``.

    Returns the inserted event, or ``None`` when nothing was inserted — either
    the run is terminal/archived (the WHERE filters it) or an idempotent
    ``(run_id, call_key, type)`` conflict (the memo dedup). In both cases no seq
    is consumed, so the sequence stays gapless. The insert + the
    ``last_event_seq`` bump are atomic (own savepoint when the caller is already
    in a transaction — e.g. the ``run_completed`` + ``set_run_status`` unit).
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
    return _row_to_wf_run_event(row)


async def list_run_events(conn: asyncpg.Connection[Any], run_id: str) -> list[WfRunEvent]:
    rows = await conn.fetch("SELECT * FROM wf_run_events WHERE run_id = $1 ORDER BY seq", run_id)
    return [_row_to_wf_run_event(r) for r in rows]


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
