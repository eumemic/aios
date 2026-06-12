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
from aios.models.agents import HttpServerSpec, McpServerSpec, ToolSpec
from aios.models.workflows import (
    WfRun,
    WfRunEvent,
    WfRunEventType,
    WfRunSignal,
    WfRunSignalKind,
    Workflow,
)

# A reserved ``call_key`` for the run-cancel side-marker. Real call_keys are
# call-site hashes, so this sentinel never collides; the ``(run_id, call_key)`` PK
# makes it one cancel signal per run — i.e. cancellation is idempotent. The next
# ``run_workflow_step`` harvests it (under the lock) and finalizes ``cancelled``.
CANCEL_SIGNAL_CALL_KEY = "__cancel__"


def _row_to_workflow(row: asyncpg.Record) -> Workflow:
    return Workflow(
        id=row["id"],
        account_id=row["account_id"],
        name=row["name"],
        version=row["version"],
        script=row["script"],
        input_schema=parse_jsonb(row["input_schema"]),
        output_schema=parse_jsonb(row["output_schema"]),
        description=row["description"],
        tools=[ToolSpec.model_validate(t) for t in parse_jsonb(row["tools"])],
        mcp_servers=[McpServerSpec.model_validate(s) for s in parse_jsonb(row["mcp_servers"])],
        http_servers=[HttpServerSpec.model_validate(s) for s in parse_jsonb(row["http_servers"])],
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
        launcher_session_id=row["launcher_session_id"],
        script=row["script"],
        script_sha=row["script_sha"],
        tools=[ToolSpec.model_validate(t) for t in parse_jsonb(row["tools"])],
        mcp_servers=[McpServerSpec.model_validate(s) for s in parse_jsonb(row["mcp_servers"])],
        http_servers=[HttpServerSpec.model_validate(s) for s in parse_jsonb(row["http_servers"])],
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
    input_schema: dict[str, Any] | None = None,
    output_schema: dict[str, Any] | None = None,
    description: str | None = None,
    tools: list[ToolSpec] | None = None,
    mcp_servers: list[McpServerSpec] | None = None,
    http_servers: list[HttpServerSpec] | None = None,
) -> Workflow:
    """Insert a workflow definition at version 1. Raises ``ConflictError`` on a
    duplicate ``(account_id, name)``."""
    new_id = make_id(WORKFLOW)
    try:
        row = await conn.fetchrow(
            """
            INSERT INTO workflows
                (id, account_id, name, version, script, input_schema, output_schema,
                 description, tools, mcp_servers, http_servers)
            VALUES ($1, $2, $3, 1, $4, $5::jsonb, $6::jsonb, $7, $8::jsonb, $9::jsonb, $10::jsonb)
            RETURNING *
            """,
            new_id,
            account_id,
            name,
            script,
            json.dumps(input_schema) if input_schema is not None else None,
            json.dumps(output_schema) if output_schema is not None else None,
            description,
            json.dumps([t.model_dump() for t in (tools or [])]),
            json.dumps([s.model_dump() for s in (mcp_servers or [])]),
            json.dumps([s.model_dump() for s in (http_servers or [])]),
        )
    except asyncpg.UniqueViolationError as exc:
        raise ConflictError(
            f"workflow {name!r} already exists",
            detail={"name": name},
        ) from exc
    assert row is not None
    return _row_to_workflow(row)


async def update_workflow(
    conn: asyncpg.Connection[Any],
    workflow_id: str,
    *,
    account_id: str,
    expected_version: int,
    name: str | None = None,
    script: str | None = None,
    input_schema: dict[str, Any] | None = None,
    output_schema: dict[str, Any] | None = None,
    description: str | None = None,
    tools: list[ToolSpec] | None = None,
    mcp_servers: list[McpServerSpec] | None = None,
    http_servers: list[HttpServerSpec] | None = None,
) -> Workflow:
    """Update a workflow in place, bumping ``version`` (the ``update_agent`` shape,
    minus the version-snapshot — runs pin script + surface onto themselves at launch).

    Requires ``expected_version`` to match the current version (optimistic
    concurrency). Omitted fields are preserved. If nothing changed, the current row
    is returned without a bump (no-op). The version match is enforced by the
    UPDATE's ``AND version = $expected`` — the authoritative serialization point;
    exactly one of two racing writers matches, the other gets a clean 409.
    """
    current = await get_workflow(conn, workflow_id, account_id=account_id)
    if expected_version != current.version:
        # Validate the token up front so the contract is uniform — without this, the
        # write-free no-op path below would return 200 for a stale token + identical
        # values. This is *in addition to* the UPDATE's ``WHERE version`` (which stays
        # the authoritative write-time serializer), not instead of it — the pre-check
        # alone would be racy for the write path (the update_agent lesson).
        raise ConflictError(
            f"version mismatch: expected {expected_version}, current is {current.version}",
            detail={
                "expected": expected_version,
                "current": current.version,
                "id": workflow_id,
            },
        )

    # Resolve final values (omitted = preserve current).
    new_name = name if name is not None else current.name
    new_script = script if script is not None else current.script
    new_input_schema = input_schema if input_schema is not None else current.input_schema
    new_output_schema = output_schema if output_schema is not None else current.output_schema
    new_desc = description if description is not None else current.description
    new_tools = tools if tools is not None else current.tools
    new_mcp = mcp_servers if mcp_servers is not None else current.mcp_servers
    new_http = http_servers if http_servers is not None else current.http_servers

    if (
        new_name == current.name
        and new_script == current.script
        and new_input_schema == current.input_schema
        and new_output_schema == current.output_schema
        and new_desc == current.description
        and new_tools == current.tools
        and new_mcp == current.mcp_servers
        and new_http == current.http_servers
    ):
        return current

    try:
        row = await conn.fetchrow(
            """
            UPDATE workflows
               SET version = workflows.version + 1, name = $3, script = $4,
                   input_schema = $5::jsonb, output_schema = $6::jsonb, description = $7,
                   tools = $8::jsonb, mcp_servers = $9::jsonb, http_servers = $10::jsonb,
                   updated_at = now()
             WHERE id = $1 AND account_id = $2 AND version = $11
            RETURNING *
            """,
            workflow_id,
            account_id,
            new_name,
            new_script,
            json.dumps(new_input_schema) if new_input_schema is not None else None,
            json.dumps(new_output_schema) if new_output_schema is not None else None,
            new_desc,
            json.dumps([t.model_dump() for t in new_tools]),
            json.dumps([s.model_dump() for s in new_mcp]),
            json.dumps([s.model_dump() for s in new_http]),
            expected_version,
        )
    except asyncpg.UniqueViolationError as exc:
        # A rename onto another live workflow's name (UNIQUE(account_id, name)).
        raise ConflictError(
            f"workflow {new_name!r} already exists",
            detail={"name": new_name},
        ) from exc
    if row is None:
        # No row matched (id, account_id, version): a stale/raced ``expected_version``.
        # The re-read makes the 409 message accurate (and would 404 a vanished row,
        # though workflows have no delete path today).
        fresh = await get_workflow(conn, workflow_id, account_id=account_id)
        raise ConflictError(
            f"version mismatch: expected {expected_version}, current is {fresh.version}",
            detail={
                "expected": expected_version,
                "current": fresh.version,
                "id": workflow_id,
            },
        )
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
    ``archived_at`` column — workflows are never archived. One row per name
    (``UNIQUE(account_id, name)``); ``version`` bumps in place on update.
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
    parent_run_id: str | None = None,
    launcher_session_id: str | None = None,
) -> list[WfRun]:
    """Keyset-paginated list of an account's runs (non-archived), newest first.

    ``parent_run_id`` scopes to a run's children — the runs a workflow's nested
    ``workflow()`` calls spawned, plus (#819) trigger-launched runs whose
    lineage parent it is (a run_completion fire threads the completing run's
    id; a timer fire on a workflow-child session threads that session's own
    parent run). The run-side analog of filtering sessions by
    ``parent_run_id`` for a run's ``agent()`` children.

    When ``launcher_session_id`` is set, the launcher filter lists ALL of a
    session's runs including terminal ones, so it is NOT fully served by the
    ``wf_runs_launcher_active_idx`` partial index (which covers only ``status IN
    ('pending','running','suspended')``); the account-scoped ``ORDER BY id DESC``
    keyset still applies.
    """
    return await _list_scoped(
        conn,
        table="wf_runs",
        account_id=account_id,
        row=_row_to_wf_run,
        limit=limit,
        after=after,
        filters=[
            ("workflow_id", workflow_id),
            ("status", status),
            ("parent_run_id", parent_run_id),
            ("launcher_session_id", launcher_session_id),
        ],
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
    launcher_session_id: str | None = None,
    tools: list[ToolSpec] | None = None,
    mcp_servers: list[McpServerSpec] | None = None,
    http_servers: list[HttpServerSpec] | None = None,
) -> WfRun:
    """Insert a fresh ``pending`` run that snapshots ``script`` (+ ``script_sha``) and the
    declared tool surface (``tools``/``mcp_servers``/``http_servers``) — pinned at launch.
    ``launcher_session_id`` records the agent session that launched it (NULL = operator)."""
    new_id = make_id(WORKFLOW_RUN)
    try:
        row = await conn.fetchrow(
            """
            INSERT INTO wf_runs
                (id, workflow_id, account_id, environment_id, parent_run_id,
                 launcher_session_id, script, script_sha, status, input,
                 tools, mcp_servers, http_servers)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'pending', $9::jsonb,
                    $10::jsonb, $11::jsonb, $12::jsonb)
            RETURNING *
            """,
            new_id,
            workflow_id,
            account_id,
            environment_id,
            parent_run_id,
            launcher_session_id,
            script,
            script_sha,
            json.dumps(input) if input is not None else None,
            json.dumps([t.model_dump() for t in (tools or [])]),
            json.dumps([s.model_dump() for s in (mcp_servers or [])]),
            json.dumps([s.model_dump() for s in (http_servers or [])]),
        )
    except asyncpg.ForeignKeyViolationError as exc:
        raise NotFoundError(
            "a referenced resource is gone: workflow, environment, parent run, or launcher session",
            detail={
                "workflow_id": workflow_id,
                "environment_id": environment_id,
                "parent_run_id": parent_run_id,
                "launcher_session_id": launcher_session_id,
            },
        ) from exc
    assert row is not None
    return _row_to_wf_run(row)


async def run_ancestor_depth(conn: asyncpg.Connection[Any], run_id: str, *, account_id: str) -> int:
    """Depth of ``run_id`` in the ``parent_run_id`` chain (a root run = 1).

    Walks ``wf_runs.parent_run_id`` upward via a recursive CTE, account-scoped on
    every hop. Returns 0 if ``run_id`` doesn't exist for the account (defensive; the
    depth cap passes a live ancestor). The chain is immutable once written, so the
    count is race-free without locking — concurrent sibling inserts can't change an
    ancestor's depth.
    """
    depth: int | None = await conn.fetchval(
        """
        WITH RECURSIVE chain AS (
            SELECT id, parent_run_id, 1 AS depth
              FROM wf_runs WHERE id = $1 AND account_id = $2
            UNION ALL
            SELECT r.id, r.parent_run_id, c.depth + 1
              FROM wf_runs r JOIN chain c ON r.id = c.parent_run_id
             WHERE r.account_id = $2
        )
        SELECT max(depth) FROM chain
        """,
        run_id,
        account_id,
    )
    return depth or 0


async def count_active_runs(
    conn: asyncpg.Connection[Any], *, account_id: str, launcher_session_id: str | None = None
) -> int:
    """Count the account's OUTSTANDING (non-terminal, non-archived) runs — optionally
    only those launched by one session. The two fan-out cap reads.

    The status list is verbatim-identical to the ``wf_runs_launcher_active_idx``
    predicate (migration 0078) — keep them in sync so the planner uses the partial
    index — and is the exact complement of ``TERMINAL_RUN_STATUSES``.

    Cap invariant: ``archived_at IS NULL`` assumes archival is terminal-only. No run
    archival exists today; if one lands, it must refuse non-terminal runs (or take
    ``acquire_account_wf_runs_lock``), else an archived-but-running run would escape
    the count while still consuming real capacity.
    """
    if launcher_session_id is None:
        count: int = await conn.fetchval(
            """
            SELECT count(*) FROM wf_runs
             WHERE account_id = $1 AND archived_at IS NULL
               AND status IN ('pending','running','suspended')
            """,
            account_id,
        )
        return count
    count = await conn.fetchval(
        """
        SELECT count(*) FROM wf_runs
         WHERE account_id = $1 AND launcher_session_id = $2 AND archived_at IS NULL
           AND status IN ('pending','running','suspended')
        """,
        account_id,
        launcher_session_id,
    )
    return count


async def acquire_account_wf_runs_lock(conn: asyncpg.Connection[Any], account_id: str) -> None:
    """Take the per-account, transaction-scoped advisory lock serializing the fan-out
    cap's COUNT+INSERT (released automatically at commit/rollback).

    One account-level lock guards BOTH caps (a launcher's runs are a subset of its
    account's), making them contractual against concurrent launches rather than
    best-effort. The key is ``hashtextextended('aios_wf_runs_cap:' || account_id, 0)``
    — a distinct namespace from the scheduled-tasks and worker-singleton locks, and
    the two cap locks are never co-held in one transaction (no deadlock cycle).
    """
    await conn.execute(
        "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
        f"aios_wf_runs_cap:{account_id}",
    )


async def set_run_vaults(
    conn: asyncpg.Connection[Any],
    run_id: str,
    vault_ids: list[str],
    *,
    account_id: str,
) -> None:
    """Bind vaults to a run, rank-ordered (first-match credential resolution).

    The run analog of :func:`set_session_vaults`, with a tighter ownership guard: each
    vault must exist **and belong to ``account_id``** — a foreign or unknown vault id
    raises ``NotFoundError``, so a run can never bind another account's vault. (The
    ``vault_id`` FK alone checks existence, not ownership — the same tenant-confusion the
    ``create_vault_credential`` lock guards against; we mirror that here rather than
    inherit the gap.) Duplicate ids are de-duplicated (a binding is a set). Called inside
    ``create_run``'s transaction, so its inner ``conn.transaction()`` is a savepoint.
    """
    deduped = list(dict.fromkeys(vault_ids))
    async with conn.transaction():
        await conn.execute(
            "DELETE FROM wf_run_vaults WHERE run_id = $1 AND account_id = $2",
            run_id,
            account_id,
        )
        if not deduped:
            return
        owned = {
            str(r["id"])
            for r in await conn.fetch(
                "SELECT id FROM vaults WHERE id = ANY($1) AND account_id = $2",
                deduped,
                account_id,
            )
        }
        missing = [v for v in deduped if v not in owned]
        if missing:
            raise NotFoundError(
                f"vault(s) not found in this account: {missing}",
                detail={"vault_ids": missing},
            )
        for rank, vault_id in enumerate(deduped):
            await conn.execute(
                "INSERT INTO wf_run_vaults (run_id, vault_id, rank, account_id) "
                "VALUES ($1, $2, $3, $4)",
                run_id,
                vault_id,
                rank,
                account_id,
            )


async def get_run_vault_ids(
    conn: asyncpg.Connection[Any], run_id: str, *, account_id: str
) -> list[str]:
    rows = await conn.fetch(
        "SELECT vault_id FROM wf_run_vaults WHERE run_id = $1 AND account_id = $2 ORDER BY rank",
        run_id,
        account_id,
    )
    return [str(r["vault_id"]) for r in rows]


async def get_run_for_step(conn: asyncpg.Connection[Any], run_id: str) -> WfRun | None:
    """Load a run by id (no account scoping — the step is internal/trusted).

    Returns ``None`` if the run no longer exists, so a stray wake is a no-op.
    """
    row = await conn.fetchrow("SELECT * FROM wf_runs WHERE id = $1", run_id)
    return _row_to_wf_run(row) if row is not None else None


async def set_run_status(
    conn: asyncpg.Connection[Any], run_id: str, status: str, *, account_id: str
) -> None:
    """Set a NON-terminal status transition. Never overwrites a terminal status:
    under procrastinate dual execution (a reaped stale worker resuming next to its
    replacement), a stale step's lease flip or park must not resurrect a run the
    replacement already completed. Terminal writes go through
    :func:`set_run_terminal`."""
    await conn.execute(
        "UPDATE wf_runs SET status = $3, updated_at = now() "
        "WHERE id = $1 AND account_id = $2 "
        "AND status NOT IN ('completed','errored','cancelled')",
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

    Called from ``_commit_terminal_and_dispatch`` inside the same txn as the
    ``run_completed`` append. ``output`` is the script's return value on
    success and ``None`` on error (the error detail lives in the
    ``run_completed`` payload).

    First-writer-wins (``status NOT IN`` terminal guard): under procrastinate
    dual execution the journal memo already makes exactly one ``run_completed``
    append win — without this guard the LOSER's flip would still overwrite the
    row last-writer-wins, letting ``wf_runs.status`` diverge from the journal
    bookend and from the run_completion fires the winner dispatched (#819: a
    statuses-filtered watcher's decision must agree with the final row).
    """
    await conn.execute(
        "UPDATE wf_runs SET status = $3, output = $4::jsonb, updated_at = now() "
        "WHERE id = $1 AND account_id = $2 "
        "AND status NOT IN ('completed', 'errored', 'cancelled')",
        run_id,
        account_id,
        status,
        json.dumps(output) if output is not None else None,
    )


async def list_run_ids_needing_step(
    conn: asyncpg.Connection[Any],
    *,
    agent_deadline_seconds: float,
    tool_stale_seconds: float,
    sandbox_stale_seconds: float,
) -> list[str]:
    """``id`` for every live run with something for a step to DO — the sweep
    predicate (#780). A parked run with nothing new is deliberately NOT matched
    (waking it costs a full memo reship + script replay), so each clause carries a
    recall obligation; the fail direction must always be "wake too much":

    - ``pending``/``running`` — seeded-but-never-stepped, and the step lease
      (every step flips ``running`` before its first journal write, so ANY
      mid-step crash — including the deliberate spawn-failed re-raise — lands
      here until the step's closing park/terminal write).
    - an unharvested signal (``gate_resume``/``tool_result``/``child_done``/
      ``cancel`` row with no matching ``call_result``) — every external resume
      and child completion commits one atomically with its payload, so a lost
      ``defer_run_wake`` is always visible here.
    - a stale inflight call: an ``agent`` past the wall-clock deadline (the step
      must force-resolve its timeout — this clause DRIVES that backstop), or a
      ``tool``/``sandbox`` past its re-dispatch horizon (its task crashed without
      writing a signal). A ``gate`` maps to NULL — resume-driven only, never
      stale. Operator
      archive/delete of a child BEFORE it answers now COMPLETES the open request
      EAGERLY (the service layer fails it ``child_gone`` and writes a ``child_done``
      signal atomically with the archive/delete, like every other completion), so
      it is recalled via the unharvested-signal clause above within a tick — not
      this deadline. The agent deadline remains only the backstop for a genuinely
      stuck or non-responding LIVE child.

    (No ``account_id``: ``defer_run_wake`` needs none and appends no journal span.)
    """
    rows = await conn.fetch(
        """
        SELECT r.id FROM wf_runs r
        WHERE r.archived_at IS NULL
          AND r.status IN ('pending','running','suspended')
          AND (
            r.status IN ('pending','running')
            OR EXISTS (
              SELECT 1 FROM wf_run_signals s
              WHERE s.run_id = r.id
                AND NOT EXISTS (
                  SELECT 1 FROM wf_run_events e
                  WHERE e.run_id = r.id AND e.call_key = s.call_key
                    AND e.type = 'call_result'))
            OR EXISTS (
              SELECT 1 FROM wf_run_events cs
              WHERE cs.run_id = r.id AND cs.type = 'call_started'
                AND cs.created_at < now() - make_interval(secs =>
                      CASE cs.payload->>'capability'
                        WHEN 'agent' THEN $1::float8
                        WHEN 'tool' THEN $2::float8
                        WHEN 'sandbox' THEN $3::float8
                      END)
                AND NOT EXISTS (
                  SELECT 1 FROM wf_run_events e
                  WHERE e.run_id = r.id AND e.call_key = cs.call_key
                    AND e.type = 'call_result'))
          )
        """,
        agent_deadline_seconds,
        tool_stale_seconds,
        sandbox_stale_seconds,
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
              AND r.status NOT IN ('completed', 'errored', 'cancelled')
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


async def get_run_completed_event(conn: asyncpg.Connection[Any], run_id: str) -> WfRunEvent | None:
    """The run's terminal ``run_completed`` event, or ``None`` if it hasn't completed.

    Its payload carries the authoritative ``{output, is_error, error}`` completion detail —
    in particular ``error.kind``, which ``wf_runs`` does not store (the row keeps only
    ``status`` + ``output``). Unscoped by account: every caller (``await_run``) pre-checks the
    run with :func:`get_wf_run`. Exactly one ``run_completed`` bookend exists per run (the
    ``UNIQUE NULLS NOT DISTINCT (run_id, call_key, type)`` memo), so a bare fetch is exact."""
    row = await conn.fetchrow(
        "SELECT * FROM wf_run_events WHERE run_id = $1 AND type = 'run_completed'", run_id
    )
    return _row_to_wf_run_event(row) if row is not None else None


async def resolve_run_error(conn: asyncpg.Connection[Any], run_id: str) -> dict[str, Any] | None:
    """An errored run's ``{kind, …}`` error detail, or ``None``.

    THE extraction of ``error`` from the ``run_completed`` journal payload —
    shared by ``await_run``'s completion record and the trigger fire path's
    composed envelope (#819), so the two surfaces can never drift on where
    ``error.kind`` lives (the row stores only ``status`` + ``output``).
    """
    completed = await get_run_completed_event(conn, run_id)
    if completed is None:
        return None
    error: dict[str, Any] | None = completed.payload.get("error")
    return error


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


async def read_run_signal(
    conn: asyncpg.Connection[Any], run_id: str, call_key: str
) -> WfRunSignal | None:
    """A fresh point-read of one signal — used to disambiguate the harvest's stale snapshot.

    A fire-and-forget tool task commits its ``tool_result`` signal **before** it pops the
    in-flight registry, and it runs outside the run lock; so when a step's signal *snapshot*
    shows nothing yet the task is no longer in-flight, the signal may have just landed. This
    point-read tells "crashed" (None → re-dispatch) from "just finished" (resolve) without
    re-dispatching a tool that already ran.
    """
    row = await conn.fetchrow(
        "SELECT * FROM wf_run_signals WHERE run_id = $1 AND call_key = $2", run_id, call_key
    )
    return _row_to_wf_run_signal(row) if row is not None else None
