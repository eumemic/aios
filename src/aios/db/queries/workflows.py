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
from typing import Any, NamedTuple

import asyncpg

from aios.db.queries import (
    _archive_scoped,
    _get_scoped,
    _get_versioned,
    _list_scoped,
    _list_versioned,
    parse_jsonb,
)
from aios.errors import ConflictError, NotFoundError
from aios.ids import WORKFLOW, WORKFLOW_EVENT, WORKFLOW_RUN, make_id
from aios.models.agents import HttpServerSpec, McpServerSpec, ToolSpec
from aios.models.workflows import (
    WfRun,
    WfRunEvent,
    WfRunEventType,
    WfRunSignal,
    WfRunSignalKind,
    WfRunStatus,
    Workflow,
    WorkflowVersion,
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
        archived_at=row["archived_at"],
    )


def _row_to_workflow_version(row: asyncpg.Record) -> WorkflowVersion:
    return WorkflowVersion(
        workflow_id=row["workflow_id"],
        version=row["version"],
        name=row["name"],
        script=row["script"],
        input_schema=parse_jsonb(row["input_schema"]),
        output_schema=parse_jsonb(row["output_schema"]),
        description=row["description"],
        tools=[ToolSpec.model_validate(t) for t in parse_jsonb(row["tools"])],
        mcp_servers=[McpServerSpec.model_validate(s) for s in parse_jsonb(row["mcp_servers"])],
        http_servers=[HttpServerSpec.model_validate(s) for s in parse_jsonb(row["http_servers"])],
        created_at=row["created_at"],
    )


def _row_to_wf_run(row: asyncpg.Record) -> WfRun:
    return WfRun(
        id=row["id"],
        workflow_id=row["workflow_id"],
        account_id=row["account_id"],
        environment_id=row["environment_id"],
        parent_run_id=row["parent_run_id"],
        launcher_session_id=row["launcher_session_id"],
        depth=row["depth"],
        request_id=row.get("request_id"),
        caller=parse_jsonb(row.get("caller")),
        request_output_schema=parse_jsonb(row.get("request_output_schema")),
        script=row["script"],
        script_sha=row["script_sha"],
        host_semantics_epoch=row["host_semantics_epoch"],
        tools=[ToolSpec.model_validate(t) for t in parse_jsonb(row["tools"])],
        mcp_servers=[McpServerSpec.model_validate(s) for s in parse_jsonb(row["mcp_servers"])],
        http_servers=[HttpServerSpec.model_validate(s) for s in parse_jsonb(row["http_servers"])],
        status=row["status"],
        input=parse_jsonb(row["input"]),
        output=parse_jsonb(row["output"]),
        budget_usd=(
            row["budget_total_microusd"] / 1_000_000
            if row.get("budget_total_microusd") is not None
            else None
        ),
        default_child_model=row.get("default_child_model"),
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
    """Insert a workflow definition at version 1 **and** snapshot it into
    ``workflow_versions(v1)`` in the same transaction. Raises ``ConflictError``
    on a duplicate ``(account_id, name)``.

    Copy-on-write (mirror of ``insert_agent``): the two writes are wrapped in a
    single ``conn.transaction()`` so a fresh workflow can never exist without its
    matching v1 history row. The version snapshot is driven off the ``workflows``
    write's ``RETURNING *`` row, so the two rows cannot disagree."""
    new_id = make_id(WORKFLOW)
    try:
        async with conn.transaction():
            row = await conn.fetchrow(
                """
                INSERT INTO workflows
                    (id, account_id, name, version, script, input_schema, output_schema,
                     description, tools, mcp_servers, http_servers)
                VALUES ($1, $2, $3, 1, $4, $5::jsonb, $6::jsonb, $7,
                        $8::jsonb, $9::jsonb, $10::jsonb)
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
            assert row is not None
            await _insert_workflow_version(conn, row)
    except asyncpg.UniqueViolationError as exc:
        raise ConflictError(
            f"workflow {name!r} already exists",
            detail={"name": name},
        ) from exc
    return _row_to_workflow(row)


async def _insert_workflow_version(conn: asyncpg.Connection[Any], wf_row: asyncpg.Record) -> None:
    """Snapshot a ``workflows`` row into ``workflow_versions`` (copy-on-write).

    Driven off the ``workflows`` write's ``RETURNING *`` row so the version row
    cannot disagree with the head it snapshots. The jsonb columns are re-dumped
    from the (already-parsed) record values; ``input_schema``/``output_schema``
    pass through as ``None`` when absent. Called inside the caller's transaction
    (``insert_workflow`` / ``update_workflow``); a torn write — a bumped
    ``workflows.version`` with no version row — would leave the head FK-dead in
    later phases, so the two writes must commit or roll back together."""
    input_schema = parse_jsonb(wf_row["input_schema"])
    output_schema = parse_jsonb(wf_row["output_schema"])
    await conn.execute(
        """
        INSERT INTO workflow_versions (
            workflow_id, account_id, version, name, script,
            input_schema, output_schema, description, tools, mcp_servers, http_servers
        )
        VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb, $8, $9::jsonb, $10::jsonb, $11::jsonb)
        """,
        wf_row["id"],
        wf_row["account_id"],
        wf_row["version"],
        wf_row["name"],
        wf_row["script"],
        json.dumps(input_schema) if input_schema is not None else None,
        json.dumps(output_schema) if output_schema is not None else None,
        wf_row["description"],
        json.dumps(parse_jsonb(wf_row["tools"])),
        json.dumps(parse_jsonb(wf_row["mcp_servers"])),
        json.dumps(parse_jsonb(wf_row["http_servers"])),
    )


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
    if current.archived_at is not None:
        raise ConflictError(f"workflow {workflow_id} is archived", detail={"id": workflow_id})
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

    # Copy-on-write: the version bump and the matching ``workflow_versions``
    # INSERT are wrapped in ONE ``conn.transaction()`` (the table ran autocommit
    # before this — unlike ``insert_agent``, already transaction-wrapped). A torn
    # write (bumped ``workflows.version`` with no version row) is corruption that
    # would leave the head FK-dead in later phases, so the two writes commit or
    # roll back together. The snapshot is driven off the UPDATE's ``RETURNING *``
    # row so the two rows cannot disagree.
    try:
        async with conn.transaction():
            row = await conn.fetchrow(
                """
                UPDATE workflows
                   SET version = workflows.version + 1, name = $3, script = $4,
                       input_schema = $5::jsonb, output_schema = $6::jsonb, description = $7,
                       tools = $8::jsonb, mcp_servers = $9::jsonb, http_servers = $10::jsonb,
                       updated_at = now()
                 WHERE id = $1 AND account_id = $2 AND archived_at IS NULL AND version = $11
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
            if row is None:
                # No row matched (id, account_id, version): a stale/raced
                # ``expected_version``. The re-read makes the 409 message accurate
                # (and would 404 a vanished row, though workflows have no delete
                # path today). Raising rolls back the (empty) transaction.
                fresh = await get_workflow(conn, workflow_id, account_id=account_id)
                raise ConflictError(
                    f"version mismatch: expected {expected_version}, current is {fresh.version}",
                    detail={
                        "expected": expected_version,
                        "current": fresh.version,
                        "id": workflow_id,
                    },
                )
            await _insert_workflow_version(conn, row)
    except asyncpg.UniqueViolationError as exc:
        # A rename onto another live workflow's name (UNIQUE(account_id, name)).
        raise ConflictError(
            f"workflow {new_name!r} already exists",
            detail={"name": new_name},
        ) from exc
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
    return await _list_scoped(
        conn,
        table="workflows",
        account_id=account_id,
        row=_row_to_workflow,
        limit=limit,
        after=after,
        filters=[("name", name)],
    )


async def archive_workflow(
    conn: asyncpg.Connection[Any], workflow_id: str, *, account_id: str
) -> Workflow:
    row = await _archive_scoped(
        conn,
        table="workflows",
        id_=workflow_id,
        account_id=account_id,
        noun="workflow",
    )
    return _row_to_workflow(row)


async def unarchive_workflow(
    conn: asyncpg.Connection[Any], workflow_id: str, *, account_id: str
) -> Workflow:
    try:
        row = await conn.fetchrow(
            """
            UPDATE workflows
               SET archived_at = NULL, updated_at = now()
             WHERE id = $1 AND account_id = $2 AND archived_at IS NOT NULL
            RETURNING *
            """,
            workflow_id,
            account_id,
        )
    except asyncpg.UniqueViolationError as exc:
        current = await get_workflow(conn, workflow_id, account_id=account_id)
        raise ConflictError(
            f"workflow {current.name!r} already exists",
            detail={"name": current.name},
        ) from exc
    if row is None:
        raise NotFoundError(
            f"workflow {workflow_id} not found or not archived", detail={"id": workflow_id}
        )
    return _row_to_workflow(row)


# ─── workflow_versions (immutable definition history) ────────────────────────


async def get_workflow_version(
    conn: asyncpg.Connection[Any],
    workflow_id: str,
    version: int,
    *,
    account_id: str,
) -> WorkflowVersion:
    """Read one historical version snapshot. Archived-blind on the parent (the
    correct behavior for post-mortem audit of an archived workflow — matches the
    agent version reads)."""
    return await _get_versioned(
        conn,
        table="workflow_versions",
        parent_column="workflow_id",
        parent_id=workflow_id,
        version=version,
        account_id=account_id,
        row=_row_to_workflow_version,
        noun="workflow",
    )


async def list_workflow_versions(
    conn: asyncpg.Connection[Any],
    workflow_id: str,
    *,
    account_id: str,
    limit: int = 50,
    after: int | None = None,
) -> list[WorkflowVersion]:
    """List a workflow's versions in descending order (newest first). Archived-blind
    on the parent (post-mortem audit of an archived workflow — matches agents)."""
    return await _list_versioned(
        conn,
        table="workflow_versions",
        parent_column="workflow_id",
        parent_id=workflow_id,
        account_id=account_id,
        row=_row_to_workflow_version,
        limit=limit,
        after=after,
    )


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


async def get_run_depth(conn: asyncpg.Connection[Any], run_id: str, *, account_id: str) -> int:
    """Read a run's DOWN-counting trusted invoke-depth (#1124), account-scoped.

    The remaining trusted-edge budget on ``run_id`` — what a sub-launch off this run
    may carry as ``depth - 1``. ACCOUNT-SCOPED: a foreign or missing parent raises
    ``NotFoundError`` (the same-account trust the deleted ``run_ancestor_depth`` CTE
    enforced per hop, now a single point read). The depth is immutable once written,
    so the read is race-free without locking.
    """
    depth: int | None = await conn.fetchval(
        "SELECT depth FROM wf_runs WHERE id = $1 AND account_id = $2",
        run_id,
        account_id,
    )
    if depth is None:
        raise NotFoundError(f"workflow run {run_id} not found", detail={"id": run_id})
    return depth


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
    host_semantics_epoch: int,
    input: Any = None,
    run_id: str | None = None,
    parent_run_id: str | None = None,
    launcher_session_id: str | None = None,
    request_id: str | None = None,
    caller: dict[str, Any] | None = None,
    request_output_schema: dict[str, Any] | None = None,
    tools: list[ToolSpec] | None = None,
    mcp_servers: list[McpServerSpec] | None = None,
    http_servers: list[HttpServerSpec] | None = None,
    budget_usd: float | None = None,
    default_child_model: str | None = None,
    depth: int,
) -> WfRun:
    """Insert a fresh ``pending`` run that snapshots ``script`` (+ ``script_sha``) and the
    declared tool surface (``tools``/``mcp_servers``/``http_servers``) — pinned at launch.
    ``launcher_session_id`` records the agent session that launched it (NULL = operator).

    ``depth`` is the DOWN-counting trusted invoke-depth budget (#1124) carried on the run:
    the remaining hops this run may spend on its OUTGOING trusted edges. The caller
    (``create_run``) computes it — full budget for an edgeless root, ``parent.depth - 1``
    for a nested launch — and refuses BEFORE calling here when the parent has none left,
    so no over-budget run row is ever written.

    ``run_id`` (#1129) pins the id to a deterministic value — the ``invoke_workflow``
    sub-run spawn's replay key. The insert is ``ON CONFLICT (id) DO NOTHING``, so a
    replay re-attaches the existing row (re-fetched here) instead of erroring; the
    default (``None``) mints a fresh ULID for which the conflict can never fire."""
    new_id = run_id if run_id is not None else make_id(WORKFLOW_RUN)
    try:
        row = await conn.fetchrow(
            """
            INSERT INTO wf_runs
                (id, workflow_id, account_id, environment_id, parent_run_id,
                 launcher_session_id, request_id, caller, request_output_schema,
                 script, script_sha, host_semantics_epoch, status, input,
                 tools, mcp_servers, http_servers, budget_total_microusd, default_child_model,
                 depth)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9::jsonb, $10, $11, $12,
                    'pending', $13::jsonb,
                    $14::jsonb, $15::jsonb, $16::jsonb, $17, $18, $19)
            ON CONFLICT (id) DO NOTHING
            RETURNING *
            """,
            new_id,
            workflow_id,
            account_id,
            environment_id,
            parent_run_id,
            launcher_session_id,
            request_id,
            json.dumps(caller) if caller is not None else None,
            json.dumps(request_output_schema) if request_output_schema is not None else None,
            script,
            script_sha,
            host_semantics_epoch,
            json.dumps(input) if input is not None else None,
            json.dumps([t.model_dump() for t in (tools or [])]),
            json.dumps([s.model_dump() for s in (mcp_servers or [])]),
            json.dumps([s.model_dump() for s in (http_servers or [])]),
            round(budget_usd * 1_000_000) if budget_usd is not None else None,
            default_child_model,
            depth,
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
    if row is None:
        # ON CONFLICT DO NOTHING fired — a concurrent replay already inserted this
        # deterministic id. Re-fetch the winning row and re-attach (#1129). Only
        # reachable on the ``run_id``-pinned path; a fresh ULID never conflicts.
        assert run_id is not None
        existing = await conn.fetchrow("SELECT * FROM wf_runs WHERE id = $1", run_id)
        assert existing is not None
        return _row_to_wf_run(existing)
    return _row_to_wf_run(row)


class RunChildrenUsage(NamedTuple):
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    cost_microusd: int


async def run_children_usage(
    conn: asyncpg.Connection[Any], run_id: str, *, account_id: str
) -> RunChildrenUsage:
    # TODO: this point-reads ``sessions.parent_run_id`` directly. The edge now
    # also carries it (``parent_run_id`` was subsumed into the ``request_opened``
    # edge in #1131); once a future contract migration retires the column, this
    # filter becomes a join through the ``request_opened`` lifecycle events
    # (``caller.kind='run' AND caller.id = $1``). #1123 leaves it untouched — the
    # dual-write keeps ``parent_run_id`` live for that future contract migration.
    row = await conn.fetchrow(
        """
        SELECT
            COALESCE(SUM(input_tokens), 0)::bigint AS input_tokens,
            COALESCE(SUM(output_tokens), 0)::bigint AS output_tokens,
            COALESCE(SUM(cache_read_input_tokens), 0)::bigint AS cache_read_input_tokens,
            COALESCE(SUM(cache_creation_input_tokens), 0)::bigint AS cache_creation_input_tokens,
            COALESCE(SUM(cost_microusd), 0)::bigint AS cost_microusd
          FROM sessions
         WHERE parent_run_id = $1 AND account_id = $2
        """,
        run_id,
        account_id,
    )
    assert row is not None
    return RunChildrenUsage(
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        cache_read_input_tokens=row["cache_read_input_tokens"],
        cache_creation_input_tokens=row["cache_creation_input_tokens"],
        cost_microusd=row["cost_microusd"],
    )


async def runs_children_usage(
    conn: asyncpg.Connection[Any], run_ids: list[str], *, account_id: str
) -> dict[str, RunChildrenUsage]:
    """Batched :func:`run_children_usage` for the read path (#1324).

    Sums each run's direct child sessions' usage in ONE grouped query, so
    ``list_runs`` enriches a whole page without an N+1 fan-out of point reads.
    Account-scoped identically to :func:`run_children_usage` (a child session
    of another tenant never contributes). Per-run semantics are verbatim the
    single-run query: ``COALESCE(SUM(...), 0)`` — a run with no children
    *appears in the result* with an all-zero record (a real, observed zero),
    so the caller can distinguish "summed to zero spend" from "run absent".

    Keep the column list and the ``parent_run_id``/``account_id`` filter in
    lockstep with :func:`run_children_usage`; the same #1131 ``parent_run_id``
    point-read note applies.
    """
    if not run_ids:
        return {}
    rows = await conn.fetch(
        """
        SELECT
            parent_run_id AS run_id,
            COALESCE(SUM(input_tokens), 0)::bigint AS input_tokens,
            COALESCE(SUM(output_tokens), 0)::bigint AS output_tokens,
            COALESCE(SUM(cache_read_input_tokens), 0)::bigint AS cache_read_input_tokens,
            COALESCE(SUM(cache_creation_input_tokens), 0)::bigint AS cache_creation_input_tokens,
            COALESCE(SUM(cost_microusd), 0)::bigint AS cost_microusd
          FROM sessions
         WHERE parent_run_id = ANY($1::text[]) AND account_id = $2
         GROUP BY parent_run_id
        """,
        run_ids,
        account_id,
    )
    summed = {
        row["run_id"]: RunChildrenUsage(
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            cache_read_input_tokens=row["cache_read_input_tokens"],
            cache_creation_input_tokens=row["cache_creation_input_tokens"],
            cost_microusd=row["cost_microusd"],
        )
        for row in rows
    }
    # A run with zero child sessions has no GROUP BY row; surface it as a real
    # all-zero record so every requested id is present in the result.
    zero = RunChildrenUsage(0, 0, 0, 0, 0)
    return {run_id: summed.get(run_id, zero) for run_id in run_ids}


async def unscoped_terminal_run_ids(conn: asyncpg.Connection[Any], run_ids: list[str]) -> set[str]:
    """Return the subset of ``run_ids`` whose ``status`` is TERMINAL (#1192).

    TERMINAL = ``status IN ('completed','errored','cancelled')`` (the exact
    members of ``models.workflows.TERMINAL_RUN_STATUSES``). This is the
    reap-set for the ``_runs/<wfr>`` per-run scratch reaper.

    ``_runs`` scratch is **NOT reconstructible** (unlike ``_session_repos``),
    so the reaper deletes ONLY on a *positively observed* terminal status. A
    run that is ``suspended`` (gate-paused, container idle-released) stays
    live in the DB and is absent here — its dir survives. A run id absent
    from the ``wf_runs`` table entirely is likewise absent here and is kept:
    we never delete non-reconstructible scratch on the mere *absence* of
    confirmation. Inverting that would be the data-loss bug PR #1193 shipped.

    Worker-side / unscoped: the reaper holds only the run ids it scraped off
    disk, across all accounts.
    """
    if not run_ids:
        return set()
    rows = await conn.fetch(
        """
        SELECT id FROM wf_runs
         WHERE id = ANY($1::text[])
           AND status IN ('completed','errored','cancelled')
        """,
        run_ids,
    )
    return {row["id"] for row in rows}


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
    conn: asyncpg.Connection[Any], run_id: str, status: WfRunStatus, *, account_id: str
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
    status: WfRunStatus,
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
      must force-resolve its timeout — this clause DRIVES that backstop) or a
      ``tool`` past the re-dispatch horizon (its task crashed without a signal).
      A ``gate`` maps to NULL — resume-driven only, never stale. Operator
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
                      END)
                AND NOT EXISTS (
                  SELECT 1 FROM wf_run_events e
                  WHERE e.run_id = r.id AND e.call_key = cs.call_key
                    AND e.type = 'call_result'))
          )
        """,
        agent_deadline_seconds,
        tool_stale_seconds,
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
    ``status`` + ``output``). Unscoped by account: every caller pre-checks the
    run with :func:`get_wf_run`. Exactly one ``run_completed`` bookend exists per run (the
    ``UNIQUE NULLS NOT DISTINCT (run_id, call_key, type)`` memo), so a bare fetch is exact."""
    row = await conn.fetchrow(
        "SELECT * FROM wf_run_events WHERE run_id = $1 AND type = 'run_completed'", run_id
    )
    return _row_to_wf_run_event(row) if row is not None else None


async def resolve_run_error(conn: asyncpg.Connection[Any], run_id: str) -> dict[str, Any] | None:
    """An errored run's ``{kind, …}`` error detail, or ``None``.

    THE extraction of ``error`` from the ``run_completed`` journal payload —
    shared by the run awaiter's completion record and the trigger fire path's
    composed envelope (#819), so the two surfaces can never drift on where
    ``error.kind`` lives (the row stores only ``status`` + ``output``).
    """
    completed = await get_run_completed_event(conn, run_id)
    if completed is None:
        return None
    error: dict[str, Any] | None = completed.payload.get("error")
    return error


async def derive_run_response(
    conn: asyncpg.Connection[Any], run_id: str, *, account_id: str
) -> dict[str, Any] | None:
    """A run-servicer's **terminal outcome** for its inbound request, or ``None`` if pending.

    The run-kind branch of the kind-agnostic resolver (the dual of the session-side
    :func:`aios.db.queries.sessions.derive_response`). A run is **singly-inbound** — its
    terminal state *is* the answer to its one request — so the outcome reads off the run's
    terminal record, not a per-request event:

    * **terminal** (``wf_runs.status``): ``cancelled`` → a clean ``cancelled`` outcome (a
      cancelled run writes no success, and its caller must learn ``cancelled``, never the
      ``child_gone`` a missing response would imply); ``completed``/``errored`` → the
      ``run_completed`` bookend payload's ``{output, is_error, error}``;
    * else **gone** (missing, or archived before completing — so it can never answer) →
      a Failed ``child_gone`` outcome;
    * else → ``None`` (alive and unanswered — still pending).

    Row status + the ``run_completed`` payload are read in one snapshot. This replaces the
    former separate ``request_response`` journal event (§3.6): the run's terminal record
    already carries the answer, so a second per-request event was redundant — and folding
    the read here is what makes a cancelled sub-run resolve as ``cancelled`` (the old event
    was gated off for cancellation, leaving the liveness arm to mislabel it ``child_gone``).
    """
    row = await conn.fetchrow(
        "SELECT r.status, (r.archived_at IS NOT NULL) AS archived, "
        "       (SELECT e.payload FROM wf_run_events e "
        "        WHERE e.run_id = r.id AND e.type = 'run_completed' "
        "        ORDER BY e.seq DESC LIMIT 1) AS completed "
        "FROM wf_runs r WHERE r.id = $1 AND r.account_id = $2",
        run_id,
        account_id,
    )
    if row is None:  # the run vanished entirely → can never answer
        return {"result": None, "is_error": True, "error": {"kind": "child_gone"}}
    status = row["status"]
    if status == "cancelled":
        return {"result": None, "is_error": True, "error": {"kind": "cancelled"}}
    if status in ("completed", "errored"):
        completed = parse_jsonb(row["completed"]) if row["completed"] is not None else {}
        is_error = bool(completed.get("is_error"))
        return {
            "result": None if is_error else completed.get("output"),
            "is_error": is_error,
            "error": completed.get("error"),
        }
    if row["archived"]:  # non-terminal but archived → can never answer
        return {"result": None, "is_error": True, "error": {"kind": "child_gone"}}
    return None


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
