"""Agent-acting workflow builtins — the strange loop.

Model-callable tools that let an agent author, edit, launch, await, and cancel
workflows the way a human operator (or the ``aios`` CLI) does. Each is a thin
wrapper over the existing workflow services; the agent's authority is bounded by
the attenuations those services already enforce, keyed on the **executing
session id** the harness supplies (``invoke_builtin(session_id, …)``,
``tools/invoke.py``) — never model input:

* ``create_workflow`` — surface attenuation: the declared tool/server surface must
  be a subset of the *creating agent's* own.
* ``update_workflow`` — merged-surface attenuation + an optimistic ``version`` pin.
* ``create_run`` — vault attenuation (bound vaults ⊆ the *launching session's*) plus
  the vertical run-depth cap and the horizontal fan-out caps (outstanding runs per
  launcher and per account); the run inherits the caller's environment.
* ``await_run`` — a bounded long-poll that blocks (as a fire-and-forget tool task,
  so the session stays responsive) until the run is terminal or the timeout lapses.
* ``cancel_run`` — cancel-time attenuation: a session may cancel only runs *it
  launched* (the self-service escape for the fan-out cap; operator-launched runs
  need the operator).

**Identity is load-bearing, so two invariants hold (see F1 in the review):**
1. The trusted ids (``creator_session_id``/``actor_session_id``/``launcher_session_id``,
   ``account_id``, ``parent_run_id``, ``environment_id``) are NEVER tool-schema fields —
   every schema is ``additionalProperties: false`` (derived from a pydantic arg model
   with ``extra="forbid"``), so an injected key is rejected before the handler runs.
2. Handlers map service kwargs explicitly from the validated model — never ``**arguments``.

All register ``transport="agent_tool"`` (model-only; the CLI broker refuses them).

**Security boundary (A2):** these builtins attenuate, but a grant of the operator-level
management API (HTTP / a future management MCP) does not — that path is unattenuated by
design. Grant the management surface only to operator-trust agents; don't reason "the
builtins attenuate, therefore agent X is contained" if X also holds it.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from aios.config import get_settings
from aios.harness import runtime
from aios.models.workflows import WfRunStatus, WorkflowCreate, WorkflowUpdate
from aios.services import sessions as sessions_service
from aios.services import workflows as wf_service
from aios.tools.invoke import ToolBail
from aios.tools.registry import registry

# Heavy snapshot fields the model already sent (or doesn't need echoed back); trimmed
# from the returned dicts to keep tool results lean.
_WORKFLOW_ECHO_EXCLUDE = {"script"}
_RUN_ECHO_EXCLUDE = {"script", "script_sha", "tools", "mcp_servers", "http_servers"}

# list_workflows summaries: strip script + heavy schema/surface blobs, leaving
# id, account_id, name, version, description, timestamps (the locked summary set;
# account_id is harmless metadata, kept like the analogous run exclude set keeps it).
_WORKFLOW_LIST_EXCLUDE = {
    "script",
    "input_schema",
    "output_schema",
    "tools",
    "mcp_servers",
    "http_servers",
}


# ─── argument models (parameters_schema + parse, in one place) ───────────────


class _UpdateWorkflowArgs(WorkflowUpdate):
    """``update_workflow`` arguments — the ``WorkflowUpdate`` body plus the path-style id.

    Subclassing inherits every field (and its constraints) and ``extra="forbid"`` —
    the trusted ``actor_session_id`` is never a field, so an injected key is rejected.
    """

    workflow_id: str


class _WorkflowIdArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workflow_id: str


class _CreateRunArgs(BaseModel):
    """``create_run`` arguments. No ``environment_id`` — the run always inherits the
    caller session's env (a caller-chosen env id would be a cross-tenant attack surface)."""

    model_config = ConfigDict(extra="forbid")

    workflow_id: str
    input: Any = None
    vault_ids: list[str] = Field(default_factory=list)


class _AwaitRunArgs(BaseModel):
    """``await_run`` arguments. ``timeout_seconds`` is a bounded per-call long-poll
    budget; re-call to keep waiting (each call holds one LISTEN connection)."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    timeout_seconds: int = Field(default=300, ge=1, le=1800)


class _CancelRunArgs(BaseModel):
    """``cancel_run`` arguments — just the run id; the canceller is the trusted
    executing session (you may cancel only runs you launched)."""

    model_config = ConfigDict(extra="forbid")

    run_id: str


class _GetWorkflowArgs(_WorkflowIdArgs):
    pass


class _ListWorkflowsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    limit: int = Field(default=50, ge=1, le=200)
    after: str | None = None
    name: str | None = None


class _GetRunArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str


class _ListRunsArgs(BaseModel):
    """Defaults to THIS session's own runs (launcher-filtered). Set account_wide=True
    to widen to all of the account's runs. There is deliberately no launcher_session_id
    or account_id field — the launcher is the trusted executing session, and the
    account derives from it; the widen control is a bool, never an id."""

    model_config = ConfigDict(extra="forbid")

    account_wide: bool = False
    workflow_id: str | None = None
    status: WfRunStatus | None = None
    parent_run_id: str | None = None
    limit: int = Field(default=50, ge=1, le=200)
    after: str | None = None


class _ListRunEventsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    after_seq: int = Field(default=0, ge=0)
    limit: int = Field(default=200, ge=1, le=500)


# ─── handler plumbing ────────────────────────────────────────────────────────
#
# Handlers map service kwargs explicitly (F1) and otherwise just let service errors
# propagate: the dispatch layer (``tool_dispatch._classify_tool_error``) turns a
# client-class (4xx) ``AiosError`` — a denied attenuation, a stale-version conflict, a
# depth-cap hit — into a clean, model-visible result without evicting the sandbox, and
# a 5xx into a genuine failure. Only argument parsing bails locally (``_parse``).


def _parse[M: BaseModel](model: type[M], arguments: dict[str, Any]) -> M:
    """Parse + validate via the pydantic arg model (semantic checks the JSON schema
    can't encode, e.g. ``McpServerSpec`` name rules) → ``ToolBail`` on failure."""
    try:
        return model.model_validate(arguments)
    except PydanticValidationError as exc:
        raise ToolBail(f"invalid arguments: {exc}") from exc


async def create_workflow_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    body = _parse(WorkflowCreate, arguments)
    wf = await wf_service.create_workflow(
        pool,
        account_id=account_id,
        name=body.name,
        script=body.script,
        input_schema=body.input_schema,
        output_schema=body.output_schema,
        description=body.description,
        tools=body.tools,
        mcp_servers=body.mcp_servers,
        http_servers=body.http_servers,
        creator_session_id=session_id,
    )
    return wf.model_dump(mode="json", exclude=_WORKFLOW_ECHO_EXCLUDE)


async def update_workflow_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    args = _parse(_UpdateWorkflowArgs, arguments)
    wf = await wf_service.update_workflow(
        pool,
        args.workflow_id,
        account_id=account_id,
        expected_version=args.version,
        name=args.name,
        script=args.script,
        input_schema=args.input_schema,
        output_schema=args.output_schema,
        description=args.description,
        tools=args.tools,
        mcp_servers=args.mcp_servers,
        http_servers=args.http_servers,
        actor_session_id=session_id,
    )
    return wf.model_dump(mode="json", exclude=_WORKFLOW_ECHO_EXCLUDE)


async def archive_workflow_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    args = _parse(_WorkflowIdArgs, arguments)
    wf = await wf_service.archive_workflow(pool, args.workflow_id, account_id=account_id)
    return wf.model_dump(mode="json", exclude=_WORKFLOW_ECHO_EXCLUDE)


async def unarchive_workflow_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    args = _parse(_WorkflowIdArgs, arguments)
    wf = await wf_service.unarchive_workflow(pool, args.workflow_id, account_id=account_id)
    return wf.model_dump(mode="json", exclude=_WORKFLOW_ECHO_EXCLUDE)


async def create_run_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    args = _parse(_CreateRunArgs, arguments)
    session = await sessions_service.get_session_basic(pool, session_id, account_id=account_id)
    run = await wf_service.create_run(
        pool,
        account_id=account_id,
        workflow_id=args.workflow_id,
        environment_id=session.environment_id,  # inherit caller's env (F2)
        input=args.input,
        vault_ids=args.vault_ids,
        launcher_session_id=session_id,
        parent_run_id=session.parent_run_id,  # lineage + depth cap
    )
    return run.model_dump(mode="json", exclude=_RUN_ECHO_EXCLUDE)


async def await_run_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    args = _parse(_AwaitRunArgs, arguments)
    resp = await wf_service.await_run(
        pool,
        get_settings().db_url,
        args.run_id,
        account_id=account_id,
        timeout_seconds=args.timeout_seconds,
    )
    return resp.model_dump(mode="json")


async def cancel_run_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    args = _parse(_CancelRunArgs, arguments)
    run = await wf_service.cancel_run(
        pool,
        run_id=args.run_id,
        account_id=account_id,
        canceller_session_id=session_id,  # cancel only what this session launched
    )
    return run.model_dump(mode="json", exclude=_RUN_ECHO_EXCLUDE)


async def get_workflow_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    args = _parse(_GetWorkflowArgs, arguments)
    wf = await wf_service.get_workflow(pool, args.workflow_id, account_id=account_id)
    return wf.model_dump(mode="json")  # FULL — includes script + version (the re-read loop)


async def list_workflows_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    args = _parse(_ListWorkflowsArgs, arguments)
    workflows = await wf_service.list_workflows(
        pool, account_id=account_id, limit=args.limit, after=args.after, name=args.name
    )
    return {
        "workflows": [w.model_dump(mode="json", exclude=_WORKFLOW_LIST_EXCLUDE) for w in workflows]
    }


async def get_run_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    args = _parse(_GetRunArgs, arguments)
    run = await wf_service.get_run(pool, args.run_id, account_id=account_id)
    return run.model_dump(mode="json")  # FULL WfRun incl. pinned script


async def list_runs_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    args = _parse(_ListRunsArgs, arguments)
    runs = await wf_service.list_runs(
        pool,
        account_id=account_id,
        limit=args.limit,
        after=args.after,
        workflow_id=args.workflow_id,
        status=args.status,
        parent_run_id=args.parent_run_id,
        # Default: only this session's own runs. account_wide widens to the whole account.
        launcher_session_id=None if args.account_wide else session_id,
    )
    return {"runs": [r.model_dump(mode="json", exclude=_RUN_ECHO_EXCLUDE) for r in runs]}


async def list_run_events_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    args = _parse(_ListRunEventsArgs, arguments)
    # Scope check (mirrors the HTTP /runs/{id}/events route): raise NotFoundError on a
    # missing/cross-account run, so the model sees a real error instead of an empty page
    # indistinguishable from "no events past after_seq".
    await wf_service.get_run(pool, args.run_id, account_id=account_id)
    events = await wf_service.list_run_events(
        pool, args.run_id, account_id=account_id, after_seq=args.after_seq, limit=args.limit
    )
    # payloads returned in full; paging is via after_seq/limit
    return {"events": [e.model_dump(mode="json") for e in events]}


# ─── descriptions + registration ─────────────────────────────────────────────

CREATE_WORKFLOW_DESCRIPTION = (
    "Author a new workflow (a deterministic Python orchestrator) under your account. "
    "Its declared tool/server surface must be a subset of your own — you cannot grant a "
    "workflow a tool, MCP server, or HTTP server you don't yourself have. Returns the "
    "created workflow (id, name, version)."
)
UPDATE_WORKFLOW_DESCRIPTION = (
    "Update one of your workflows in place, bumping its version. Pass the current "
    "'version' as an optimistic-concurrency token (a stale token is rejected — re-read "
    "with get_workflow and retry). Omitted fields are preserved. The resulting tool/server "
    "surface must still be a subset of your own. Archived workflows cannot be updated; "
    "unarchive first. In-flight runs are unaffected (a run pins its workflow's script + "
    "surface at launch)."
)
ARCHIVE_WORKFLOW_DESCRIPTION = (
    "Archive one of your workflows. Archived workflows disappear from list_workflows and "
    "refuse new create_run launches, but remain fetchable by id and keep their run/journal "
    "history. Archiving releases the workflow name for a fresh create_workflow."
)
UNARCHIVE_WORKFLOW_DESCRIPTION = (
    "Restore an archived workflow so it appears in list_workflows and can launch new runs "
    "again. If another live workflow has reclaimed the same name, unarchive is rejected; "
    "rename or archive the conflicting live workflow first."
)
CREATE_RUN_DESCRIPTION = (
    "Launch a run of a workflow you can access. The run executes with the workflow's "
    "declared surface but only the vaults you attach via 'vault_ids' — which must be a "
    "subset of the vaults bound to you. It runs in your own environment. Returns the run "
    "id and status; use await_run to block for its result. The number of runs you may "
    "have outstanding at once is capped — check your outstanding runs with list_runs, and "
    "finish (await_run) or cancel (cancel_run) them to free slots."
)
AWAIT_RUN_DESCRIPTION = (
    "Block until a run reaches a terminal state (completed/errored/cancelled), or until "
    "'timeout_seconds' elapses. Returns {done, run_status, output, is_error, error}; if "
    "not yet done, call again to keep waiting. The session stays responsive while you wait."
)
CANCEL_RUN_DESCRIPTION = (
    "Cancel a run YOU launched (you cannot cancel runs launched by others or by the "
    "operator). The run finalizes 'cancelled' on its next wake — usually within moments "
    "— freeing one of your outstanding-run slots once it does. Idempotent: an "
    "already-finished run is returned unchanged."
)
GET_WORKFLOW_DESCRIPTION = (
    "Fetch one of your workflows in full by id — including its 'script' and current "
    "'version'. Use this to re-read a workflow before retrying an update_workflow whose "
    "optimistic-concurrency token was rejected: read the fresh version, reconcile your "
    "edit, and retry with that version."
)
LIST_WORKFLOWS_DESCRIPTION = (
    "List your account's workflows, newest first, as lean summaries (id, name, "
    "description, version, timestamps) — no script bodies. Optional 'name' filter; "
    "page with 'limit' and 'after' (the last id seen); a full page means there may "
    "be more — call again. To read a workflow's script, fetch it with get_workflow."
)
GET_RUN_DESCRIPTION = (
    "Fetch one workflow run in full by id — including the immutable 'script' the run "
    "pinned at launch, its status, input, and output. Inspect what a specific run "
    "actually executed."
)
LIST_RUNS_DESCRIPTION = (
    "List workflow runs, newest first. By default returns only the runs YOU launched; "
    "set 'account_wide' true to list every run in the account. Optional 'workflow_id' / "
    "'status' / 'parent_run_id' filters; page with 'limit' and 'after' (the last id seen); "
    "a full page means there may be more — call again. Rows are lean (no "
    "script or tool surface) — fetch a single run with get_run for its full body."
)
LIST_RUN_EVENTS_DESCRIPTION = (
    "Page a run's journal in sequence order (oldest first): every event — run_started, "
    "call_started, call_result, run_completed, and the annotation log/phase progress "
    "markers a workflow emits — so you can watch a mid-flight run advance. Page forward "
    "with 'after_seq' (the last seq seen); a full page means call again. Event payloads "
    "are returned in full — the journal text is never clipped."
)


def _register() -> None:
    registry.register(
        name="create_workflow",
        description=CREATE_WORKFLOW_DESCRIPTION,
        parameters_schema=WorkflowCreate.model_json_schema(),
        handler=create_workflow_handler,
        transport="agent_tool",
    )
    registry.register(
        name="update_workflow",
        description=UPDATE_WORKFLOW_DESCRIPTION,
        parameters_schema=_UpdateWorkflowArgs.model_json_schema(),
        handler=update_workflow_handler,
        transport="agent_tool",
    )
    registry.register(
        name="archive_workflow",
        description=ARCHIVE_WORKFLOW_DESCRIPTION,
        parameters_schema=_WorkflowIdArgs.model_json_schema(),
        handler=archive_workflow_handler,
        transport="agent_tool",
    )
    registry.register(
        name="unarchive_workflow",
        description=UNARCHIVE_WORKFLOW_DESCRIPTION,
        parameters_schema=_WorkflowIdArgs.model_json_schema(),
        handler=unarchive_workflow_handler,
        transport="agent_tool",
    )
    registry.register(
        name="create_run",
        description=CREATE_RUN_DESCRIPTION,
        parameters_schema=_CreateRunArgs.model_json_schema(),
        handler=create_run_handler,
        transport="agent_tool",
    )
    registry.register(
        name="await_run",
        description=AWAIT_RUN_DESCRIPTION,
        parameters_schema=_AwaitRunArgs.model_json_schema(),
        handler=await_run_handler,
        transport="agent_tool",
    )
    registry.register(
        name="cancel_run",
        description=CANCEL_RUN_DESCRIPTION,
        parameters_schema=_CancelRunArgs.model_json_schema(),
        handler=cancel_run_handler,
        transport="agent_tool",
    )
    registry.register(
        name="get_workflow",
        description=GET_WORKFLOW_DESCRIPTION,
        parameters_schema=_GetWorkflowArgs.model_json_schema(),
        handler=get_workflow_handler,
        transport="agent_tool",
    )
    registry.register(
        name="list_workflows",
        description=LIST_WORKFLOWS_DESCRIPTION,
        parameters_schema=_ListWorkflowsArgs.model_json_schema(),
        handler=list_workflows_handler,
        transport="agent_tool",
    )
    registry.register(
        name="get_run",
        description=GET_RUN_DESCRIPTION,
        parameters_schema=_GetRunArgs.model_json_schema(),
        handler=get_run_handler,
        transport="agent_tool",
    )
    registry.register(
        name="list_runs",
        description=LIST_RUNS_DESCRIPTION,
        parameters_schema=_ListRunsArgs.model_json_schema(),
        handler=list_runs_handler,
        transport="agent_tool",
    )
    registry.register(
        name="list_run_events",
        description=LIST_RUN_EVENTS_DESCRIPTION,
        parameters_schema=_ListRunEventsArgs.model_json_schema(),
        handler=list_run_events_handler,
        transport="agent_tool",
    )


_register()
