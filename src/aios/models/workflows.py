"""Workflows: read models for the durable runtime core (Block 1).

A *workflow* is a deterministic-Python orchestrator (the dual of an agent):
``workflows`` are versioned definitions (updated in place, agent-style); a ``WfRun``
is a durable execution instance whose state lives entirely in its append-only journal
(``WfRunEvent``) and which pins its workflow's script + declared surface at launch;
``WfRunSignal`` is the side-marker an external resume writes so the journal keeps a
single writer.

The read views below carry ``account_id`` (internal); the ``*Create`` / resume
request models at the bottom back the public HTTP surface (Block 3). Responses
reuse the read views directly, the way ``Agent``/``Session`` do.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aios.actors import Actor
from aios.models.accounting import AttributedUsage, UsageNodeRef
from aios.models.agents import (
    HttpServerRef,
    HttpServerSpec,
    McpServerSpec,
    SshServerRef,
    SshServerSpec,
    ToolSpec,
    validate_http_servers,
    validate_mcp_servers,
    validate_ssh_servers,
    validate_tools,
)

WorkspaceMode = Literal["shared", "fresh"]

RunPrincipal = Literal["operator", "session", "agent"]  # see ``WfRun.principal``
RunVisibility = Literal["account", "session"]  # see ``WfRun.visibility``


class AsAgent(BaseModel):
    """An agent version whose surface an operator re-rooted a sub-run to
    (``invoke_workflow``'s ``as_agent``), so an eval arm runs with the authority that
    agent would give it."""

    model_config = ConfigDict(extra="forbid")

    agent_id: str
    version: int = Field(ge=1, lt=2**31)


class RequestRef(BaseModel):
    """A reference to a request a session sent: the span that captured it (#2471),
    named the way ``GET /v1/sessions/{session_id}/requests/{request_id}`` names it.

    A ref means something only in a field typed as one. A run resolves only the ref
    it was created with (``WfRun.request_ref``), never a ref-shaped value it finds
    in its input or in a tool result."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    request_id: str


@dataclass(frozen=True, slots=True)
class OperatorAuthority:
    """A run launched through the operator API. Its workflow's surface binds as-is."""


@dataclass(frozen=True, slots=True)
class SessionAuthority:
    """A run an agent session launched: ``call_workflow``, a workflow-as-model turn, or
    a trigger fire. The session's current surface and vaults bound it.
    ``parent_run_id`` is lineage only: it sets the depth budget and the spend ceiling the
    run inherits (#2476), never the authority."""

    session_id: str
    parent_run_id: str | None


@dataclass(frozen=True, slots=True)
class RunAuthority:
    """A sub-run a run launched with ``invoke_workflow``. The parent run's frozen surface
    and vaults bound it, and the parent is also its lineage parent.

    ``as_agent`` (operator runs only) re-roots the surface bound at that agent version:
    the bound becomes ``clamp(surface_of(agent_version), parent surface)``. The model
    and vaults still come from the parent."""

    run_id: str
    as_agent: AsAgent | None = None


# Who a new run acts under (``create_run``). It decides the run's principal, the
# surface its snapshot is clamped to, the vaults it may bind and its lineage.
type RunAuthoritySource = OperatorAuthority | SessionAuthority | RunAuthority

WfRunStatus = Literal["pending", "running", "suspended", "completed", "errored", "cancelled"]
WfRunEventType = Literal[
    "run_started",
    "call_started",
    "call_result",
    "run_completed",
    "annotation",
    "frontier_deferred",
]
WfRunSignalKind = Literal["gate_resume", "child_done", "cancel", "tool_result"]

# The terminal run statuses — monotonic: once here, a run never leaves. The one source for
# every "is this run done?" check (the step loop's early-out, the SSE stream's close, the await
# predicate). ``cancelled`` is terminal too (a user cancel finalizes the run).
TERMINAL_RUN_STATUSES: frozenset[str] = frozenset({"completed", "errored", "cancelled"})


class Workflow(BaseModel):
    """A versioned workflow definition (updated in place; ``version`` bumps per change)."""

    id: str
    account_id: str
    name: str
    version: int
    script: str
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    # The declared effective model for a ``workflow:`` model binding (#1637): the raw
    # provider model this workflow ultimately emits (e.g. ``anthropic/claude-opus-4-6``).
    # When an agent's ``model`` is ``workflow:<this-id>``, the capability gates (vision /
    # extended-thinking continuity / token-window calibration) key on THIS string instead
    # of the opaque ``workflow:`` string — otherwise a bound model silently degrades
    # (images dropped, thinking-blocks stripped, token counting under-counts). ``None``
    # keeps the pre-#1637 degraded posture (raw ``workflow:`` string drives the gates).
    output_model: str | None = None
    description: str | None = None  # optional human blurb (the agent ``description`` analog)
    # The declared tool surface — the verbatim agent envelope. A run reaches these
    # (authed MCP / http_request / builtins) directly via ``tool()`` (a later slice);
    # an agent authoring a workflow may only declare a subset of its own surface.
    tools: list[ToolSpec] = Field(default_factory=list)
    mcp_servers: list[McpServerSpec] = Field(default_factory=list)
    http_servers: list[HttpServerSpec] = Field(default_factory=list)
    ssh_servers: list[SshServerSpec] = Field(default_factory=list)
    created_by: Actor | None = None
    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None = None


class WorkflowVersion(BaseModel):
    """Read view of a specific workflow version from the immutable history.

    The workflow analog of :class:`aios.models.agents.AgentVersion`: a complete,
    immutable snapshot of a workflow's definition at one ``version``. ``name`` IS
    versioned — a rename mints a new version — so this carries it alongside the
    script + declared surface. Snapshots exactly ``update_workflow``'s no-op
    comparison set (``name, script, input_schema, output_schema, description,
    tools, mcp_servers, http_servers``).
    """

    workflow_id: str
    version: int
    name: str
    script: str
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    output_model: str | None = None  # declared effective model for the binding (#1637)
    description: str | None = None
    tools: list[ToolSpec] = Field(default_factory=list)
    mcp_servers: list[McpServerSpec] = Field(default_factory=list)
    http_servers: list[HttpServerSpec] = Field(default_factory=list)
    ssh_servers: list[SshServerSpec] = Field(default_factory=list)
    created_at: datetime


class WfRunUsage(AttributedUsage):
    """Per-run cost / token / iteration / wall-clock — the machine-observer's substrate.

    The read-path projection of a run's actual spend (#1324). The numbers are summed
    over the run's direct child sessions via ``run_children_usage`` — the SAME source
    ``step.py``'s ``budget()`` builtin consumes — so a run's ``budget_usd`` *ceiling*
    (on ``WfRun``) and its realized ``cost_microusd`` *spend* (here) are finally both
    legible from the read path.

    Every legacy flat metric is ``int | None``, and absence is reported as
    **explicit null**, never a silent ``0`` or an omitted key (cf. the
    ``vault_ids:null`` read-path disease this must not inherit — see the
    substrate-different-verdict invariant). The observer reads null as
    *cannot-determine* and fails loud, NOT as "zero spend". Creation accounting
    is carried separately by the inherited ``own``/``subtree`` and rate fields:

    * ``cost_microusd`` / ``*_tokens`` — summed over the run's child sessions. A run
      with no children sums to ``0`` (a real, observed zero — distinct from null).
    * ``iteration_count`` — the run's wake/step count. The host keeps **no** per-run
      iteration counter on any substrate today, so this is reported as ``None``
      (cannot-determine) rather than fabricated from an unrelated proxy. Reserved for
      when a real counter lands; surfaced now so the observer's contract is stable.
    * ``wall_clock_ms`` — wall-clock span ``updated_at - created_at`` in milliseconds,
      reported ONLY for a TERMINAL run (``updated_at`` is its completion instant). A
      still-running run's ``updated_at`` is a moving "last touched" stamp, not an end,
      so it is reported as ``None`` rather than a misleading partial span.
    """

    cost_microusd: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    iteration_count: int | None = None
    wall_clock_ms: int | None = None


class WfRun(BaseModel):
    """A durable workflow execution instance.

    ``script`` is the run's own immutable snapshot of the workflow source at
    creation time (``script_sha`` is its hash); every wake execs exactly this.
    ``tools``/``mcp_servers``/``http_servers`` are the matching snapshot of the
    declared tool surface — pinned at launch like ``script``, so a later
    ``update_workflow`` never shifts an in-flight run's authority.
    ``status`` is persisted (unlike sessions): the run loop writes
    ``suspended``/``completed``/``errored``.

    NB (#1140): the run's lifecycle field is ``status`` — there is no ``state``
    field on a run. A watcher polling ``.state`` reads ``None`` forever even
    though ``output`` is already populated; poll ``status`` (terminal values:
    ``completed``/``errored``/``cancelled``).
    """

    id: str
    # The source definition this run snapshotted, or ``None`` for an INLINE run
    # (T5, #1466): a one-shot run launched directly from an inline ``{script,
    # schemas, surface}`` body with NO ``workflows`` row created. A run has always
    # pinned its own ``script`` snapshot at launch — the workflow row was merely the
    # source — so an inline run simply has no source row to point at.
    workflow_id: str | None = None
    account_id: str
    environment_id: str  # the run binds to an environment; agent() children inherit it
    workspace: WorkspaceMode = "fresh"
    workspace_path: str | None = None
    # Lineage + the vertical depth cap's walk key. Set by nested workflow()
    # launches AND by trigger fires (#819): a run_completion fire threads the
    # completing run's id, a timer fire threads the owner session's own parent
    # run — so reactive cascades and self-fire loops are depth-bounded.
    parent_run_id: str | None = None
    # The agent session that launched this run: lineage, plus the per-launcher fan-out
    # cap's count key. None for an operator run, and also once the launching session is
    # deleted (the FK is ON DELETE SET NULL), so authority checks read ``principal``.
    launcher_session_id: str | None = None
    principal: RunPrincipal = Field(
        description=(
            "Who the run acts for: `operator` (launched through the operator API, or a "
            "sub-run of an operator run), `session` (launched by an agent session, "
            "directly or through its runs) or `agent` (an operator run's sub-run invoked "
            "with `as_agent`, or a sub-run of one: it acts with that agent's authority, "
            "so operator-only capabilities are refused). Fixed at creation; deleting the "
            "launching session doesn't change it."
        )
    )
    as_agent: AsAgent | None = Field(
        default=None,
        description=(
            "Set when an operator run invoked this run with `as_agent`: the agent "
            "version whose surface it was clamped to, within the parent run's. The model "
            "and vaults still come from the parent run. The run and its sub-runs act for "
            "that agent (principal `agent`), so operator-only capabilities are refused."
        ),
    )
    request_ref: RequestRef | None = Field(
        default=None,
        description=(
            "The request this run works on, when it was created with one: a "
            "workflow-as-model turn's own request, or one handed to a sub-run. The run "
            "can resolve this request and no other. Such a run's `input` is null: the "
            "script's input (the request plus this ref) is the `run_started` event in its "
            "journal, which is pruned a day after the run ends. A sub-run handed a ref "
            "rebuilds the request on its first step."
        ),
    )
    visibility: RunVisibility = Field(
        description=(
            "Who may read the run through agent tools: `account` (any session in the "
            "account) or `session` (only its launching session). A workflow-as-model "
            "run is `session`, and so is a run that declares a replay tool "
            "(`sample_requests`, `get_request`); it has no launching session, so no "
            "agent can read it. Sub-runs inherit their parent's. The operator API "
            "reads every run."
        )
    )
    # The DOWN-counting trusted invoke-depth (#1124): the budget remaining for
    # this run's OUTGOING trusted edges (run→run sub-launches, run→session ``agent()``
    # children). An edgeless root seeds at the full budget; a nested launch carries
    # ``parent.depth - 1``. The decrement IS the cycle bound — a run at depth 0 may
    # open no further trusted edges.
    depth: int = 0
    # The run's INBOUND request edge (#1126/#1129): set when the run was spawned
    # *in service of a request* (a parent run's ``invoke_workflow``). ``request_id``
    # is what the terminal ``request_response`` is keyed on; ``caller`` is the
    # kind-agnostic provenance ({kind:'run'|'session'|'api', id});
    # ``request_output_schema`` is the JSON Schema the request demands of this run's
    # terminal output (validated fail-loud at completion). All None = an edgeless
    # operator/HTTP run, which answers no request and emits no ``request_response``.
    request_id: str | None = None
    caller: dict[str, Any] | None = None
    request_output_schema: dict[str, Any] | None = None
    script: str
    script_sha: str
    # The workflow version this run snapshotted its script + declared surface from
    # (#1321). NULL on legacy rows that predate the column (and on rows the
    # best-effort backfill left ambiguous). The run-side analog of
    # ``Session.agent_version``; bound by a strict composite FK to
    # ``workflow_versions`` so a non-NULL pointer always resolves. Purely
    # audit/integrity today — the run still execs its inline ``script`` snapshot
    # (reading the script *through* this FK is the deferred Phase 3).
    source_version: int | None = None
    host_semantics_epoch: int
    tools: list[ToolSpec] = Field(default_factory=list)
    mcp_servers: list[McpServerSpec] = Field(default_factory=list)
    http_servers: list[HttpServerSpec] = Field(default_factory=list)
    ssh_servers: list[SshServerSpec] = Field(default_factory=list)
    status: WfRunStatus = Field(
        description=(
            "The run's lifecycle status — the ONLY lifecycle field on a run "
            "(there is no `state` field; a watcher keying on `.state` waits "
            "forever). Terminal values: `completed`/`errored`/`cancelled`."
        )
    )
    input: Any = None  # arbitrary JSON: a workflow's input need not be an object
    output: Any = None  # arbitrary JSON: the script's return value
    budget_usd: float | None = None
    # The nearest ancestor run with a ``budget_usd`` (#2476). A run with no budget of
    # its own is held to that run's ceiling; ``None`` when no ancestor has one.
    budget_run_id: str | None = None
    default_child_model: str | None = None
    # The run's own ``call_llm`` inference spend (#1633), in micro-USD. Raw inference
    # the run runs on the worker (``call_llm()``) has no child-session row, so it lives
    # in this run-level meter; the ``budget_usd`` gate is the SUM of this and the
    # child-session rollup (``usage.cost_microusd``). Charged once at the inference site.
    call_llm_cost_microusd: int = 0
    # False on pre-0168 runs unless an explicit bounded repair later proves the
    # historical journals complete. Never render unavailable tokens as measured zero.
    call_llm_tokens_complete: bool = True
    last_event_seq: int
    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None = None
    terminal_summary: dict[str, Any] | None = None
    journal_pruned_at: datetime | None = None
    # The realized per-run usage (#1324) — cost/tokens summed over child sessions,
    # plus iteration/wall-clock. Populated ONLY on the public read path (get_run /
    # list_runs); ``None`` on the internal step-loop read (``get_run_for_step``),
    # which never needs it and must not pay the extra aggregate query. ``budget_usd``
    # above is the ceiling; ``usage.cost_microusd`` is the spend against it.
    usage: WfRunUsage | None = None
    usage_parent: UsageNodeRef | None = None


class RunReader(NamedTuple):
    """An agent-side read of runs, acting for ``session_id``: an agent tool reads as its
    own session, and a run-side tool as its run's launching session (``None`` when the
    run has none). The operator API reads with no ``RunReader`` at all."""

    session_id: str | None

    def can_see(self, run: WfRun) -> bool:
        """Mirrored in SQL by ``wf_queries.list_wf_runs``'s ``reader`` filter."""
        if run.visibility == "account":
            return True
        return self.session_id is not None and run.launcher_session_id == self.session_id


class WfRunEvent(BaseModel):
    """One row of a run's append-only journal (the replay-with-memo source).

    ``call_key`` is set for ``call_started``/``call_result`` (the memo key) and for
    ``annotation`` (the branch-local dedup key that makes ``log()``/``phase()``
    emit-once across replays); it is ``None`` for the ``run_started``/``run_completed``
    bookends. An ``annotation`` is a journaled progress marker (``payload`` =
    ``{"kind": "log" | "phase", "text": ...}``), not a capability call.

    Schema note (#1140): a *run* event is ``{type, payload, seq}``. This is a
    DIFFERENT shape from a child-*session* event (``{kind, data}`` — see
    ``aios.models.events.Event``). Don't assume one schema across both
    endpoints; ``docs/reference/run-observability.md`` documents the split.
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
    result: Any = None  # arbitrary JSON: the externally-delivered resume value
    delivered_at: datetime


# ─── request models (the public HTTP surface) ────────────────────────────────

WORKFLOW_SCRIPT_CONTRACT = """Workflow script contract:
- Entry point: define `async def main(input)`. A run's output is the value returned by
  `main`.
- Injected capability API, available without imports:
  - `agent(input, *, agent_id=None, output_schema=None, model=None, label=None)`: invoke a generic or named agent and await its result.
  - `invoke_workflow(workflow_id, input, *, version=None, output_schema=None, label=None, as_agent=None, request_ref=None, budget_usd=None)`: invoke another workflow as a sub-run and await its result (the run dual of `agent`). `version` pins a registered version; omitted, the version current at launch runs. The sub-run runs under this run's surface intersected with the target's, and binds this run's vaults; a failed or gone sub-run raises like a failed `agent`. `as_agent={"agent_id": ..., "version": N}` also intersects the sub-run's surface with that agent version's, and the sub-run and its own sub-runs then act for that agent, so operator-only capabilities (the replay tools, `workflow:` child models, `as_agent`) are refused there; only a run that acts for the operator may pass it. `request_ref` hands the sub-run a request this run can resolve (the one it was created with, or one its `sample_requests` call returned), which the sub-run may then resolve itself; with `input=None` it starts with that request as a workflow-as-model run would, with an `input` it starts with that input. `budget_usd` gives the sub-run a ceiling of its own, clamped to what this run may still spend; without one it is held to this run's.
  - `tool(name, input)`: invoke a declared tool; tool errors are returned, not raised. An operator run that declares the replay tools can read an agent's past requests: `tool("sample_requests", {"agent_id", "start", "end", "n", "seed", "cluster_cap"})` returns `{"items": [...]}`, a seeded sample of the agent's answered requests in `[start, end)` (ISO-8601 times), each with a `request_ref`, its session, time, model, `response_event_id` and whether it is `missing`. `tool("get_request", {"request_ref", "model"})` rebuilds one as `{messages, tools, params, fidelity}`. A ref a sample returned can be passed to `call_llm(request_ref=)`, `invoke_workflow(request_ref=)` and `get_request`.
  - `call_llm(request)`: run one raw inference turn and await the assistant turn. `request` carries `model` (omit to use the run's default child model; a `workflow:` target is rejected), `messages` (required), optional `tools` (schemas OFFERED — the model may request a call, but call_llm never runs it), and optional `params` (provider knobs). The result is `{"content", "tool_calls", "finish_reason", "usage", "cost", "message"}`, or `{"error": ...}` — a model error is returned, not raised. Its cost is metered against this run's `budget_usd` ceiling, so a budget-exhausted run refuses further `call_llm`. Use it to route/judge/fact-check around inference; use `agent(...)` when you want the tool calls executed. `call_llm(request_ref=ref, model=None)` sends a request this run can resolve instead (the one it was created with, or one its `sample_requests` call returned), rendered for `model`; with the model it was captured for it keeps that request's provider params; a request captured for a `workflow:` binding keeps them for any model, with an endpoint admitted only from the allowlist; otherwise another model gets none. Any other ref resolves as an error value.
  - `gate()`: suspend until an external resume delivers a value.
  - `budget()`: read the budget this run is held to, `{total_usd, spent_usd, remaining_usd}` over the budget run's whole subtree: its own `budget_usd`, else its nearest budgeted ancestor's; None when there is none. Once it is spent, new `agent()`, `call_llm` and `invoke_workflow` calls are refused.
  - `sub_runs()`: read facts about every session and run this run created, directly or through descendants: `{"nodes": [...], "truncated": bool, "litellm_version": str}`, each node with its kind, label, workflow or agent and the version that ran, status, timings, and usage per model (tokens, `cost_microusd` as charged, and `uncached_cost_microusd`, the same tokens priced with no prompt-cache discount, or null for a model outside the cost map). Metadata only.
  - `parallel(thunks)`: run zero-argument callables concurrently (for example,
    `lambda: agent(...)`). A failed agent branch yields `None` at the barrier instead
    of raising. Fan-out width is capped by `MAX_PARALLEL_FANOUT` (currently 1000).
  - `pipeline(items, *stages)`: run each item through staged transforms concurrently.
  - `log(msg)`: record progress on the run journal.
  - `phase(label)`: record a phase marker on the run journal.
- Shell execution: `tool('bash', {"command": str, "timeout_seconds": float | None})` runs the
  command in a per-run sandbox (provisioned lazily on first use, in the run's
  environment). `bash` must be a member of the workflow's declared tools or the call
  resolves to a `{"error": ...}` value. Result: `{exit_code, stdout, stderr, timed_out,
  truncated}` — a nonzero exit or in-command timeout is a successful result to branch
  on, not an error.
- Crash semantics: at-least-once at the call boundary. A capability call interrupted by
  a crash re-runs on resume; completed calls never re-run. The sandbox filesystem is
  ephemeral scratch — write re-run-tolerant commands (e.g. `rm -rf dir && git clone ...`).
- Irreversible external effects (a POST that charges, sends, or publishes): a per-call
  idempotency token (stable across crash re-runs of the same call, distinct per call) is
  available so you can have the external service drop a re-fired duplicate, or knowingly
  accept at-least-once. Both deliveries opt in the same way with the same `$AIOS_IDEMPOTENCY_KEY`
  ergonomic: in `tool('bash')` the environment exposes `$AIOS_IDEMPOTENCY_KEY`; in
  `tool('http_request')` pass the sentinel string `"$AIOS_IDEMPOTENCY_KEY"` as an
  `Idempotency-Key` header value and the worker substitutes the real token before dispatch.
- Partition rule: put re-run-tolerant mechanical work in `tool('bash')`; put work whose
  uncertain completion needs judgment to resolve inside `agent(...)`.
- Environment: the SCRIPT runs in a deterministic, credential-free, isolated child
  process — imports restricted to a curated stdlib allowlist, no network or filesystem
  access from script code itself; all effects go through the capability API. The
  `tool('bash')` sandbox is different: it has a filesystem (ephemeral scratch) and
  network egress per the run's ENVIRONMENT network policy (Unrestricted, or Limited to
  the environment's allowed hosts) — commands can curl, clone, and install within that
  policy.

Minimal example:
```python
async def main(input):
    result = await agent(
        {"task": input["task"]},
        agent_id=input["agent_id"],
    )
    return result
```
"""


class WorkflowCreate(BaseModel):
    """Request body for ``POST /v1/workflows`` — a new workflow definition at v1."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=128)
    script: str = Field(description=WORKFLOW_SCRIPT_CONTRACT)
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    # The declared effective model for a ``workflow:`` model binding (#1637) — the raw
    # provider model this workflow emits, so the capability gates resolve to it instead
    # of the opaque ``workflow:`` string. See :class:`Workflow.output_model`.
    output_model: str | None = Field(default=None, min_length=1)
    description: str | None = None
    # The declared tool surface (verbatim agent envelope). When an agent authors a
    # workflow, these must be a subset of the creating agent's own surface; the HTTP
    # path is unattenuated operator authority.
    tools: list[ToolSpec] = Field(default_factory=list)
    mcp_servers: list[McpServerSpec] = Field(default_factory=list)
    # ``http_servers`` accepts either a full ``HttpServerSpec`` (identity-match, #949)
    # or a bare name string (names-only sugar, #953) resolved against the acting
    # agent at the authoring edge. The HTTP/operator path has no acting agent, so a
    # bare name there is rejected by the service (nothing to resolve against).
    http_servers: list[HttpServerRef] = Field(default_factory=list)
    ssh_servers: list[SshServerRef] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_http_servers(self) -> WorkflowCreate:
        # Cross-item base_url uniqueness applies to full specs only; bare names
        # carry no base_url until resolved against the agent (validated there).
        validate_http_servers([s for s in self.http_servers if isinstance(s, HttpServerSpec)])
        validate_ssh_servers([s for s in self.ssh_servers if isinstance(s, SshServerSpec)])
        validate_mcp_servers(self.mcp_servers)
        validate_tools(self.tools)
        return self


class WorkflowUpdate(BaseModel):
    """Request body for ``PUT /v1/workflows/{id}`` — update in place, bumping ``version``.

    ``version`` is the optimistic-concurrency token: it must match the workflow's
    current version or the update 409s (re-fetch and retry). Omitted fields are
    preserved — nullable fields (``input_schema``/``output_schema``/``description``)
    can therefore be replaced but never cleared back to null, as on ``AgentUpdate``.
    An identical update is a no-op (no bump). Each real bump is snapshotted into the
    immutable ``workflow_versions`` history (see :class:`WorkflowVersion`) in the same
    transaction, copy-on-write like ``agent_versions``. A run additionally pins
    ``script`` + the declared surface onto itself at launch, so in-flight runs never
    observe an update. (The ``AgentUpdate`` shape.)
    """

    model_config = ConfigDict(extra="forbid")

    version: int
    name: str | None = Field(default=None, min_length=1, max_length=128)
    script: str | None = Field(default=None, description=WORKFLOW_SCRIPT_CONTRACT)
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    # ``None`` preserves the current declared effective model (cannot be cleared back to
    # null, as on the other nullable fields). See :class:`Workflow.output_model` (#1637).
    output_model: str | None = Field(default=None, min_length=1)
    description: str | None = None
    tools: list[ToolSpec] | None = None
    mcp_servers: list[McpServerSpec] | None = None
    # See ``WorkflowCreate.http_servers`` — bare names (names-only sugar, #953) or
    # full ``HttpServerSpec`` (identity-match, #949); ``None`` preserves current.
    http_servers: list[HttpServerRef] | None = None
    ssh_servers: list[SshServerRef] | None = None

    @model_validator(mode="after")
    def _validate_http_servers(self) -> WorkflowUpdate:
        if self.http_servers is not None:
            validate_http_servers([s for s in self.http_servers if isinstance(s, HttpServerSpec)])
        if self.ssh_servers is not None:
            validate_ssh_servers([s for s in self.ssh_servers if isinstance(s, SshServerSpec)])
        if self.mcp_servers is not None:
            validate_mcp_servers(self.mcp_servers)
        if self.tools is not None:
            validate_tools(self.tools)
        return self


class InlineScriptBody(BaseModel):
    """The inline-script body of an anonymous run launch (T5, #1466).

    The alternative to ``WfRunCreate.workflow_id``: a one-shot run launched directly
    from this ``{script, schemas, surface}`` body, with NO ``workflows`` row created.
    The run snapshots ``script`` exactly as it snapshots a registered workflow's, and
    the declared surface (``tools``/``mcp_servers``/``http_servers``) is clamped to the
    launcher with the same create-time clamp ``create_workflow`` uses (a surface
    exceeding the launcher raises ``ForbiddenError``). The HTTP/operator path is
    unattenuated operator authority; names-only http sugar is rejected there (no acting
    agent to resolve a bare name against).
    """

    model_config = ConfigDict(extra="forbid")

    script: str = Field(description=WORKFLOW_SCRIPT_CONTRACT)
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    tools: list[ToolSpec] = Field(default_factory=list)
    mcp_servers: list[McpServerSpec] = Field(default_factory=list)
    http_servers: list[HttpServerRef] = Field(default_factory=list)
    ssh_servers: list[SshServerRef] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_http_servers(self) -> InlineScriptBody:
        validate_http_servers([s for s in self.http_servers if isinstance(s, HttpServerSpec)])
        validate_ssh_servers([s for s in self.ssh_servers if isinstance(s, SshServerSpec)])
        validate_mcp_servers(self.mcp_servers)
        validate_tools(self.tools)
        return self


class WfRunCreate(BaseModel):
    """Request body for ``POST /v1/runs`` — launch a run.

    Exactly ONE source arm (validated below):

    * ``workflow_id`` (+ optional ``version``) — the registered path: snapshot a
      pre-registered workflow's script + declared surface.
    * ``inline`` (:class:`InlineScriptBody`) — the inline-script arm (T5, #1466): a
      one-shot run launched from an inline ``{script, schemas, surface}`` body with NO
      ``workflows`` row created. ``version`` is meaningless on this arm (no definition
      history) and is rejected if combined with it.

    ``input`` is arbitrary JSON (a workflow's input need not be an object). The run
    binds to ``environment_id`` (like a session), into which its ``agent()`` children
    spawn. (``launcher_session_id`` is deliberately NOT a field — trusted ids never
    ride in request bodies; the HTTP path is always an operator launch.)
    """

    model_config = ConfigDict(extra="forbid")

    workflow_id: str | None = Field(
        default=None,
        description=(
            "The registered workflow to run. Supply EITHER this or `inline` (exactly one). "
            "Omit when launching an inline one-shot run."
        ),
    )
    inline: InlineScriptBody | None = Field(
        default=None,
        description=(
            "Inline-script body for an anonymous one-shot run (T5). Supply EITHER this or "
            "`workflow_id` (exactly one). No `workflows` row is created."
        ),
    )
    environment_id: str
    workspace: WorkspaceMode = Field(
        default="fresh",
        description="Workspace mode. HTTP launches default to a fresh run workspace.",
    )
    version: int | None = Field(
        default=None,
        ge=1,
        lt=2**31,  # workflow_versions.version is int4: a larger value is a 422, not a 500
        description=(
            "Optional historical workflow version to run. `None` (default) launches the "
            "workflow's CURRENT version. An integer re-runs that specific version: the run "
            "snapshots that version's script + declared surface (clamped against the current "
            "launcher's authority) and binds `source_version` to it. Launching ANY version "
            "of an archived workflow is refused (409). This is a SELECTOR — distinct from the "
            "trigger's `workflow_version` drift assertion. Not valid with `inline`."
        ),
    )
    input: Any = None
    vault_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Vault ids to bind to the run for credential resolution. When an agent "
            "launches the run, these must be a subset of the launcher's own vaults; "
            "the HTTP path is unattenuated operator authority."
        ),
    )
    budget_usd: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Optional USD spend ceiling for this run's whole creation subtree (its "
            "agent() children, their sub-agents, and sub-runs). Once the subtree has "
            "spent it, the run and every descendant run without a budget of its own "
            "are refused new agent(), call_llm and invoke_workflow calls. A sub-run "
            "given its own budget is held to that budget, set at launch to at most what "
            "remained, so the subtree can overshoot by up to the budgets of sub-runs "
            "still in flight."
        ),
    )
    default_child_model: str | None = Field(
        default=None,
        description="Optional model used by generic agent() children when they omit model=.",
    )

    @model_validator(mode="after")
    def _validate_source_arm(self) -> WfRunCreate:
        if (self.workflow_id is None) == (self.inline is None):
            got = "both" if self.inline is not None else "neither"
            raise ValueError(f"exactly one of workflow_id or inline must be provided (got {got})")
        if self.inline is not None and self.version is not None:
            raise ValueError("version is not valid for an inline run (no version history)")
        return self


class GateResume(BaseModel):
    """Request body for ``POST /v1/runs/{run_id}/resume`` — deliver a gate's value.

    Keyed by ``gate_nonce`` (the unguessable capability token minted into the gate's
    ``call_started`` event), not the internal ``call_key``. ``result`` is the
    externally-delivered resume value (arbitrary JSON).
    """

    model_config = ConfigDict(extra="forbid")

    gate_nonce: str
    result: Any = None
