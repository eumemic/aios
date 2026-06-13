"""Session resource: a running agent instance + its event log handle.

A session references an agent and an environment, and tracks its current
status (see :data:`SessionStatus`) plus the workspace volume path the
sandbox uses on the host. The harness lease columns and `container_id`
are internal — they live in the DB row but are not exposed on the wire
shape.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aios.models.events import Event
from aios.models.github_repositories import (
    MAX_REPOS_PER_SESSION,
    GithubRepositoryResource,
    GithubRepositoryResourceEcho,
)
from aios.models.github_repositories import validate_resources as _validate_github_resources
from aios.models.memory_stores import (
    MAX_STORES_PER_SESSION,
    MemoryStoreResource,
    MemoryStoreResourceEcho,
)
from aios.models.memory_stores import validate_resources as _validate_memory_resources
from aios.models.triggers import (
    MAX_TRIGGERS_PER_SESSION,
    TriggerCreate,
    TriggerEcho,
    validate_triggers,
)

# Discriminated union over resource types. New types extend this union.
SessionResource = Annotated[
    MemoryStoreResource | GithubRepositoryResource,
    Field(discriminator="type"),
]
SessionResourceEcho = Annotated[
    MemoryStoreResourceEcho | GithubRepositoryResourceEcho,
    Field(discriminator="type"),
]

# Combined per-session cap. Each type still enforces its own narrower cap
# inside ``_validate_session_resources`` below.
MAX_RESOURCES_PER_SESSION = MAX_STORES_PER_SESSION + MAX_REPOS_PER_SESSION


def split_resources_by_type(
    resources: list[SessionResource],
) -> tuple[list[MemoryStoreResource], list[GithubRepositoryResource]]:
    """Partition a heterogeneous resource list by its discriminator."""
    memory: list[MemoryStoreResource] = []
    github: list[GithubRepositoryResource] = []
    for r in resources:
        if isinstance(r, MemoryStoreResource):
            memory.append(r)
        else:
            github.append(r)
    return memory, github


def _validate_session_resources(resources: list[SessionResource]) -> None:
    """Pre-DB cross-item validation; failures surface as 4xx instead of
    constraint violations."""
    memory, github = split_resources_by_type(resources)
    _validate_memory_resources(memory)
    _validate_github_resources(github)


SessionStatus = Literal["active", "idle", "archived"]
"""Derived session activity, computed from the event log + ``archived_at`` at
read time (there is no persisted status column).

* ``active`` — the session has owed/in-flight work that will advance WITHOUT a
  new unprompted user message: an unreacted user/tool stimulus, or an unresolved
  tool_call (a built-in tool running, or one blocked on operator approval / a
  custom-tool result — see :class:`AwaitingToolCall`).
* ``idle`` — quiescent: nothing is owed; only a new unprompted user message (or
  an armed timer firing) will move it. ``idle`` implies ``awaiting == []``.
* ``archived`` — terminal: the session has been soft-archived (``archived_at``
  set), e.g. a workflow ``agent()`` child that reclaimed itself on idle
  (``archive_when_idle``). It will never wake again. This dominates the
  active/idle derivation so a listing can report a run's spent judgment nodes.

An *errored* session (model-call retry budget exhausted, not yet recovered) is a
special case of ``idle``: it reads ``idle`` with ``stop_reason.type == "error"``
and will not wake until a user message arrives.
"""


SessionOrigin = Literal["foreground", "background"]
"""How the session was created. ``foreground`` (default) — by the API/UI/a
connector. ``background`` — spawned by a workflow run's ``agent()`` (carries a
``parent_run_id``); the completion tools (``return``/``error``) are injected only
into background children, and the totality backstop supervises them."""

MAX_USER_MESSAGE_CHARS = 1_000_000


class SessionUsage(BaseModel):
    """Cumulative token usage across all model calls in a session."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


class AwaitingToolCall(BaseModel):
    """One pending tool call the harness will not dispatch itself.

    Derived view on session reads. Each entry is a tool_call on any
    assistant turn (#741) with no paired tool_result and no in-process
    executor:

    * ``kind == "custom"`` — client-executed; awaits POST to
      ``/sessions/:id/tool-results`` (operator-facing) or
      ``/connectors/runtime/tool-results`` (runtime-container-facing).
    * ``kind == "builtin" | "mcp"`` — ``always_ask``-gated and not yet
      confirmed; awaits POST to ``/sessions/:id/tool-confirmations``.
      Confirmed-but-not-yet-dispatched and ``always_allow`` calls don't
      appear here — they're harness-internal.

    ``pending_since`` is the ``created_at`` of the assistant event that
    declared this tool_call (tz-aware UTC). For a ``kind == "custom"``
    call an entry exists from the moment the assistant declares it until
    a result is posted, so a healthy in-flight connector call (e.g.
    ``signal_react``, ~2s) is otherwise indistinguishable from a stuck
    one whose client died. Clients age-threshold custom calls against
    ``pending_since`` (fresh = in-flight, present quietly; stale = stuck,
    alert); builtin/mcp are approval-gated and alert immediately.

    This is the SAME clock the sweep's ``client_tool_call_max_age_seconds``
    abandonment bound (#752) keys off — the assistant turn's
    ``created_at``. The two thresholds are intentionally different
    timescales for different purposes (a cosmetic seconds-scale UI hint
    here vs. an irreversible 24h "client is gone" abandonment there), not
    an inconsistency.
    """

    tool_call_id: str
    name: str
    kind: Literal["builtin", "mcp", "custom"]
    pending_since: datetime


class SessionCreate(BaseModel):
    """Request body for `POST /v1/sessions`."""

    model_config = ConfigDict(extra="forbid")

    agent_id: str
    environment_id: str
    agent_version: int | None = Field(
        default=None,
        description=(
            "Pin to a specific agent version. Omit or pass null for 'latest' "
            "(auto-updating — the session uses whatever version is current)."
        ),
    )
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    vault_ids: list[str] = Field(
        default_factory=list,
        description="Vault ids to bind to this session for MCP credential resolution.",
    )
    archive_when_idle: bool = Field(
        default=False,
        description=(
            "When true, the session is soft-archived the first time it goes idle "
            "owing nothing — a self-reclaiming, one-shot session. Immutable after "
            "launch. Workflow agent() children launch with this set."
        ),
    )

    workspace_path: str | None = Field(
        default=None,
        description=(
            "Absolute host path to use as the session workspace. "
            "If omitted, defaults to workspace_root/<account_id>/<session_id>. "
            "Must resolve within the account's workspace subdirectory. "
            "The directory must exist; aios will not create it."
        ),
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Environment variables injected into the sandbox container. A "
            "vaulted environment_variable credential whose secret_name "
            "matches a key here takes precedence: that key resolves to the "
            "credential's opaque placeholder, not the value set here."
        ),
    )
    initial_message: str | None = Field(
        default=None,
        max_length=MAX_USER_MESSAGE_CHARS,
        description=(
            "Convenience: when set, the server appends a user.message event "
            "with this content immediately after creating the session and "
            "enqueues a wake job. Equivalent to a follow-up POST /messages."
        ),
    )
    resources: list[SessionResource] = Field(
        default_factory=list,
        max_length=MAX_RESOURCES_PER_SESSION,
        description=(
            "Resources to attach. Mix of memory stores (mounted under "
            "/mnt/memory/<name>/) and github repositories (cloned to a "
            "user-specified mount_path). Each type has its own per-session "
            "cap; duplicates within a type are rejected. After creation, "
            "prefer the granular sub-collection endpoints — "
            "``POST /v1/sessions/{id}/resources`` to attach one resource "
            "without touching the others, and "
            "``DELETE /v1/sessions/{id}/resources/{resource_id}`` to detach "
            "one. ``PUT /v1/sessions/{id}`` with ``resources`` replaces the "
            "WHOLE list (omitting a resource detaches it), so the granular "
            "endpoints are the safe path for add/remove (#270)."
        ),
    )
    triggers: list[TriggerCreate] = Field(
        default_factory=list,
        max_length=MAX_TRIGGERS_PER_SESSION,
        description=(
            "Triggers attached at session creation. Each pairs a ``source`` "
            "(cron / one_shot) with an ``action`` (a ``sandbox_command`` bash "
            "task that fires without waking the model, or a ``wake_owner`` "
            "message that wakes it). Manage after creation via "
            "``POST/DELETE/PUT /v1/sessions/{id}/triggers``; ``SessionUpdate`` "
            "deliberately does not accept this field (granular ops only)."
        ),
    )

    @field_validator("workspace_path")
    @classmethod
    def _validate_workspace_path(cls, v: str | None) -> str | None:
        if v is not None and not Path(v).is_absolute():
            raise ValueError("workspace_path must be an absolute path")
        return v

    @model_validator(mode="after")
    def _validate_resources(self) -> SessionCreate:
        _validate_session_resources(self.resources)
        return self

    @model_validator(mode="after")
    def _validate_triggers_list(self) -> SessionCreate:
        validate_triggers(self.triggers)
        return self


class SessionUpdate(BaseModel):
    """Request body for ``PUT /v1/sessions/{id}``.

    All fields are optional; omitted fields are preserved. Changing
    ``agent_id`` resets ``agent_version`` to null (latest) unless
    ``agent_version`` is also provided. ``resources`` and ``vault_ids``
    use full-list-replacement semantics: ``None`` (the default) leaves
    the current set alone, ``[]`` detaches everything, and a non-empty
    list replaces the bound set entirely.

    To add or remove a SINGLE resource without re-supplying the rest of
    the list, use the granular sub-collection endpoints —
    ``POST /v1/sessions/{id}/resources`` (attach one) and
    ``DELETE /v1/sessions/{id}/resources/{resource_id}`` (detach one).
    A one-resource ``resources`` list here silently detaches everything
    else; the granular endpoints are the safe add/remove path (#270).
    """

    model_config = ConfigDict(extra="forbid")

    agent_id: str | None = None
    agent_version: int | None = None
    title: str | None = None
    metadata: dict[str, Any] | None = None
    vault_ids: list[str] | None = None
    resources: list[SessionResource] | None = None

    @model_validator(mode="after")
    def _validate_resources(self) -> SessionUpdate:
        if self.resources is not None:
            _validate_session_resources(self.resources)
        return self


class Session(BaseModel):
    """Read view of a session. Internal-only columns are not exposed.

    ``status`` ({active, idle}) is derived per read from the event log; see
    :data:`SessionStatus`. ``stop_reason`` records why the most recent step
    ended. Possible ``type`` values: ``"end_turn"``, ``"interrupt"``,
    ``"rescheduling"``, ``"error"`` (``idle`` + ``error`` = the errored
    landing pad). ``awaiting`` lists tool calls the session is blocked on
    (derived per read from the event log + agent tool specs).
    """

    id: str
    agent_id: str | None
    environment_id: str
    agent_version: int | None
    model: str | None = None
    title: str | None
    metadata: dict[str, Any]
    status: SessionStatus
    stop_reason: dict[str, Any] | None
    awaiting: list[AwaitingToolCall] = Field(default_factory=list)
    vault_ids: list[str] = Field(default_factory=list)
    last_event_seq: int
    usage: SessionUsage = Field(default_factory=SessionUsage)
    resources: list[SessionResourceEcho] = Field(default_factory=list)
    triggers: list[TriggerEcho] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None = None
    focal_channel: str | None = None
    focal_locked: bool = False
    origin: SessionOrigin = "foreground"
    parent_run_id: str | None = None  # set for a workflow-spawned (background) child
    archive_when_idle: bool = False  # self-archive on first idle (immutable launch property)
    last_event_at: datetime | None = None
    total_events: int = 0


class SessionCloneRequest(BaseModel):
    """Request body for ``POST /v1/sessions/{id}/clone``.

    All fields optional; the clone inherits everything not overridden from
    the parent at clone time.
    """

    model_config = ConfigDict(extra="forbid")

    workspace_path: str | None = Field(
        default=None,
        description=(
            "Override the clone's workspace volume path. Defaults to a fresh "
            "``workspace_root/<account_id>/<new_session_id>`` so clones don't "
            "fight over files. Must resolve within the account's workspace "
            "subdirectory. The directory must exist; aios will not create it."
        ),
    )

    @field_validator("workspace_path")
    @classmethod
    def _validate_workspace_path(cls, v: str | None) -> str | None:
        if v is not None and not Path(v).is_absolute():
            raise ValueError("workspace_path must be an absolute path")
        return v


class SessionUserMessage(BaseModel):
    """Request body for `POST /v1/sessions/{id}/messages`."""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionInterruptRequest(BaseModel):
    """Request body for `POST /v1/sessions/{id}/interrupt`."""

    model_config = ConfigDict(extra="forbid")

    reason: str | None = None


class ToolResultRequest(BaseModel):
    """Request body for ``POST /v1/sessions/{id}/tool-results``.

    ``content`` accepts either a plain string OR a multimodal content
    array shaped per the OpenAI chat-completions tool-result format
    (e.g. ``[{"type": "text", "text": "..."}, {"type": "image_url",
    "image_url": {"url": "..."}}]``).  Built-in tools have always
    produced multimodal results; this widening lets external clients
    (#301 — connectors as HTTP clients) do the same when posting
    custom-tool results.
    """

    model_config = ConfigDict(extra="forbid")

    tool_call_id: str = Field(description="The tool_call_id from the assistant's tool_calls.")
    content: str | list[dict[str, Any]] = Field(
        description="The result of executing the tool.  String or multimodal content array.",
    )
    is_error: bool = Field(default=False, description="True if the tool execution failed.")


class WaitResponse(BaseModel):
    """Response for ``GET /v1/sessions/{id}/wait``."""

    events: list[Event]
    session_status: SessionStatus
    session_stop_reason: dict[str, Any] | None
    session_awaiting: list[AwaitingToolCall] = Field(default_factory=list)
    next_after: int


class SessionAwaitResponse(BaseModel):
    """Response for GET /v1/sessions/{id}/await — the await-a-completion primitive,
    session backing. One envelope over two monotonic predicates (request_id
    correlation, or reacted>=watermark). Poll until `done`."""

    done: bool
    last_reacted_seq: int
    result: Any = None
    is_error: bool = False
    error: dict[str, Any] | None = None


class ContextResponse(BaseModel):
    """Response for ``GET /v1/sessions/{id}/context``.

    The exact chat-completions payload the worker would send to LiteLLM
    if a step ran right now.  Dry-run: no side effects — no events
    appended, no status bump, no skill files provisioned.
    """

    session_id: str
    model: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]


class ToolConfirmationRequest(BaseModel):
    """Request body for ``POST /v1/sessions/{id}/tool-confirmations``.

    Used for built-in tools with ``permission: "always_ask"``. The client
    inspects the pending tool call and either allows it (the worker will
    execute it) or denies it (the model receives an error with the deny
    message).
    """

    model_config = ConfigDict(extra="forbid")

    tool_call_id: str = Field(description="The tool_call_id to confirm or deny.")
    result: Literal["allow", "deny"]
    deny_message: str | None = Field(
        default=None,
        description="When result='deny', an optional message explaining why. Shown to the model.",
    )
