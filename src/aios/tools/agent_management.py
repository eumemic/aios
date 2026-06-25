"""Agent-acting agent builtins — self-management of the agent resource (T1).

Model-callable tools that let an agent author, edit, archive, and read the
versioned *agent* resource the way a human operator (or the ``aios`` CLI / the
``/v1/agents`` HTTP plane) does. Agents were the only major versioned resource
with CRUD on the HTTP plane only and **no** model tools; this closes that
plane-asymmetry, reaching full parity with the workflow author/edit/read trio
(``workflow_management``) and the skill builtins (``skill_management``).

Built from two existing, proven pieces — **no new machinery**:

* the ``skill_upsert`` / ``create_workflow`` model-tool plumbing (an arg model
  with ``extra="forbid"`` so an injected ``account_id`` / ``creator_session_id``
  is rejected before the handler; ``account_id`` derived server-side from the
  executing ``session_id`` via ``load_session_account_id`` — **never** model input),
  and
* the create-time surface-attenuation clamp that already guards ``create_workflow``
  (``services.agents._enforce_surface_attenuation``): the declared agent surface
  must be a subset of the creating/editing agent's own (``ForbiddenError`` on a
  breach, ``detail=surface_diff``). ``update_agent`` re-clamps the merged surface.

The #794/#823 spawn-edge reclamp stays INDEPENDENT and unchanged: when an
authored agent is later spawned via ``agent()`` / ``call_agent``, ``step.py``
recomputes ``surface = clamp(agent ∩ run)`` and freezes model identity / api_base.
Create-time clamps authoring to ⊆ creator; spawn-time re-clamps to ⊆ run. This
module is the create-time arm; it does not touch the spawn edge.

**Identity is load-bearing, so two invariants hold (see F1/F2 in the issue):**
1. The trusted ids (``creator_session_id`` / ``editor_session_id``, ``account_id``)
   are NEVER tool-schema fields — every arg model is ``extra="forbid"``
   (``additionalProperties: false`` on the wire), so an injected key is rejected
   before the handler runs.
2. Handlers map service kwargs explicitly from the validated model — never
   ``**arguments`` — and ``account_id`` is loaded server-side from the session id;
   the creator/editor identity is the harness-supplied executing ``session_id``,
   never an argument.

All register ``transport="agent_tool"`` (model-only; the CLI broker refuses them).

This module deliberately does NOT introduce an ``agent_upsert`` (per T2: retire
upsert as a shape) — ``create_agent`` and ``update_agent`` are distinct verbs and
the model branches create-vs-update by choosing the tool. ``create_agent`` 409s on
a duplicate ``(account_id, name)`` (existing service behavior).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from aios.harness import runtime
from aios.models.agents import AgentCreate, AgentUpdate
from aios.services import agents as agents_service
from aios.services import sessions as sessions_service
from aios.tools.input import tool_input
from aios.tools.registry import registry

# Heavy/internal fields trimmed from list_agents summaries — keep id, name,
# version, description, model, timestamps (the lean summary set; account-scoped
# already, so no account_id leak concern). get_agent returns the FULL agent so an
# editor can re-read before retrying an update whose version token was rejected.
_AGENT_LIST_EXCLUDE = {
    "system",
    "tools",
    "skills",
    "mcp_servers",
    "http_servers",
    "metadata",
    "litellm_extra",
}


# ─── argument models (parameters_schema + parse, in one place) ───────────────
#
# ``litellm_extra`` is deliberately WITHHELD from the agent-authoring model tools
# (#823 tripwire, ``tests/unit/test_attenuation.py::test_no_registered_tool_exposes_
# litellm_extra``): it carries ``api_base``, which redirects the authored agent's
# model call. The surface meet does not clamp model routing, so the model-tool plane
# must not let a self-authoring agent MINT a new ``api_base`` — neither from the
# model-visible schema (``_drop_litellm_extra`` strips the property) nor by smuggling
# it past ``extra="forbid"`` (the inherited field is *re-typed closed* below so any
# value other than its default is rejected). Model identity stays operator-only on the
# authoring plane (set it via the HTTP plane); the #823 spawn-edge clamp
# (``workflows/step.py``) remains the independent backstop that fail-closes an untrusted
# ``api_base`` at spawn. ``model`` (the LiteLLM model string) IS authorable — only the
# routing override is withheld.


def _drop_litellm_extra(schema: dict[str, Any]) -> None:
    """``json_schema_extra`` hook: remove ``litellm_extra`` from the emitted JSON schema
    so the model never sees the model-routing field (the #823 tripwire's contract)."""
    schema.get("properties", {}).pop("litellm_extra", None)


class _CreateAgentArgs(AgentCreate):
    """``create_agent`` arguments — the ``AgentCreate`` body with the model-routing
    override field withheld.

    Inherits ``extra="forbid"`` so the trusted ``creator_session_id`` / ``account_id``
    are never fields (an injected key is rejected up front). The model-routing override
    is dropped from the model-visible schema and rejected if non-empty — it is not
    authorable from inside a session (see the module note above).
    """

    model_config = ConfigDict(extra="forbid", json_schema_extra=_drop_litellm_extra)

    @field_validator("litellm_extra")
    @classmethod
    def _reject_model_routing(cls, v: dict[str, Any]) -> dict[str, Any]:
        if v:
            raise ValueError("model routing override is not authorable via the create_agent tool")
        return v


class _UpdateAgentArgs(AgentUpdate):
    """``update_agent`` arguments — the ``AgentUpdate`` body plus the path-style id,
    with the model-routing override field withheld.

    Subclassing inherits every field (and its constraints) and ``extra="forbid"`` —
    the trusted ``editor_session_id`` is never a field, so an injected key is rejected;
    the model-routing override is dropped from the schema and rejected if set (see the
    module note above).
    """

    model_config = ConfigDict(extra="forbid", json_schema_extra=_drop_litellm_extra)

    agent_id: str

    @field_validator("litellm_extra")
    @classmethod
    def _reject_model_routing(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        if v:
            raise ValueError("model routing override is not authorable via the update_agent tool")
        return v


class _AgentIdArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str


class _GetAgentArgs(_AgentIdArgs):
    pass


class _ListAgentsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    limit: int = Field(default=50, ge=1, le=200)
    after: str | None = None
    name: str | None = None


# ─── handler plumbing ────────────────────────────────────────────────────────
#
# Handlers map service kwargs explicitly (F1) and otherwise let service errors
# propagate: the dispatch layer (``tool_dispatch._classify_tool_error``) turns a
# client-class (4xx) ``AiosError`` — a denied attenuation (``ForbiddenError``), a
# duplicate name (``ConflictError``, 409), a stale-version conflict, an unknown
# ``agent_id`` (``NotFoundError``, 404) — into a clean, model-visible result without
# evicting the sandbox, and a 5xx into a genuine failure. Only argument parsing bails
# locally (``tool_input``).


async def create_agent_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    body = tool_input(_CreateAgentArgs, arguments)
    agent = await agents_service.create_agent(
        pool,
        account_id=account_id,
        name=body.name,
        model=body.model,
        system=body.system,
        tools=body.tools,
        skills=body.skills,
        mcp_servers=body.mcp_servers,
        http_servers=body.http_servers,
        description=body.description,
        metadata=body.metadata,
        litellm_extra=body.litellm_extra,
        window_min=body.window_min,
        window_max=body.window_max,
        creator_session_id=session_id,
    )
    return agent.model_dump(mode="json")


async def update_agent_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    args = tool_input(_UpdateAgentArgs, arguments)
    agent = await agents_service.update_agent(
        pool,
        args.agent_id,
        account_id=account_id,
        expected_version=args.version,
        name=args.name,
        model=args.model,
        system=args.system,
        tools=args.tools,
        skills=args.skills,
        mcp_servers=args.mcp_servers,
        http_servers=args.http_servers,
        description=args.description,
        metadata=args.metadata,
        litellm_extra=args.litellm_extra,
        window_min=args.window_min,
        window_max=args.window_max,
        editor_session_id=session_id,
    )
    return agent.model_dump(mode="json")


async def archive_agent_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    args = tool_input(_AgentIdArgs, arguments)
    await agents_service.archive_agent(pool, args.agent_id, account_id=account_id)
    return {"agent_id": args.agent_id, "archived": True}


async def get_agent_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    args = tool_input(_GetAgentArgs, arguments)
    agent = await agents_service.get_agent(pool, args.agent_id, account_id=account_id)
    return agent.model_dump(mode="json")  # FULL — incl. surface + version (the re-read loop)


async def list_agents_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    args = tool_input(_ListAgentsArgs, arguments)
    agents = await agents_service.list_agents(
        pool, account_id=account_id, limit=args.limit, after=args.after, name=args.name
    )
    return {"agents": [a.model_dump(mode="json", exclude=_AGENT_LIST_EXCLUDE) for a in agents]}


# ─── descriptions + registration ─────────────────────────────────────────────

CREATE_AGENT_DESCRIPTION = (
    "Author a new agent under your account. Its declared tool / MCP-server / "
    "HTTP-server surface must be a subset of your own — you cannot grant an agent a "
    "tool, MCP server, or HTTP server you don't yourself have (a breach is rejected). "
    "The agent's name must be unique among your live agents (a duplicate is rejected). "
    "Returns the created agent (id, name, version, surface). When this agent is later "
    "spawned (via agent()/call_agent) its surface is re-clamped to that run's surface."
)
UPDATE_AGENT_DESCRIPTION = (
    "Update one of your agents in place, creating a new immutable version. Pass the "
    "current 'version' as an optimistic-concurrency token (a stale token is rejected — "
    "re-read with get_agent and retry). Omitted fields are preserved from the prior "
    "version. The resulting (merged) tool/server surface must still be a subset of your "
    "own. In-flight sessions are unaffected (they pin their agent version)."
)
ARCHIVE_AGENT_DESCRIPTION = (
    "Archive one of your agents by id; it disappears from list_agents and is hidden "
    "from default lists, but its row and full version history persist and existing "
    "sessions referencing it keep working. Archiving releases the agent name for a "
    "fresh create_agent."
)
GET_AGENT_DESCRIPTION = (
    "Fetch one of your agents in full by id — including its system prompt, full "
    "tool/MCP/HTTP surface, and current 'version'. Use this to re-read an agent before "
    "retrying an update_agent whose optimistic-concurrency token was rejected: read the "
    "fresh version, reconcile your edit, and retry with that version."
)
LIST_AGENTS_DESCRIPTION = (
    "List your account's agents (latest version of each), newest first, as lean "
    "summaries (id, name, model, description, version, timestamps) — no system prompt "
    "or surface bodies. Optional 'name' filter; page with 'limit' and 'after' (the last "
    "id seen); a full page means there may be more — call again. To read an agent's "
    "full config, fetch it with get_agent."
)


def _register() -> None:
    registry.register(
        name="create_agent",
        description=CREATE_AGENT_DESCRIPTION,
        parameters_schema=_CreateAgentArgs.model_json_schema(),
        handler=create_agent_handler,
        transport="agent_tool",
    )
    registry.register(
        name="update_agent",
        description=UPDATE_AGENT_DESCRIPTION,
        parameters_schema=_UpdateAgentArgs.model_json_schema(),
        handler=update_agent_handler,
        transport="agent_tool",
    )
    registry.register(
        name="archive_agent",
        description=ARCHIVE_AGENT_DESCRIPTION,
        parameters_schema=_AgentIdArgs.model_json_schema(),
        handler=archive_agent_handler,
        transport="agent_tool",
    )
    registry.register(
        name="get_agent",
        description=GET_AGENT_DESCRIPTION,
        parameters_schema=_GetAgentArgs.model_json_schema(),
        handler=get_agent_handler,
        transport="agent_tool",
    )
    registry.register(
        name="list_agents",
        description=LIST_AGENTS_DESCRIPTION,
        parameters_schema=_ListAgentsArgs.model_json_schema(),
        handler=list_agents_handler,
        transport="agent_tool",
    )


_register()
