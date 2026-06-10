"""Context composition for a single step.

Extracted from :func:`aios.harness.loop.run_session_step` so the same code
path feeds both the worker's next model call and ``GET /v1/sessions/:id/
context`` (issue #60).  Keeping the two paths byte-identical is the whole
point of the endpoint — a ``/context`` response that diverges from what
the worker is about to send is useless for diagnosis.

Side-effects kept OUT of this function (so the endpoint is a true
dry-run):

- ``provision_skill_files`` — filesystem writes.  Returned via
  ``StepContext.skill_versions`` so ``run_session_step`` can call it
  afterward, before the model runs.
- Session-state mutations (``set_session_status``, event appends).
- Tool dispatch (the confirmed-tool early-return path in
  ``run_session_step`` runs BEFORE this function).
- Span emission (``context_build_start``/``end`` live in
  ``run_session_step``).

I/O still happens: MCP discovery, skill-ref resolution, read-only
database queries.  That's unavoidable — the endpoint has to do the same
work to honor the "byte-identical" promise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from aios.harness._text import join_blocks
from aios.harness.context import (
    build_messages,
    merge_adjacent_user_messages,
    message_is_notification_marker,
    stub_missing_reasoning_content,
)
from aios.tools.registry import to_openai_tools

if TYPE_CHECKING:
    import asyncpg

    from aios.models.agents import (
        Agent,
        AgentVersion,
        HttpServerSpec,
        McpServerSpec,
        ToolSpec,
    )
    from aios.models.events import Event
    from aios.models.memory_stores import MemoryStoreResourceEcho
    from aios.models.sessions import Session
    from aios.models.skills import SkillVersion


# Generic affordance prose explaining the in-sandbox ``tool`` CLI. Rendered
# into the system prompt whenever the agent has at least one
# ``always_allow`` MCP toolset entry. Worded in stable runtime terms — no
# dev-world references — so it remains agent-actionable across releases.
# ``<method>`` is used as the placeholder for the MCP method name so the
# binary name (``tool``) and the meta-variable don't collide visually.
_MCP_CLI_HINT = (
    "## Sandbox tool CLI\n\n"
    "Permitted MCP tools are also callable from inside the sandbox via the "
    "`tool` binary, so you can invoke them programmatically from `bash` "
    "without paying an inference cycle per call:\n\n"
    "    tool                              list reachable tools (built-ins + MCP servers)\n"
    "    tool <server>                     list methods on a server\n"
    "    tool <server> <method> --help     show description + JSON schema\n"
    "    tool <server> <method> '{...}'    invoke with JSON arguments\n\n"
    "Use the CLI when you want scriptable invocation (composition with `jq`, "
    "`xargs`, redirection, scheduled wakes). The model-tool invocation path "
    "remains available for the same tools."
)


def _has_always_allow_mcp_tool(agent_tools: list[ToolSpec]) -> bool:
    """True iff at least one enabled mcp_toolset entry resolves any tool
    to ``always_allow``.

    The CLI hint is purely informational — emitting it for an agent whose
    toolset has only ``always_ask`` policies would lie to the model
    (every CLI call would 403). Showing it whenever there's at least one
    ``always_allow`` path is the conservative truthful default.
    """
    for spec in agent_tools:
        if spec.type != "mcp_toolset" or not spec.enabled:
            continue
        default = spec.default_config
        if (
            default
            and default.permission_policy
            and default.permission_policy.type == "always_allow"
        ):
            return True
        if spec.configs:
            for cfg in spec.configs:
                if (
                    cfg.enabled
                    and cfg.permission_policy
                    and cfg.permission_policy.type == "always_allow"
                ):
                    return True
    return False


@dataclass(frozen=True)
class StepPrelude:
    """Events-independent portion of a step's payload.

    Everything here depends only on ``agent`` / ``channels`` / ``session``
    — not on which events windowing picks.  Computed before windowing so
    ``read_windowed_events`` can subtract the overhead from the budget
    (see ``overhead_local`` there).

    ``tail_block_upper_bound_local`` is the worst-case size of the
    channels tail block the composer will append after windowing — a
    conservative bound computed from ``channels`` alone (no events, no
    unread counts).  Reserving this ahead of time keeps the send-time
    payload under ``window_max`` even when the tail renders at its
    fattest (every channel at 9999 unread with a maxed-out preview).
    """

    system_prompt: str
    tools: list[dict[str, Any]]
    skill_versions: list[SkillVersion]
    tail_block_upper_bound_local: int


@dataclass(frozen=True)
class StepContext:
    """Composed inputs for a single model call."""

    model: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    reacting_to: int
    skill_versions: list[SkillVersion]


async def compute_step_prelude(
    pool: asyncpg.Pool[Any],
    session_id: str,
    *,
    account_id: str,
    session: Session,
    agent: Agent | AgentVersion,
    channels: list[str],
    memory_store_echoes: list[MemoryStoreResourceEcho],
) -> StepPrelude:
    """Build the events-independent parts of the step payload.

    Exists so windowing can know the system+tools overhead before it
    picks the event slate.  The returned ``StepPrelude`` feeds
    :func:`compose_step_context` unchanged, so the composed prompt stays
    byte-identical to what it was before the split.
    """
    from aios.harness.channels import (
        augment_with_focal_paradigm,
        max_tail_block_local,
    )
    from aios.harness.loop import (
        _switch_channel_tool_spec,
        discover_session_mcp_tools,
    )
    from aios.harness.memory_stores import augment_with_memory_stores
    from aios.harness.skills import augment_system_prompt
    from aios.services import skills as skills_service

    tools = to_openai_tools(agent.tools)
    # The switch_channel built-in is the agent's only path to mutate
    # focal attention; inject it whenever the session has bound channels.
    if channels:
        tools.append(_switch_channel_tool_spec())
    # return/error are a workflow agent child's only way to finish — injected
    # only for a background child of a run (§3.5), never a foreground session.
    if session.origin == "background" and session.parent_run_id is not None:
        from aios.tools.workflow_completion import workflow_completion_tool_specs

        tools.extend(workflow_completion_tool_specs())

    mcp_servers_block = ""
    if agent.mcp_servers:
        mcp_tools, mcp_instructions = await discover_session_mcp_tools(
            pool, session_id, agent, account_id=account_id
        )
        tools.extend(mcp_tools)
        mcp_servers_block = _build_instructions_block(agent.mcp_servers, mcp_instructions)
    http_servers_block = _build_http_servers_block(agent.http_servers)
    cli_hint = _MCP_CLI_HINT if _has_always_allow_mcp_tool(agent.tools) else ""
    instructions_block = join_blocks(cli_hint, mcp_servers_block, http_servers_block)

    # Custom tools declared on connections attached to this session
    # (single_session, per_chat origin, or operator-bound chat).  Each
    # entry sits unresolved in the event log until the connector
    # executes it externally and POSTs the result back via
    # ``/tool-results`` (#301).  Resolved via the ``ToolProvider``
    # Protocol (#328) so the harness doesn't import connector-subsystem
    # code directly.
    from aios.harness import runtime as harness_runtime
    from aios.models.agents import ToolSpec

    connection_tool_dicts = await harness_runtime.require_tool_provider().list_tools_for_session(
        pool, session_id
    )
    if connection_tool_dicts:
        connection_tools = [ToolSpec.model_validate(d) for d in connection_tool_dicts]
        tools.extend(to_openai_tools(connection_tools))

    skill_versions = (
        await skills_service.resolve_skill_refs(pool, agent.skills, account_id=account_id)
        if agent.skills
        else []
    )
    system_prompt = augment_system_prompt(agent.system, skill_versions)
    system_prompt = augment_with_focal_paradigm(system_prompt, channels)
    system_prompt = join_blocks(system_prompt, instructions_block)
    system_prompt = augment_with_memory_stores(system_prompt, memory_store_echoes)

    return StepPrelude(
        system_prompt=system_prompt,
        tools=tools,
        skill_versions=skill_versions,
        tail_block_upper_bound_local=max_tail_block_local(channels),
    )


def _build_instructions_block(
    mcp_servers: list[McpServerSpec], instructions_by_server: dict[str, str]
) -> str:
    """Render per-server affordance prose, respecting ``include_instructions``.

    Servers iterate in ``agent.mcp_servers`` declaration order, which is
    fixed across steps — keeping the rendered block prefix-cache-stable.
    """
    sections: list[str] = []
    for s in mcp_servers:
        if not s.include_instructions:
            continue
        text = instructions_by_server.get(s.name)
        if not text:
            continue
        sections.append(f"## MCP server: {s.name}\n\n{text}")
    return "\n\n".join(sections)


def _build_http_servers_block(http_servers: list[HttpServerSpec]) -> str:
    """Render the agent's ``http_servers`` allowlist for the system prompt.

    Includes server description plus each enabled route's pattern and
    description, so the model knows what ``http_request`` calls it can
    make. Iteration order is ``agent.http_servers`` declaration order
    (prefix-cache-stable across steps).
    """
    if not http_servers:
        return ""
    sections: list[str] = []
    for s in http_servers:
        lines = [f"## HTTP server: {s.name} ({s.base_url})"]
        if s.description:
            lines.append("")
            lines.append(s.description)
        enabled_routes = [r for r in s.routes if r.enabled]
        if enabled_routes:
            lines.append("")
            lines.append("Routes:")
            for r in enabled_routes:
                suffix = f" — {r.description}" if r.description else ""
                lines.append(f"- {r.path_pattern}{suffix}")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def _agent_owes_response(messages: list[dict[str, Any]]) -> bool:
    """True when the conversation ends with a *direct* stimulus to answer.

    Gates the ephemeral channels tail block. Two trailing cases are a direct
    stimulus the agent must engage with in focal context, where appending the
    tail would make a status listing the literal final message and mute
    literal-minded models (claude-fable-5): a **focal user inbound** (full
    content) and a **tool result**. Keep the real stimulus last for those.

    Two trailing cases are NOT a direct stimulus, so keep the tail:
    * a non-focal **notification marker** (``🔔 …``) — the tail's channel
      listing is its navigation companion (how to ``switch_channel`` to it);
    * an **assistant** turn — an idle/sweep re-check where the channel-status
      listing is the useful signal.
    """
    if not messages:
        return False
    last = messages[-1]
    role = last.get("role")
    if role == "tool":
        return True
    if role == "user":
        return not message_is_notification_marker(last)
    return False


async def compose_step_context(
    *,
    pool: asyncpg.Pool[Any],
    session: Session,
    account_id: str,
    agent: Agent | AgentVersion,
    channels: list[str],
    prelude: StepPrelude,
    events: list[Event],
    in_flight_tool_call_ids: frozenset[str] = frozenset(),
) -> StepContext:
    """Compose the chat-completions payload for a step.

    Takes a prelude built by :func:`compute_step_prelude` and the
    windowed events slate; glues them into the final message list.

    ``pool`` + ``account_id`` back a single read-only query — the
    session's ``workspace_volume_path`` — so the renderer can resolve
    ``/workspace``-prefixed image attachments to host bytes.

    ``in_flight_tool_call_ids`` selects the pending placeholder variant
    for each unresolved tool_call. Background-executing tasks get the
    "still executing in the background" wording; everything else
    (custom, awaiting-confirm) gets the "external action" wording.
    """
    from aios.harness.channels import build_channels_tail_block
    from aios.services import accounts as accounts_service
    from aios.services import sessions as sessions_service

    # Issue #630 follow-up: the renderer's ``/workspace`` attachment branch
    # needs the actual bind-mount source.  Read it from the session row
    # (``workspace_volume_path``) — the authoritative, always-present
    # source — rather than a live ``SandboxHandle``.  A handle is absent
    # for chat-only sessions, idle-evicted sandboxes, the window between a
    # worker restart and the next cold-start, and the API process
    # (``GET /v1/sessions/:id/context``), which never initializes the
    # sandbox registry.  Sourcing from the row resolves ``/workspace``
    # attachments correctly in all of those cases.
    workspace_path = await sessions_service.load_session_workspace_path(
        pool, session.id, account_id=account_id
    )
    # Effective account timezone for the ``received=`` envelope — see
    # ``services.accounts.resolve_effective_timezone``. Stable across rebuilds
    # while the config is unchanged; a tz config change re-renders history
    # once (a deliberate one-time prompt-cache bust, same class as any
    # renderer change).
    tz_name = await accounts_service.resolve_effective_timezone(pool, account_id)

    ctx = build_messages(
        events,
        system_prompt=prelude.system_prompt,
        model=agent.model,
        session_id=session.id,
        workspace_path=workspace_path,
        in_flight_tool_call_ids=in_flight_tool_call_ids,
        tz_name=tz_name,
    )

    # Tail block lives *after* build_messages so its per-step mutations
    # (unread counts, previews) don't bust the prefix cache.  Paradigm
    # prose stays in the cache-stable system prompt above.
    #
    # Only append it when the conversation already ends with an assistant turn
    # (an idle/sweep re-check, where the channel-status listing is the useful
    # signal). When the last message is a user or tool turn, the agent owes a
    # response: appending the tail makes a "0 unread" status block the literal
    # final message, and literal-minded models (claude-fable-5) anchor on it and
    # emit an empty turn instead of answering (opus looks back past it; fable does
    # not). Keep the real stimulus last in that case.
    tail = build_channels_tail_block(channels, events, session.focal_channel)
    if tail is not None and not _agent_owes_response(ctx.messages):
        ctx.messages.append(tail)

    # Merge consecutive user inbounds into one turn (Anthropic requires
    # alternating roles). This replaces the old "." placeholder separator,
    # which degenerate-poisoned literal models like claude-fable-5.
    messages = merge_adjacent_user_messages(ctx.messages)

    # Unblock thinking-mode models: DeepSeek V4 Flash rejects assistant
    # turns without reasoning_content.  Empty stub is ignored by all
    # non-thinking providers we've tested (Anthropic, OpenAI, Gemini,
    # Llama, non-thinking DeepSeek).
    stub_missing_reasoning_content(messages)

    return StepContext(
        model=agent.model,
        messages=messages,
        tools=prelude.tools,
        reacting_to=ctx.reacting_to,
        skill_versions=prelude.skill_versions,
    )
