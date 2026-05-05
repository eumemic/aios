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

from aios.harness.context import (
    build_messages,
    separate_adjacent_user_messages,
    stub_missing_reasoning_content,
)
from aios.tools.registry import to_openai_tools

if TYPE_CHECKING:
    import asyncpg

    from aios.models.agents import Agent, AgentVersion, McpServerSpec
    from aios.models.events import Event
    from aios.models.memory_stores import MemoryStoreResourceEcho
    from aios.models.sessions import Session
    from aios.models.skills import SkillVersion


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
    from aios.harness import runtime
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

    instructions_block = ""
    if agent.mcp_servers:
        mcp_tools, mcp_instructions = await discover_session_mcp_tools(pool, session_id, agent)
        tools.extend(mcp_tools)
        instructions_block = _build_instructions_block(agent.mcp_servers, mcp_instructions)

    # Connector-subprocess (stdio MCP) tools come from the worker-scoped
    # registry, NOT from ``agent.mcp_servers`` (which only covers HTTP MCP
    # servers declared on the agent).  Without this enumeration the model
    # sees connector tool *names* only when they appear in prior tool_use
    # events from the session log — and never sees their schemas, forcing
    # it to guess parameter shapes and learn from JSON-schema validation
    # errors the harness sends back.
    connector_registry = runtime.connector_subprocess_registry
    if connector_registry is not None:
        tools.extend(await connector_registry.list_tools())

    skill_versions = (
        await skills_service.resolve_skill_refs(pool, agent.skills) if agent.skills else []
    )
    system_prompt = augment_system_prompt(agent.system, skill_versions)
    system_prompt = augment_with_focal_paradigm(system_prompt, channels)
    if instructions_block:
        if system_prompt:
            system_prompt = system_prompt + "\n\n" + instructions_block
        else:
            system_prompt = instructions_block
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


async def compose_step_context(
    *,
    session: Session,
    agent: Agent | AgentVersion,
    channels: list[str],
    prelude: StepPrelude,
    events: list[Event],
) -> StepContext:
    """Compose the chat-completions payload for a step.

    Takes a prelude built by :func:`compute_step_prelude` and the
    windowed events slate; glues them into the final message list.
    """
    from aios.harness.channels import build_channels_tail_block

    ctx = build_messages(
        events,
        system_prompt=prelude.system_prompt,
        model=agent.model,
        session_id=session.id,
    )

    # Tail block lives *after* build_messages so its per-step mutations
    # (unread counts, previews) don't bust the prefix cache.  Paradigm
    # prose stays in the cache-stable system prompt above.
    tail = build_channels_tail_block(channels, events, session.focal_channel)
    if tail is not None:
        ctx.messages.append(tail)

    # Block LiteLLM's adjacent-same-role merge on Anthropic so the tail
    # isn't concatenated into the preceding user inbound.
    messages = separate_adjacent_user_messages(ctx.messages)

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
