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

    from aios.models.agents import Agent, AgentVersion
    from aios.models.channel_bindings import ChannelBinding
    from aios.models.connections import Connection
    from aios.models.events import Event
    from aios.models.sessions import Session
    from aios.models.skills import SkillVersion


@dataclass(frozen=True)
class StepPrelude:
    """Events-independent portion of a step's payload.

    Everything here depends only on ``agent`` / ``bindings`` /
    ``connections`` / ``session`` — not on which events windowing picks.
    Computed before windowing so ``read_windowed_events`` can subtract
    the overhead from the budget (see ``overhead_local`` there).

    ``tail_block_upper_bound_local`` is the worst-case size of the
    channels tail block the composer will append after windowing — a
    conservative bound computed from ``bindings`` alone (no events, no
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
    bindings: list[ChannelBinding],
    connections: list[Connection],
) -> StepPrelude:
    """Build the events-independent parts of the step payload.

    Exists so windowing can know the system+tools overhead before it
    picks the event slate.  The returned ``StepPrelude`` feeds
    :func:`compose_step_context` unchanged, so the composed prompt stays
    byte-identical to what it was before the split.
    """
    from aios.harness.channels import (
        augment_with_connector_instructions,
        augment_with_focal_paradigm,
        max_tail_block_local,
    )
    from aios.harness.loop import (
        _hide_focal_channel_tools_when_phone_down,
        _switch_channel_tool_spec,
        discover_session_mcp_tools,
        mcp_channel_context_by_server,
    )
    from aios.harness.skills import augment_system_prompt
    from aios.services import skills as skills_service

    tools = to_openai_tools(agent.tools)
    # Inject the built-in switch_channel tool when the session has any
    # active bindings — the only way the agent can mutate focal attention.
    if bindings:
        tools.append(_switch_channel_tool_spec())

    mcp_instructions: dict[str, str] = {}
    if agent.mcp_servers or connections:
        mcp_tools, mcp_instructions = await discover_session_mcp_tools(
            pool, session_id, agent, connections
        )
        # Hide focal-channel MCP tools when focal is NULL — can't type
        # into a chat you aren't attending to.
        channel_context_by_server = mcp_channel_context_by_server(
            agent.tools,
            connections,
            agent_mcp_server_names={s.name for s in agent.mcp_servers},
            agent_mcp_server_urls={s.url for s in agent.mcp_servers},
        )
        mcp_tools = _hide_focal_channel_tools_when_phone_down(
            mcp_tools, session.focal_channel, channel_context_by_server
        )
        tools.extend(mcp_tools)

    skill_versions = (
        await skills_service.resolve_skill_refs(pool, agent.skills) if agent.skills else []
    )
    system_prompt = augment_system_prompt(agent.system, skill_versions)
    system_prompt = augment_with_focal_paradigm(system_prompt, bindings)
    system_prompt = augment_with_connector_instructions(
        system_prompt, mcp_instructions, connections
    )

    return StepPrelude(
        system_prompt=system_prompt,
        tools=tools,
        skill_versions=skill_versions,
        tail_block_upper_bound_local=max_tail_block_local(bindings),
    )


async def compose_step_context(
    *,
    session: Session,
    agent: Agent | AgentVersion,
    bindings: list[ChannelBinding],
    prelude: StepPrelude,
    events: list[Event],
) -> StepContext:
    """Compose the chat-completions payload for a step.

    Takes a prelude built by :func:`compute_step_prelude` and the
    windowed events slate; glues them into the final message list.
    """
    from aios.harness.channels import build_channels_tail_block

    ctx = build_messages(events, system_prompt=prelude.system_prompt)

    # Tail block lives *after* build_messages so its per-step mutations
    # (unread counts, previews) don't bust the prefix cache.  Paradigm
    # prose stays in the cache-stable system prompt above.
    tail = build_channels_tail_block(bindings, events, session.focal_channel)
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
