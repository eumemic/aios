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

from aios.harness.context import build_messages, separate_adjacent_user_messages
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
class StepContext:
    """Composed inputs for a single model call."""

    model: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    reacting_to: int
    skill_versions: list[SkillVersion]


async def compose_step_context(
    pool: asyncpg.Pool[Any],
    session_id: str,
    *,
    session: Session,
    agent: Agent | AgentVersion,
    bindings: list[ChannelBinding],
    connections: list[Connection],
    events: list[Event],
) -> StepContext:
    """Compose the chat-completions payload for a step.

    Callers must have already loaded ``session`` / ``agent`` / ``bindings``
    / ``connections`` / ``events`` so the endpoint and the worker pay
    the same I/O cost profile.  This function adds MCP discovery and
    skill-ref resolution on top.
    """
    from aios.harness.channels import (
        augment_with_connector_instructions,
        augment_with_focal_paradigm,
        build_channels_tail_block,
    )
    from aios.harness.loop import (
        _hide_conn_tools_when_phone_down,
        _switch_channel_tool_spec,
        discover_session_mcp_tools,
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
        # Hide connection-provided MCP tools when focal is NULL — can't
        # type into a chat you aren't attending to.
        mcp_tools = _hide_conn_tools_when_phone_down(mcp_tools, session.focal_channel)
        tools.extend(mcp_tools)

    skill_versions = (
        await skills_service.resolve_skill_refs(pool, agent.skills) if agent.skills else []
    )
    system_prompt = augment_system_prompt(agent.system, skill_versions)
    system_prompt = augment_with_focal_paradigm(system_prompt, bindings)
    system_prompt = augment_with_connector_instructions(
        system_prompt, mcp_instructions, connections
    )

    ctx = build_messages(events, system_prompt=system_prompt)

    # Tail block lives *after* build_messages so its per-step mutations
    # (unread counts, previews) don't bust the prefix cache.  Paradigm
    # prose stays in the cache-stable system prompt above.
    tail = build_channels_tail_block(bindings, events, session.focal_channel)
    if tail is not None:
        ctx.messages.append(tail)

    # Block LiteLLM's adjacent-same-role merge on Anthropic so the tail
    # isn't concatenated into the preceding user inbound.
    messages = separate_adjacent_user_messages(ctx.messages)

    return StepContext(
        model=agent.model,
        messages=messages,
        tools=tools,
        reacting_to=ctx.reacting_to,
        skill_versions=skill_versions,
    )
