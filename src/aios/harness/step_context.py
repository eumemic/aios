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
from typing import TYPE_CHECKING, Any, NamedTuple

from aios.harness._text import join_blocks
from aios.harness.context import (
    OMISSION_MARKER_UPPER_BOUND_LOCAL,
    build_messages,
    merge_adjacent_user_messages,
    message_is_notification_marker,
    stub_missing_reasoning_content,
)
from aios.harness.tokens import approx_tokens
from aios.harness.window import WindowOmission
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
    from aios.models.sessions import Obligation, Session
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
    "Use the CLI when you want scriptable use (composition with `jq`, "
    "`xargs`, redirection, scheduled wakes). The model-tool call path "
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

    ``obligations`` is the session's open **awaited** obligations (#1413),
    fetched once here (the unconditional ``get_open_obligations`` that also
    decides the ``return``/``error`` tool gate) and reused by the composer to
    render the obligations tail block — no second query.
    ``obligations_block_upper_bound_local`` is the worst-case size of that
    block, bounded from the actual fetched obligations (real count + each real
    summary, capped) so reserving it keeps the payload under ``window_max``.
    """

    system_prompt: str
    tools: list[dict[str, Any]]
    skill_versions: list[SkillVersion]
    tail_block_upper_bound_local: int
    obligations: list[Obligation]
    obligations_block_upper_bound_local: int


class PreludeOverheadSplit(NamedTuple):
    """The step's overhead-local cost split by content class (#1609).

    The windower weights ``system`` and ``tools`` overhead by their own
    per-class coefficients (the system prompt and tool schemas price
    differently against the provider tokenizer), so the overhead is no
    longer a single opaque scalar.  ``reserves`` is the post-windowing
    reserved upper bounds (channels tail, obligations tail, omission
    marker) — conservative text-shaped padding, weighted as ``text``.

    ``total`` reproduces the pre-#1609 single ``overhead_local`` integer
    (the three fields summed), so any caller that only needs the scalar
    can read ``.total`` and stay byte-identical.
    """

    system: int
    tools: int
    reserves: int

    @property
    def total(self) -> int:
        return self.system + self.tools + self.reserves


def prelude_overhead_local(prelude: StepPrelude) -> PreludeOverheadSplit:
    """Token cost the composer adds on top of the windowed events, split
    by content class, in local (``approx_tokens``) units — the
    ``overhead_local`` argument to ``read_windowed_events`` (#1609).

    System prompt + tool schemas (each weighted separately by the
    windower), plus the reserved upper bounds for the post-windowing
    additions: the channels tail block, the obligations tail block
    (#1413), and the omission marker (#738). All reserves are reserved
    unconditionally — any may not render, but the budget must hold when
    they do — and are accounted as ``text``-class padding.

    Returns a :class:`PreludeOverheadSplit`; ``.total`` reproduces the
    old single scalar exactly (system+tools costed together previously,
    now costed separately and summed — same payload, same total).
    """
    system_local = approx_tokens([{"role": "system", "content": prelude.system_prompt}])
    tools_local = approx_tokens([], tools=prelude.tools) if prelude.tools else 0
    reserves_local = (
        prelude.tail_block_upper_bound_local
        + prelude.obligations_block_upper_bound_local
        + OMISSION_MARKER_UPPER_BOUND_LOCAL
    )
    return PreludeOverheadSplit(
        system=system_local,
        tools=tools_local,
        reserves=reserves_local,
    )


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
    from aios.db import queries
    from aios.harness.channels import (
        augment_with_focal_paradigm,
        max_tail_block_local,
    )
    from aios.harness.loop import (
        _switch_channel_tool_spec,
        discover_session_mcp_tools,
    )
    from aios.harness.memory_stores import augment_with_memory_stores
    from aios.harness.obligations import max_obligations_block_local
    from aios.harness.skills import augment_system_prompt
    from aios.services import skills as skills_service

    tools = to_openai_tools(agent.tools)
    # The switch_channel built-in is the agent's only path to mutate
    # focal attention; inject it whenever the session has bound channels.
    if channels:
        tools.append(_switch_channel_tool_spec())
    # return/error are how a session ANSWERS a request it owes — a background child
    # of a run (§3.5), OR a session-caller invoke target (#1127). The gate is owning
    # an open request edge (#1123), not child-ness: a plain foreground session that
    # was invoked owes a response and must be handed the means to give one.
    #
    # #1413: run ``get_open_obligations`` UNCONDITIONALLY (the prior background-child
    # fast-path short-circuit is gone). The obligations tail block MUST be computed
    # for background children too — their obligation is exactly what windowing
    # erases, so they are the headline beneficiary of the always-on reminder. The
    # ``return``/``error`` tool gate is preserved EXACTLY: ``owes_request`` is now
    # ``bool(obligations)``, correctness-equivalent to the old gate (the same
    # awaited anti-join), trading the fast-path for one indexed anti-join per
    # background-child step (a stated, accepted cost).
    async with pool.acquire() as conn:
        obligations = await queries.get_open_obligations(conn, session_id, account_id=account_id)
    owes_request = bool(obligations)
    if owes_request:
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
        obligations=obligations,
        obligations_block_upper_bound_local=max_obligations_block_local(obligations),
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

    Includes server description plus each enabled route's allowed HTTP
    methods, pattern, and description, so the model knows what
    ``http_request`` calls it can make. The method prefix renders the
    route's scoped verbs (``ANY`` when unrestricted) so the model does not
    attempt a verb the route gate would refuse (#828). Iteration order is
    ``agent.http_servers`` declaration order (prefix-cache-stable across
    steps).
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
                verbs = "ANY" if r.methods is None else ",".join(sorted(set(r.methods)))
                suffix = f" — {r.description}" if r.description else ""
                lines.append(f"- {verbs} {r.path_pattern}{suffix}")
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


def _stub_reasoning_content_for_thinking_target(
    messages: list[dict[str, Any]], model: str
) -> list[dict[str, Any]]:
    """Stub ``reasoning_content`` onto bare assistant turns **only** for a
    thinking-capable target.

    Gated on the same capability axis the message pipeline already computes
    (``model_descriptor(model).supports_thinking``). For a non-thinking
    target, ``_strip_to_spec`` (in ``build_messages``) has already removed
    ``reasoning_content`` from assistant turns; re-adding an empty stub here
    would contradict that strip pass, so we leave the list untouched. For a
    thinking target (DeepSeek V4 Flash, Claude family, …), the provider
    rejects replayed assistant turns lacking the field, so we stub it.

    Mutates and returns the list (the stub pass is in-place); a no-op gate
    returns the list unchanged.
    """
    # Function-local import mirrors context.build_messages to avoid an
    # import cycle with completion.py.
    from aios.harness.completion import model_descriptor

    if model_descriptor(model).supports_thinking:
        stub_missing_reasoning_content(messages)
    return messages


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
    omission: WindowOmission | None = None,
    capability_model: str | None = None,
) -> StepContext:
    """Compose the chat-completions payload for a step.

    Takes a prelude built by :func:`compute_step_prelude` and the
    windowed events slate; glues them into the final message list.

    ``capability_model`` is the model string the capability gates (vision
    inlining, extended-thinking continuity) key on (#1637): for a ``workflow:``
    model binding it is the bound workflow's declared effective model, so a bound
    model does not silently degrade those gates. Defaults to ``agent.model`` when
    not given (every raw-model caller — the gate keys on the agent's own model).

    ``pool`` + ``account_id`` back a single read-only query — the
    session's ``workspace_volume_path`` — so the renderer can resolve
    ``/workspace``-prefixed image attachments to host bytes.

    ``in_flight_tool_call_ids`` selects the pending placeholder variant
    for each unresolved tool_call. Background-executing tasks get the
    "still executing in the background" wording; everything else
    (custom, awaiting-confirm) gets the "external action" wording.
    """
    from aios.harness.channels import build_channels_tail_block
    from aios.harness.obligations import build_obligations_tail_block
    from aios.services import accounts as accounts_service
    from aios.services import sessions as sessions_service

    # The capability gates (vision inlining + thinking-block continuity) key on the
    # EFFECTIVE model — the bound workflow's declared output model for a ``workflow:``
    # binding (#1637), else ``agent.model`` unchanged.
    gate_model = capability_model if capability_model is not None else agent.model

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
        model=gate_model,
        session_id=session.id,
        workspace_path=workspace_path,
        in_flight_tool_call_ids=in_flight_tool_call_ids,
        tz_name=tz_name,
        omission=omission,
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

    # Obligations tail block (#1413): the always-on reminder of every open
    # awaited request the session owes a response to, rebuilt each step from the
    # full log (``prelude.obligations``) so it survives windowing erasure of the
    # original request user message. Appended AFTER the channels tail and BEFORE
    # the merge so an unanswered obligation is the FINAL user-role line — the
    # higher-priority stimulus (literal-minded models anchor on the last line).
    #
    # Gated on the open set being non-empty ALONE — deliberately NOT
    # ``_agent_owes_response`` (which suppresses the channels tail on a trailing
    # direct stimulus). The obligations block needs the OPPOSITE bias: an open
    # obligation IS the stimulus to act on, so it renders even after a tool
    # result (where ``build_obligations_tail_block`` returning non-None already
    # encodes "non-empty").
    obligations_block = build_obligations_tail_block(prelude.obligations, session_id=session.id)
    if obligations_block is not None:
        ctx.messages.append(obligations_block)

    # Merge consecutive user inbounds into one turn (Anthropic requires
    # alternating roles). This replaces the old "." placeholder separator,
    # which degenerate-poisoned literal models like claude-fable-5.
    messages = merge_adjacent_user_messages(ctx.messages)

    # Unblock thinking-mode targets only: DeepSeek V4 Flash and other
    # reasoning models reject replayed assistant turns that lack
    # reasoning_content.  Non-thinking targets had the field correctly
    # stripped by _strip_to_spec (build_messages); do NOT re-add it for
    # them — that re-introduces a field the strip pass just removed.
    messages = _stub_reasoning_content_for_thinking_target(messages, gate_model)

    return StepContext(
        model=agent.model,
        messages=messages,
        tools=prelude.tools,
        reacting_to=ctx.reacting_to,
        skill_versions=prelude.skill_versions,
    )
