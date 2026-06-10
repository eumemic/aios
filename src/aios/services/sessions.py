"""Business logic for sessions and their event log.

Session creation persists the workspace volume path (caller-supplied or
defaulting to ``settings.workspace_root / session_id``) and optional
per-session env vars on the row so workers can mount the correct volume
and inject environment variables at container provisioning time.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import EllipsisType
from typing import Any

import asyncpg

from aios.config import get_settings
from aios.crypto.vault import CryptoBox
from aios.db import queries
from aios.db.listen import open_listen_for_events
from aios.errors import (
    ConflictError,
    NotFoundError,
    PayloadTooLargeError,
    RateLimitedError,
    ValidationError,
)
from aios.models.agents import (
    Agent,
    AgentVersion,
    is_mcp_tool_name,
    resolve_permission,
)
from aios.models.attenuation import Surface
from aios.models.events import Event, EventKind
from aios.models.scheduled_tasks import (
    ScheduledTaskCreate,
    compute_initial_next_fire,
)
from aios.models.sessions import (
    MAX_USER_MESSAGE_CHARS,
    AwaitingToolCall,
    Session,
    SessionAwaitResponse,
    SessionResource,
    SessionResourceEcho,
    SessionStatus,
    split_resources_by_type,
)
from aios.sandbox.volumes import validate_workspace_path
from aios.services import agents as agents_service
from aios.services import github_repositories as github_repo_service
from aios.services import memory_stores as memory_service
from aios.services.await_completion import await_completion


async def load_session_account_id(pool: asyncpg.Pool[Any], session_id: str) -> str:
    """Bootstrap helper: load ``account_id`` for a session by id, no scoping.

    Used by worker / harness / tool entry points that have a ``session_id``
    but don't yet know the account context. The result is then threaded to
    every downstream query that requires ``account_id``.
    """
    async with pool.acquire() as conn:
        return await queries.unscoped_get_session_account_id(conn, session_id)


async def load_live_session_account_id(pool: asyncpg.Pool[Any], session_id: str) -> str | None:
    """``account_id`` for a live session (exists + not archived), or ``None`` if it
    has been archived/deleted — the ``run_session_step`` entry guard, so a wake for
    a gone session is an idempotent no-op rather than a crash."""
    async with pool.acquire() as conn:
        return await queries.unscoped_live_session_account_id(conn, session_id)


async def load_session_workspace_path(
    pool: asyncpg.Pool[Any], session_id: str, *, account_id: str
) -> Path:
    """Return the session's host-side workspace directory as a ``Path``.

    Reads ``sessions.workspace_volume_path`` — the authoritative bind-mount
    source for ``/workspace``. The column is internal-only (not on the
    public :class:`Session` model, since it leaks host layout), so callers
    that need it for host-path resolution fetch it explicitly via this
    helper rather than off the session object.
    """
    async with pool.acquire() as conn:
        return Path(
            await queries.get_session_workspace_path(conn, session_id, account_id=account_id)
        )


async def _list_all_echoes(
    conn: asyncpg.Connection[Any],
    session_id: str,
    *,
    account_id: str,
) -> list[SessionResourceEcho]:
    """Memory echoes first then github echoes, each in rank order."""
    memory_echoes = await queries.list_session_memory_store_echoes(
        conn, session_id, account_id=account_id
    )
    github_echoes = await queries.list_session_github_repo_echoes(
        conn, session_id, account_id=account_id
    )
    out: list[SessionResourceEcho] = []
    out.extend(memory_echoes)
    out.extend(github_echoes)
    return out


async def _batch_list_all_echoes(
    conn: asyncpg.Connection[Any],
    session_ids: list[str],
    *,
    account_id: str,
) -> dict[str, list[SessionResourceEcho]]:
    """Batched form of :func:`_list_all_echoes` — one query per echo family
    across all sessions instead of two per session."""
    if not session_ids:
        return {}
    memory_map = await queries.batch_list_session_memory_store_echoes(
        conn, session_ids, account_id=account_id
    )
    github_map = await queries.batch_list_session_github_repo_echoes(
        conn, session_ids, account_id=account_id
    )
    return {sid: [*memory_map[sid], *github_map[sid]] for sid in session_ids}


def _evict_sandbox_for_resource_change(session_id: str) -> None:
    """Force a fresh sandbox provision after a session-scoped resource
    mutation commits (#713).

    No-op in the API process (the registry global is worker-only); the
    worker process recycles so the NEXT step re-reads build_spec_from_session.
    unload_session_caches=False: a between-steps mutation re-provisions
    cleanly on the next step.

    Memory-store and github-repository bindings feed build_spec_from_session
    and MUST evict. Vault session-bindings do NOT feed the sandbox spec (they
    reach the agent via the MCP pool, keyed on (url, vault_id)); we evict
    anyway as defense-in-depth so 'any session-resource mutation forces a
    clean re-read' holds — harmless, one extra cold-start. In-place vault
    credential rotation (refresh_credential / ciphertext overwrite) does NOT
    evict: the pool keys on (url, vault_id) and a rotation overwrites the row
    contents, so the stable key already serves the new secret. Layer 2's
    spec_version triggers are deliberately NOT placed on vault/connection
    tables (they don't change the spec) — this Layer1/Layer2 asymmetry is
    intentional.
    """
    from aios.harness import runtime

    if runtime.sandbox_registry is not None:
        runtime.sandbox_registry.evict(session_id, unload_session_caches=False)


async def create_session(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    agent_id: str,
    environment_id: str,
    agent_version: int | None = None,
    title: str | None,
    metadata: dict[str, Any],
    vault_ids: list[str] | None = None,
    resources: list[SessionResource] | None = None,
    scheduled_tasks: list[ScheduledTaskCreate] | None = None,
    crypto_box: CryptoBox | None = None,
    workspace_path: str | None = None,
    env: dict[str, str] | None = None,
    focal_channel: str | None = None,
    focal_locked: bool = False,
    archive_when_idle: bool = False,
) -> Session:
    """Create a session row and return it.

    ``agent_version=None`` means "latest" — the session will always use
    whatever version of the agent is current at step time. Resource
    attachment runs in the same transaction as the session insert so a
    failed attach (e.g. archived store, name collision) leaves no
    orphaned session.

    ``focal_channel`` + ``focal_locked`` are written atomically with
    the row insert so ``switch_channel``'s focal-locked invariant
    holds from creation.  Per-chat-spawned sessions pass
    ``focal_locked=True``.

    ``crypto_box`` is required when ``resources`` includes any
    ``github_repository`` entries (their auth tokens are encrypted on
    insert). Memory-store-only attachments don't need it.
    """
    if workspace_path is not None:
        validate_workspace_path(workspace_path, account_id)
    # No sandbox eviction here (#713): a brand-new session has no cached
    # sandbox to recycle — its first step cold-starts from the freshly
    # written spec. Eviction is only meaningful for mutations to an
    # already-provisioned session (see update_session / connections).
    async with pool.acquire() as conn, conn.transaction():
        # Validate the environment is account-owned before binding the session to
        # it. A bare FK would accept another tenant's env id and leak its image /
        # env-vars / networking into this session — mirrors create_run (issue #755).
        await queries.get_environment(conn, environment_id, account_id=account_id)
        session = await queries.insert_session(
            conn,
            agent_id=agent_id,
            environment_id=environment_id,
            agent_version=agent_version,
            title=title,
            metadata=metadata,
            workspace_path=workspace_path,
            env=env,
            focal_channel=focal_channel,
            focal_locked=focal_locked,
            archive_when_idle=archive_when_idle,
            account_id=account_id,
        )
        if vault_ids:
            await queries.set_session_vaults(conn, session.id, vault_ids, account_id=account_id)
            session = session.model_copy(update={"vault_ids": vault_ids})
        if resources:
            memory_resources, github_resources = split_resources_by_type(resources)
            if memory_resources:
                await memory_service.attach_to_session(
                    conn, session.id, memory_resources, account_id=account_id
                )
            if github_resources:
                assert crypto_box is not None, (
                    "API surface requires CryptoBox when attaching github_repository"
                )
                await github_repo_service.attach_to_session(
                    conn, session.id, github_resources, crypto_box, account_id=account_id
                )
            echoes = await _list_all_echoes(conn, session.id, account_id=account_id)
            session = session.model_copy(update={"resources": echoes})
        if scheduled_tasks:
            now = datetime.now(UTC)
            enabled_new = sum(1 for spec in scheduled_tasks if spec.enabled)
            # Take the per-account advisory lock for the duration of the
            # count + batch INSERT so concurrent session creates against
            # the same account can't race past the cap. The lock is
            # transaction-scoped, released on COMMIT/ROLLBACK.
            await queries.acquire_account_scheduled_tasks_lock(conn, account_id)
            if enabled_new:
                cap = get_settings().scheduled_tasks_per_account_max
                existing = await queries.count_account_scheduled_tasks(
                    conn, account_id=account_id, enabled_only=True
                )
                if existing + enabled_new > cap:
                    raise RateLimitedError(
                        f"account at active-timer cap ({existing}/{cap}); the "
                        f"{enabled_new} enabled scheduled task(s) in this session "
                        "would exceed the cap — disable some entries or remove an "
                        "older session's tasks first"
                    )
            for spec in scheduled_tasks:
                next_fire = (
                    compute_initial_next_fire(spec.schedule, spec.fire_at, now)
                    if spec.enabled
                    else None
                )
                await queries.add_scheduled_task(
                    conn,
                    session.id,
                    name=spec.name,
                    schedule=spec.schedule,
                    fire_at=spec.fire_at,
                    command=spec.command,
                    enabled=spec.enabled,
                    timeout_seconds=spec.timeout_seconds,
                    max_output_bytes=spec.max_output_bytes,
                    metadata=spec.metadata,
                    next_fire=next_fire,
                    account_id=account_id,
                )
            task_echoes = await queries.list_scheduled_tasks(
                conn, session.id, account_id=account_id
            )
            session = session.model_copy(update={"scheduled_tasks": task_echoes})
        return session


async def create_child_session(
    pool: asyncpg.Pool[Any],
    *,
    session_id: str,
    account_id: str,
    agent_id: str,
    environment_id: str,
    agent_version: int,
    parent_run_id: str,
    surface: Surface,
    vault_ids: list[str],
    request_id: str,
    input: Any,
    output_schema: dict[str, Any] | None = None,
) -> bool:
    """Idempotently spawn a workflow ``agent()`` child and inject the first request.

    `invoke_agent` = create_session + invoke_session: the child's first user
    message **is a request** — its content is ``input``, and ``metadata.request``
    carries ``{request_id, caller}`` so the target can correlate its response and
    the caller (the run) can be resumed. The child must answer this request via
    ``return``/``error`` (exactly once); the ``request_id`` is surfaced to the model
    as a render-time marker on the message (see ``render_user_event``) so it knows
    which id to echo back.

    ``output_schema`` (optional) is the JSON Schema the request demands of the
    response ``value``. It rides ``metadata.request.output_schema`` — per-request, so
    a child owing several requests can carry a distinct schema for each — surfaced to
    the model alongside the request and enforced when it calls ``return``.

    ``surface`` is the child's **frozen, run-attenuated** capability surface
    (``attenuate(agent, run)`` — #794); ``vault_ids`` is the run's vault bindings,
    copied into the child's ``session_vaults`` so it resolves credentials off its own
    (subset) table. Both are written **only on a real insert**, inside the one
    transaction, so a replay never re-freezes a shifted surface or re-binds vaults.

    One transaction: insert the child row (``ON CONFLICT (id) DO NOTHING``) and,
    **only on a real insert**, freeze the surface, bind the vaults, and deliver the
    request — without a stimulus the child would be born-idle. Atomic, so a crash can
    never leave a child row without its request. Returns ``True`` on first spawn,
    ``False`` on conflict (replay → the caller harvests the response instead of
    re-spawning).
    """
    content = input if isinstance(input, str) else json.dumps(input)
    async with pool.acquire() as conn, conn.transaction():
        child = await queries.insert_child_session(
            conn,
            session_id=session_id,
            account_id=account_id,
            agent_id=agent_id,
            environment_id=environment_id,
            agent_version=agent_version,
            parent_run_id=parent_run_id,
            tools=surface.tools,
            mcp_servers=surface.mcp_servers,
            http_servers=surface.http_servers,
        )
        if child is None:
            return False  # replay: row exists — do NOT re-deliver the request
        if vault_ids:
            await queries.set_session_vaults(conn, session_id, vault_ids, account_id=account_id)
        request_meta: dict[str, Any] = {
            "request_id": request_id,
            "caller": {"kind": "run", "id": parent_run_id},
        }
        if output_schema is not None:
            request_meta["output_schema"] = output_schema
        await queries.append_event(
            conn,
            account_id=account_id,
            session_id=session_id,
            kind="message",
            data={
                "role": "user",
                "content": content,
                "metadata": {"request": request_meta},
            },
        )
        return True


def _classify_awaiting(
    tc: dict[str, Any],
    agent: Agent | AgentVersion,
) -> AwaitingToolCall | None:
    """Classify an unresolved tool_call into an ``AwaitingToolCall``, or
    ``None`` if the session is not blocked on external action for it
    (``always_allow`` builtin/mcp, or ``always_ask`` already confirmed).

    Mirrors :func:`aios.harness.loop._classify_tool_call`'s dispatch
    rules — including arg-aware refinement via
    ``ToolDefinition.classify_permission`` — so the view stays
    consistent with what the harness will do.
    """
    # Late import: aios.tools package init pulls in services.wake which
    # imports this module.
    from aios.tools.invoke import parse_arguments
    from aios.tools.registry import registry as tool_registry

    name = tc["name"]
    tool_call_id = tc["tool_call_id"]
    has_allow_lifecycle = tc["has_allow_lifecycle"]

    if is_mcp_tool_name(name):
        if (
            agents_service.effective_mcp_permission(name, agent.tools) == "always_ask"
            and not has_allow_lifecycle
        ):
            return AwaitingToolCall(tool_call_id=tool_call_id, name=name, kind="mcp")
        return None
    if tool_registry.has(name):
        perm_tool = resolve_permission(name, agent.tools)
        perm_route: str | None = None
        tool_def = tool_registry.get(name)
        if tool_def.classify_permission is not None:
            args = parse_arguments(tc.get("arguments"))
            if args is not None:
                perm_route = tool_def.classify_permission(args, agent)
        if (perm_tool == "always_ask" or perm_route == "always_ask") and not has_allow_lifecycle:
            return AwaitingToolCall(tool_call_id=tool_call_id, name=name, kind="builtin")
        return None
    # Unknown name: client-executed custom tool.
    return AwaitingToolCall(tool_call_id=tool_call_id, name=name, kind="custom")


async def compute_awaiting(
    pool: asyncpg.Pool[Any], sessions: list[Session], *, account_id: str
) -> dict[str, list[AwaitingToolCall]]:
    """Compute the ``awaiting`` view for a batch of sessions.

    One SQL round-trip for unresolved tool_calls across the batch, then
    one agent load per distinct (agent_id, agent_version) pair, then
    in-memory classification.
    """
    if not sessions:
        return {}
    async with pool.acquire() as conn:
        unresolved_by_sid = await queries.list_unresolved_tool_calls_batch(
            conn, [s.id for s in sessions], account_id=account_id
        )
    if not unresolved_by_sid:
        return {}
    # Foreground sessions sharing an (agent_id, agent_version) share a surface, so they
    # dedupe the agent load. Workflow children do NOT: each carries its own frozen,
    # run-attenuated surface (#794), so two children of the same agent/version under
    # different runs would collide on the old key and misclassify authority — they key
    # per-session instead.
    agent_cache: dict[str | tuple[str, int | None], Agent | AgentVersion] = {}
    out: dict[str, list[AwaitingToolCall]] = {}
    for session in sessions:
        unresolved = unresolved_by_sid.get(session.id)
        if not unresolved:
            continue
        key: str | tuple[str, int | None] = (
            session.id
            if session.parent_run_id is not None
            else (session.agent_id, session.agent_version)
        )
        if key not in agent_cache:
            agent_cache[key] = await agents_service.load_for_session(
                pool, session, account_id=account_id
            )
        agent = agent_cache[key]
        entries: list[AwaitingToolCall] = []
        for tc in unresolved:
            classified = _classify_awaiting(tc, agent)
            if classified is not None:
                entries.append(classified)
        if entries:
            out[session.id] = entries
    return out


async def get_session(pool: asyncpg.Pool[Any], session_id: str, *, account_id: str) -> Session:
    # REPEATABLE READ snapshot pins all three enrichment reads to one
    # moment in time so a concurrent ``update_session`` committing
    # between reads can't return a torn ``Session`` (e.g. row columns
    # from before the update with ``resources`` from after).
    # ``compute_awaiting`` is event-derived and monotonic — running it on
    # a separate connection sees ≥ the snapshot's events, not a tear.
    async with pool.acquire() as conn, conn.transaction(isolation="repeatable_read", readonly=True):
        session = await queries.get_session(conn, session_id, account_id=account_id)
        vault_ids = await queries.get_session_vault_ids(conn, session_id, account_id=account_id)
        echoes = await _list_all_echoes(conn, session_id, account_id=account_id)
        task_echoes = await queries.list_scheduled_tasks(conn, session_id, account_id=account_id)
    awaiting_by_sid = await compute_awaiting(pool, [session], account_id=account_id)
    return session.model_copy(
        update={
            "vault_ids": vault_ids,
            "resources": echoes,
            "scheduled_tasks": task_echoes,
            "awaiting": awaiting_by_sid.get(session_id, []),
        }
    )


async def await_session(
    pool: asyncpg.Pool[Any],
    db_url: str,
    session_id: str,
    *,
    account_id: str,
    request_id: str | None,
    watermark: int | None,
    timeout_seconds: float,
) -> SessionAwaitResponse:
    """Block until a correlated response lands (request_id mode) or the session has fully
    reacted to a fixed stimulus (watermark mode; watermark defaults to last_stimulus_seq
    captured at call time), or timeout. The session backing of the await primitive.

    Account-scopes the session FIRST (cross-tenant/missing 404s before any LISTEN opens),
    then subscribes to events_<session_id> BEFORE the first predicate read (LISTEN-before-read),
    drives await_completion with the monotonic done-predicate, and returns the completion
    envelope — or, on timeout, done=False so the caller re-polls. Never waits on bare idle:
    both modes are monotonic w.r.t. a fixed stimulus (request_id, or reacted>=watermark).
    """
    if request_id is not None and watermark is not None:
        raise ValidationError("provide request_id or watermark, not both")

    # Scope-check FIRST (404s cross-tenant before any LISTEN opens) and capture the
    # default watermark's ``last_stimulus_seq`` in the same acquire.
    async with pool.acquire() as conn:
        await queries.get_session(conn, session_id, account_id=account_id)  # 404s cross-tenant
        captured = await queries.read_session_watermarks(conn, session_id, account_id=account_id)
    captured_last_stimulus_seq = captured[1] if captured is not None else 0
    effective_watermark = watermark if watermark is not None else captured_last_stimulus_seq

    if request_id is not None:

        async def _read() -> Any:
            async with pool.acquire() as conn:
                return await queries.derive_response(
                    conn, session_id, account_id=account_id, request_id=request_id
                )

        def _is_done(state: Any) -> bool:
            return state is not None  # a response (or child_gone) has landed
    else:

        async def _read() -> Any:
            async with pool.acquire() as conn:
                wm = await queries.read_session_watermarks(conn, session_id, account_id=account_id)
                return wm[0] if wm is not None else 0  # last_reacted_seq

        def _is_done(state: Any) -> bool:
            return bool(state >= effective_watermark)

    subscription = await open_listen_for_events(db_url, session_id)
    try:
        state = await await_completion(
            subscription.queue,
            read_state=_read,
            is_done=_is_done,
            timeout_seconds=timeout_seconds,
        )
    finally:
        subscription.terminate()

    async with pool.acquire() as conn:
        wm = await queries.read_session_watermarks(conn, session_id, account_id=account_id)
    last_reacted_seq = wm[0] if wm is not None else 0

    if request_id is not None:
        if state is not None:
            return SessionAwaitResponse(
                done=True,
                last_reacted_seq=last_reacted_seq,
                result=state["result"],
                is_error=state["is_error"],
                error=state["error"],
            )
        return SessionAwaitResponse(done=False, last_reacted_seq=last_reacted_seq)

    return SessionAwaitResponse(
        done=last_reacted_seq >= effective_watermark, last_reacted_seq=last_reacted_seq
    )


async def get_session_basic(
    pool: asyncpg.Pool[Any], session_id: str, *, account_id: str
) -> Session:
    """Bare session read with no enrichment (no vaults / resources / awaiting).

    Use this when the caller reads only core columns — the worker step
    path (``agent_id``, ``agent_version``, ``focal_channel``) and the
    long-poll wait endpoint's existence check. Skips the ``status``
    derivation (``status`` defaults to ``idle`` and is never surfaced here).
    """
    async with pool.acquire() as conn:
        return await queries.get_session_bare(conn, session_id, account_id=account_id)


async def get_session_event_stats(
    pool: asyncpg.Pool[Any], session_id: str, *, account_id: str
) -> tuple[int, datetime | None]:
    async with pool.acquire() as conn:
        return await queries.get_session_event_stats(conn, session_id, account_id=account_id)


async def get_session_model(pool: asyncpg.Pool[Any], session_id: str, *, account_id: str) -> str:
    """Bound model for ``session_id`` (pinned agent version wins)."""
    async with pool.acquire() as conn:
        return await queries.get_session_model(conn, session_id, account_id=account_id)


async def list_sessions(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    agent_id: str | None = None,
    status: SessionStatus | None = None,
    parent_run_id: str | None = None,
    limit: int = 50,
    after: str | None = None,
) -> list[Session]:
    # See ``get_session`` for the rationale on the snapshot wrap.
    async with pool.acquire() as conn, conn.transaction(isolation="repeatable_read", readonly=True):
        sessions = await queries.list_sessions(
            conn,
            agent_id=agent_id,
            status=status,
            parent_run_id=parent_run_id,
            limit=limit,
            after=after,
            account_id=account_id,
        )
        if not sessions:
            return sessions
        sid_list = [s.id for s in sessions]
        vault_map = await queries.batch_get_session_vault_ids(conn, sid_list, account_id=account_id)
        echoes_map = await _batch_list_all_echoes(conn, sid_list, account_id=account_id)
        task_map = await queries.batch_list_session_scheduled_tasks(
            conn, sid_list, account_id=account_id
        )
    awaiting_by_sid = await compute_awaiting(pool, sessions, account_id=account_id)
    enriched: list[Session] = [
        s.model_copy(
            update={
                "vault_ids": vault_map[s.id],
                "resources": echoes_map[s.id],
                "scheduled_tasks": task_map[s.id],
                "awaiting": awaiting_by_sid.get(s.id, []),
            }
        )
        for s in sessions
    ]
    return enriched


async def append_user_message(
    pool: asyncpg.Pool[Any],
    session_id: str,
    content: str,
    metadata: dict[str, Any] | None = None,
    *,
    account_id: str,
) -> Event:
    """Append a `role: user` message event to the session log.

    When the inbound path stamps ``metadata["channel"]`` (the connector's
    full channel address), we lift it into the event's ``orig_channel``
    column so the context builder and unread-derivation helpers can key
    off it directly — without re-parsing a JSONB blob on every read.
    """
    if len(content) > MAX_USER_MESSAGE_CHARS:
        raise PayloadTooLargeError(
            f"user message exceeds {MAX_USER_MESSAGE_CHARS:,} characters "
            f"(got {len(content):,}); split into multiple messages",
            detail={"max_chars": MAX_USER_MESSAGE_CHARS, "got_chars": len(content)},
        )
    data: dict[str, Any] = {"role": "user", "content": content}
    if metadata:
        data["metadata"] = metadata
    orig_channel: str | None = None
    if metadata is not None:
        channel = metadata.get("channel")
        if isinstance(channel, str):
            orig_channel = channel
    async with pool.acquire() as conn:
        return await queries.append_event(
            conn,
            session_id=session_id,
            kind="message",
            data=data,
            orig_channel=orig_channel,
            account_id=account_id,
        )


async def append_event(
    pool: asyncpg.Pool[Any],
    session_id: str,
    kind: EventKind,
    data: dict[str, Any],
    *,
    account_id: str,
) -> Event:
    """Append an arbitrary event. Used by the harness loop."""
    async with pool.acquire() as conn:
        return await queries.append_event(
            conn, session_id=session_id, kind=kind, data=data, account_id=account_id
        )


# A session gets this many nudges to answer an owed request before the harness
# answers it on the model's behalf with a ``no_return`` error. Derived per request
# from the count of nudge messages (see ``queries.count_request_nudges``).
REQUEST_NUDGE_BUDGET = 3


def _nudge_content(request_ids: list[str]) -> str:
    ids = ", ".join(request_ids)
    return (
        f"You still owe a response to these request(s): {ids}. Answer each one with "
        "return(request_id, value) if you have a result, or error(request_id, message) "
        "if you can't complete it — you must answer before you can finish."
    )


async def append_assistant_and_guard_quiescence(
    pool: asyncpg.Pool[Any],
    session_id: str,
    assistant_msg: dict[str, Any],
    *,
    account_id: str,
    parent_run_id: str | None,
) -> tuple[bool, str | None]:
    """Append the model's assistant message and, **atomically**, enforce the
    request-totality invariant: a session may not go idle while it owes a response.

    Only a workflow child (``parent_run_id`` set) can ever owe a request, so for any
    other session this is a plain assistant append — the totality machinery (and its
    per-turn open-request scan) is skipped entirely.

    For a workflow child, in one transaction: append the assistant message; if the
    session would now be idle (a tool-call-free turn — ``derive_session_status``
    reads the just-appended event) and it still owes request responses, then for
    each open request either append a **nudge** (a user message that re-triggers
    inference, so the session stays active) when under :data:`REQUEST_NUDGE_BUDGET`,
    or write its **no_return** error response when the budget is spent. Because the
    nudge commits in the same transaction as the idling assistant event, no reader
    can ever observe the session idle while a request is open — the invariant holds
    at write time, with no sweep backstop.

    Returns ``(nudged, autoerror_caller_run_id)``; the caller (the harness loop) does
    the post-commit wakes — ``defer_wake(session)`` if nudged,
    ``defer_run_wake(run_id)`` if a run id is returned (a request was auto-errored
    and its caller run must harvest the no_return). The run id is the child's
    ``parent_run_id`` — the single caller of every request in v1.
    """
    nudged = False
    autoerror_caller_run_id: str | None = None
    async with pool.acquire() as conn, conn.transaction():
        await queries.append_event(
            conn, session_id=session_id, kind="message", data=assistant_msg, account_id=account_id
        )
        # Gates, cheapest first (this runs on EVERY end-of-turn append):
        #   1. not a workflow child → cannot owe a request → plain append, done.
        #   2. tool calls present → unresolved tool_call → active by construction
        #      (the tools haven't run yet), so it can't be idle — settle in memory.
        #   3. nothing owed → every request already answered → one indexed anti-join.
        #   4. only now pay the multi-EXISTS idleness derivation (the same
        #      _SESSION_STATUS_EXPR every external reader uses) — it could still be
        #      active via a user message that arrived during inference.
        if parent_run_id is None or assistant_msg.get("tool_calls"):
            return (False, None)
        open_ids = await queries.get_open_request_ids(conn, session_id, account_id=account_id)
        if not open_ids:
            return (False, None)  # every request answered
        if await queries.derive_session_status(conn, session_id, account_id=account_id) != "idle":
            return (False, None)
        to_nudge: list[str] = []
        for request_id in open_ids:
            nudges = await queries.count_request_nudges(
                conn, session_id, account_id=account_id, request_id=request_id
            )
            if nudges >= REQUEST_NUDGE_BUDGET:
                if await queries.write_response_if_absent(
                    conn,
                    session_id,
                    account_id=account_id,
                    request_id=request_id,
                    is_error=True,
                    result=None,
                    error={"kind": "no_return"},
                ):
                    autoerror_caller_run_id = parent_run_id
            else:
                to_nudge.append(request_id)
        if to_nudge:
            await queries.append_event(
                conn,
                session_id=session_id,
                kind="message",
                data={
                    "role": "user",
                    "content": _nudge_content(to_nudge),
                    "metadata": {"nudged_request_ids": to_nudge},
                },
                account_id=account_id,
            )
            nudged = True
    return (nudged, autoerror_caller_run_id)


async def append_tool_result(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    session_id: str,
    tool_call_id: str,
    content: str | list[dict[str, Any]],
    is_error: bool = False,
) -> Event:
    """Append a tool-role event for a custom tool call (#133).

    Idempotent on ``(session_id, tool_call_id)``: a retried POST (network
    blip, 502, mid-flight client disconnect) returns the existing event
    instead of appending a duplicate that would violate the
    monotonic-context invariant (CLAUDE.md #2) by silently rewriting the
    model's view of an earlier turn. Mirrors the ``WHERE status =
    'pending'`` guard in :func:`queries.mark_management_call_resolved`.

    Stamps the tool's ``name`` from the parent assistant's ``tool_calls``
    array so the derived ``tool_name`` column on ``events`` stays
    populated for custom tools.  Raises :class:`NotFoundError` if there's
    no parent — a result with no matching call would leave an orphan row.

    Takes a connection (not a pool) so the caller can group additional
    work in the same transaction (e.g. a connection-binding auth check
    in the connector-facing endpoint).  The caller is responsible for
    deferring the wake afterwards.
    """
    from aios.sandbox.tool_result_spill import cap_tool_result_content

    if isinstance(content, str):
        content = await cap_tool_result_content(
            session_id, tool_call_id, content, max_chars=get_settings().tool_result_max_chars
        )
    async with conn.transaction():
        await queries.lock_active_session_for_update(conn, session_id, account_id=account_id)
        existing = await queries.find_tool_result_event(
            conn, session_id, tool_call_id, account_id=account_id
        )
        if existing is not None:
            # Idempotent retry: a prior call with matching intent
            # (same ``is_error`` outcome) — return the original event.
            # Intent-mismatch is a CONFLICT, not a retry: e.g., a deny
            # arriving after the tool already produced a success result
            # (two-tab race; always_allow tool; bogus pre-confirm
            # pre-#533). Returning the success event would lie to the
            # operator ("you denied successfully") while the model's
            # context still carries the success result. Raise so the
            # operator learns the deny is too late.
            existing_is_error = bool(existing.data.get("is_error", False))
            if existing_is_error == is_error:
                await queries.decrement_open_tool_call_count(
                    conn, session_id, account_id=account_id
                )
                return existing
            raise ConflictError(
                f"tool_call_id {tool_call_id!r} already has a "
                f"{'error' if existing_is_error else 'success'} "
                f"result; cannot append a "
                f"{'error' if is_error else 'success'} result with "
                f"the same tool_call_id",
                detail={
                    "session_id": session_id,
                    "tool_call_id": tool_call_id,
                    "existing_is_error": existing_is_error,
                    "requested_is_error": is_error,
                },
            )

        name = await queries.lookup_tool_name_by_call_id(
            conn, session_id, tool_call_id, account_id=account_id
        )
        if name is None:
            raise NotFoundError(
                f"tool_call_id {tool_call_id!r} not found",
                detail={"session_id": session_id, "tool_call_id": tool_call_id},
            )
        data: dict[str, Any] = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
            "name": name,
        }
        if is_error:
            data["is_error"] = True
        return await queries.append_event(
            conn,
            session_id=session_id,
            kind="message",
            data=data,
            account_id=account_id,
        )


async def read_events(
    pool: asyncpg.Pool[Any],
    session_id: str,
    *,
    account_id: str,
    after_seq: int = 0,
    before: int | None = None,
    kind: EventKind | None = None,
    limit: int = 200,
    newest_first: bool = False,
    error_only: bool = False,
) -> list[Event]:
    async with pool.acquire() as conn:
        return await queries.read_events(
            conn,
            session_id,
            after_seq=after_seq,
            before=before,
            kind=kind,
            limit=limit,
            newest_first=newest_first,
            account_id=account_id,
            error_only=error_only,
        )


async def list_confirmed_unresolved_tool_calls(
    pool: asyncpg.Pool[Any],
    session_id: str,
    *,
    account_id: str,
) -> list[dict[str, Any]]:
    """Dispatchable ``tool_call`` dicts for operator-confirmed tools in a
    session that have no result yet, sourced unwindowed from the log.

    One indexed query (see :func:`queries.list_confirmed_unresolved_tool_calls`)
    that mirrors the sweep's case-(c) wake predicate, so a confirmed
    ``always_ask`` tool whose parent assistant has scrolled out of the token
    window (#737) — or simply isn't the latest assistant — is still recovered,
    and one that already has a result is not re-dispatched (invariant #4).

    Passes ``settings.confirmed_dispatch_max_age_seconds`` as the age bound on
    the CONFIRM event: a confirmation older than that is skipped, so a
    weeks-stale confirmed side-effecting call is not re-dispatched on a worker
    restart (#746). This path is dispatch-only (no read-model caller), so the
    bound is always applied here; it stays in sync with the sweep's detection
    predicate (``sweep.CONFIRMED_ROWS_SQL``), which reads the same setting.
    """
    max_age_seconds = get_settings().confirmed_dispatch_max_age_seconds
    async with pool.acquire() as conn:
        return await queries.list_confirmed_unresolved_tool_calls(
            conn, session_id, account_id=account_id, max_age_seconds=max_age_seconds
        )


async def get_event(
    pool: asyncpg.Pool[Any], session_id: str, event_id: str, *, account_id: str
) -> Event:
    async with pool.acquire() as conn:
        return await queries.get_event(conn, session_id, event_id, account_id=account_id)


async def read_message_events(
    pool: asyncpg.Pool[Any], session_id: str, *, account_id: str
) -> list[Event]:
    async with pool.acquire() as conn:
        return await queries.read_message_events(conn, session_id, account_id=account_id)


async def read_windowed_events(
    pool: asyncpg.Pool[Any],
    session_id: str,
    *,
    account_id: str,
    window_min: int,
    window_max: int,
    model: str,
    overhead_local: int,
) -> list[Event]:
    async with pool.acquire() as conn:
        return await queries.read_windowed_events(
            conn,
            session_id,
            window_min=window_min,
            window_max=window_max,
            model=model,
            overhead_local=overhead_local,
            account_id=account_id,
        )


async def set_session_stop_reason(
    pool: asyncpg.Pool[Any],
    session_id: str,
    stop_reason: dict[str, Any],
    *,
    account_id: str,
) -> None:
    """Record why the most recent step ended (end_turn/error/interrupt/
    rescheduling). ``status`` is derived from the event log, so this no longer
    writes a status column."""
    async with pool.acquire() as conn:
        await queries.set_session_stop_reason(conn, session_id, stop_reason, account_id=account_id)


async def reclaim_session_if_idle(
    pool: asyncpg.Pool[Any], session_id: str, *, account_id: str
) -> bool:
    """Pool wrapper for :func:`queries.reclaim_session_if_idle` — soft-archive iff idle."""
    async with pool.acquire() as conn:
        return await queries.reclaim_session_if_idle(conn, session_id, account_id=account_id)


async def increment_usage(
    pool: asyncpg.Pool[Any],
    session_id: str,
    *,
    account_id: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
) -> None:
    """Atomically add token counts to a session's cumulative usage."""
    async with pool.acquire() as conn:
        await queries.increment_session_usage(
            conn,
            session_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            account_id=account_id,
        )


async def archive_session(pool: asyncpg.Pool[Any], session_id: str, *, account_id: str) -> Session:
    # Enrich vault_ids / resources / scheduled_tasks so the API response
    # shape matches GET /sessions/{id}. Archive itself is a single column
    # flip; the lists are read post-update to surface any concurrent
    # mutation that committed before archive landed.
    async with pool.acquire() as conn:
        session = await queries.archive_session(conn, session_id, account_id=account_id)
        vault_ids = await queries.get_session_vault_ids(conn, session_id, account_id=account_id)
        echoes = await _list_all_echoes(conn, session_id, account_id=account_id)
        task_echoes = await queries.list_scheduled_tasks(conn, session_id, account_id=account_id)
    return session.model_copy(
        update={
            "vault_ids": vault_ids,
            "resources": echoes,
            "scheduled_tasks": task_echoes,
        }
    )


async def clone_session(
    pool: asyncpg.Pool[Any],
    parent_session_id: str,
    *,
    account_id: str,
    workspace_path: str | None = None,
) -> Session:
    """Clone a session — see :func:`queries.clone_session`."""
    if workspace_path is not None:
        validate_workspace_path(workspace_path, account_id)
    async with pool.acquire() as conn:
        session = await queries.clone_session(
            conn, parent_session_id, workspace_path=workspace_path, account_id=account_id
        )
        vault_ids = await queries.get_session_vault_ids(conn, session.id, account_id=account_id)
        echoes = await _list_all_echoes(conn, session.id, account_id=account_id)
        task_echoes = await queries.list_scheduled_tasks(conn, session.id, account_id=account_id)
        return session.model_copy(
            update={
                "vault_ids": vault_ids,
                "resources": echoes,
                "scheduled_tasks": task_echoes,
            }
        )


async def delete_session(pool: asyncpg.Pool[Any], session_id: str, *, account_id: str) -> None:
    async with pool.acquire() as conn:
        await queries.delete_session(conn, session_id, account_id=account_id)


async def update_session(
    pool: asyncpg.Pool[Any],
    session_id: str,
    *,
    account_id: str,
    agent_id: str | None = None,
    agent_version: int | None | EllipsisType = ...,
    title: str | None | EllipsisType = ...,
    metadata: dict[str, Any] | None = None,
    vault_ids: list[str] | None = None,
    resources: list[SessionResource] | None = None,
    crypto_box: CryptoBox | None = None,
) -> Session:
    # One transaction so a 4xx from resource attach (e.g. name collision)
    # rolls back the earlier title/agent/vault writes.
    async with pool.acquire() as conn, conn.transaction():
        session = await queries.update_session(
            conn,
            session_id,
            agent_id=agent_id,
            agent_version=agent_version,
            title=title,
            metadata=metadata,
            account_id=account_id,
        )
        changed = False
        if vault_ids is not None:
            old_vault_ids = await queries.get_session_vault_ids(
                conn, session_id, account_id=account_id
            )
            if vault_ids != old_vault_ids:
                await queries.set_session_vaults(conn, session_id, vault_ids, account_id=account_id)
                changed = True
        if resources is not None:
            # Wire-level semantics is full-list-replace across all
            # resource types, so an incoming list that omits a type
            # detaches every existing attachment of that type.
            memory_resources, github_resources = split_resources_by_type(resources)
            changed |= await memory_service.set_session_resources(
                conn, session_id, memory_resources, account_id=account_id
            )
            if github_resources:
                assert crypto_box is not None, (
                    "API surface requires CryptoBox when attaching github_repository"
                )
                changed |= await github_repo_service.set_session_resources(
                    conn, session_id, github_resources, crypto_box, account_id=account_id
                )
            else:
                changed |= await github_repo_service.detach_all_from_session(
                    conn, session_id, account_id=account_id
                )
        vids = await queries.get_session_vault_ids(conn, session_id, account_id=account_id)
        echoes = await _list_all_echoes(conn, session_id, account_id=account_id)
        task_echoes = await queries.list_scheduled_tasks(conn, session_id, account_id=account_id)
        result = session.model_copy(
            update={
                "vault_ids": vids,
                "resources": echoes,
                "scheduled_tasks": task_echoes,
            }
        )

    # Eviction fires AFTER the transaction commits — recycling mid-transaction
    # could re-provision against an uncommitted spec (#713). A resource or
    # vault-binding change forces the worker to re-read build_spec_from_session
    # on the next step; no-op in the API process (registry global is
    # worker-only). An idempotent re-PUT (same vaults, same resources) writes
    # nothing and must not recycle the sandbox.
    if changed:
        _evict_sandbox_for_resource_change(session_id)
    return result


# ─── tool confirmations ────────────────────────────────────────────────────


def _find_tool_call(events: list[Event], tool_call_id: str) -> dict[str, Any] | None:
    """Find a tool call dict by its id in the session's message events.

    Scans assistant messages in reverse order and returns the raw tool_call
    dict matching ``tool_call_id``, or ``None`` if not found.
    """
    for e in reversed(events):
        if e.kind != "message" or e.data.get("role") != "assistant":
            continue
        for tc in e.data.get("tool_calls") or []:
            if tc.get("id") == tool_call_id:
                result: dict[str, Any] = tc
                return result
    return None


async def confirm_tool_allow(
    pool: asyncpg.Pool[Any],
    session_id: str,
    tool_call_id: str,
    *,
    account_id: str,
) -> Event:
    """Record an allow confirmation for an ``always_ask`` tool call.

    Appends a ``lifecycle/tool_confirmed`` event. The worker's step
    function will see this and dispatch the tool call.

    Idempotent on ``(session_id, tool_call_id)``: a retried POST
    (network blip, 502, mid-flight client disconnect, double-click)
    returns the original event instead of appending a duplicate. Mirrors
    the deny twin's same-event-shape dedup (#447); ``_dispatch_confirmed_tools``
    further set-deduplicates so the tool dispatches once even pre-fix,
    but the lifecycle event log must not accumulate duplicate rows.

    Rejects with :class:`ConflictError` when the tool call already has
    a tool-role result event — a prior deny or a racing dispatch
    pinned the model's view; appending a lifecycle ``allow`` here would
    be a phantom no-op surfaced as 201 (operator UI shows "allow
    accepted" while the model has moved on with the prior result).
    Symmetric twin to #535 (``confirm_tool_deny`` rejected when tool
    already succeeded); same defect class — confirm endpoints must not
    silently accept impossible inputs.
    """
    async with pool.acquire() as conn, conn.transaction():
        await queries.lock_active_session_for_update(conn, session_id, account_id=account_id)
        existing = await queries.find_tool_confirmed_event(
            conn, session_id, tool_call_id, account_id=account_id
        )
        if existing is not None:
            return existing
        # Validate the tool_call_id corresponds to a real assistant
        # ``tool_calls`` entry in the session's event log. Without this,
        # any authenticated POST to /v1/sessions/:id/tool-confirmations
        # with decision=allow appends an arbitrary
        # ``lifecycle/tool_confirmed`` row — poisoning the event log
        # and (if the bogus id later collides with a real provider-
        # generated tool_call_id) pre-confirming a tool the operator
        # never saw. Mirrors the deny path's existing call to
        # ``lookup_tool_name_by_call_id`` (#445) — same defense, same
        # error surface, restores the asymmetric validation gap
        # between allow and deny.
        if (
            await queries.lookup_tool_name_by_call_id(
                conn, session_id, tool_call_id, account_id=account_id
            )
            is None
        ):
            raise NotFoundError(
                f"tool_call_id {tool_call_id!r} not found",
                detail={"session_id": session_id, "tool_call_id": tool_call_id},
            )
        # Reject when the tool call has already resolved (deny error
        # or successful dispatch).  Pinning the model's view is
        # one-way; an allow appended after the result has no dispatch
        # effect (``CONFIRMED_ROWS_SQL`` filters out sessions whose
        # confirmed-and-not-resolved set is empty), yet returning the
        # lifecycle event would lie to the operator ("allow
        # accepted").  Surface as ``ConflictError`` (→ 409) so the
        # operator learns "too late" — same shape as #535's
        # deny-after-success.
        prior_result = await queries.find_tool_result_event(
            conn, session_id, tool_call_id, account_id=account_id
        )
        if prior_result is not None:
            prior_is_error = bool(prior_result.data.get("is_error", False))
            raise ConflictError(
                f"tool_call_id {tool_call_id!r} already has a "
                f"{'error' if prior_is_error else 'success'} "
                f"result; cannot allow a tool call that has already resolved",
                detail={
                    "session_id": session_id,
                    "tool_call_id": tool_call_id,
                    "existing_is_error": prior_is_error,
                },
            )
        return await queries.append_event(
            conn,
            session_id=session_id,
            kind="lifecycle",
            data={"event": "tool_confirmed", "tool_call_id": tool_call_id, "result": "allow"},
            account_id=account_id,
        )


async def confirm_tool_deny(
    pool: asyncpg.Pool[Any],
    session_id: str,
    tool_call_id: str,
    deny_message: str,
    *,
    account_id: str,
) -> Event:
    """Deny an ``always_ask`` tool call.

    Appends a tool-role error event that the model will see in its next
    context window. The deny message is formatted to match Anthropic's
    ``"Permission to use <tool> has been rejected."`` pattern.

    Idempotent on ``(session_id, tool_call_id)``: a retried POST
    returns the original event instead of appending a duplicate that
    would violate the monotonic-context invariant (CLAUDE.md #2).
    Delegates to :func:`append_tool_result`, which carries the dedup
    machinery (#445) — same event shape (``role:"tool"`` with
    ``is_error=True``), so the dedup applies uniformly to the deny path.
    """
    # Find the tool name from the event log for the error message.
    events = await read_message_events(pool, session_id, account_id=account_id)
    tc = _find_tool_call(events, tool_call_id)
    tool_name = ((tc.get("function") or {}).get("name", "unknown")) if tc else "unknown"

    content = json.dumps(
        {
            "error": (
                f"Permission to use {tool_name} has been rejected. "
                f"Rejection message: {deny_message}"
            )
        },
        ensure_ascii=False,
    )
    async with pool.acquire() as conn:
        return await append_tool_result(
            conn,
            account_id=account_id,
            session_id=session_id,
            tool_call_id=tool_call_id,
            content=content,
            is_error=True,
        )
