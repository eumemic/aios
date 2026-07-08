"""Single-step session harness.

Phase 5 replaces the synchronous multi-turn loop with an event-driven
step function. Each procrastinate ``wake_session`` job calls
:func:`run_session_step`, which:

1. Checks whether the model needs to be called
   (:func:`~aios.harness.sweep.find_sessions_needing_inference`).
2. Builds the chat-completions message list with pending-result synthesis.
3. Calls LiteLLM exactly once.
4. Appends the assistant message to the session log.
5. Kicks off tool calls as fire-and-forget asyncio tasks (if any).
6. Returns — the procrastinate lock is released immediately.

Tool completion triggers a new ``wake_session`` job, which runs another
step. The "loop" is the job queue re-entering this function.

Mid-turn user injection is free: a new user message is just another
event in the log. The next step's gate sees it via the ``reacting_to``
watermark and proceeds.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import litellm.exceptions as litellm_exceptions
from structlog.contextvars import bind_contextvars, clear_contextvars

from aios.config import HARNESS_STEP_TIMEOUT_S as HARNESS_STEP_TIMEOUT_S
from aios.config import get_settings
from aios.db.sse_lock import has_subscriber
from aios.harness import runtime
from aios.harness.completion import (
    LlmRequest,
    ModelCallDeadlineError,
    call_litellm,
    estimate_cost_usd,
    stream_litellm,
)
from aios.harness.model_binding import (
    BindingBoundaryError,
    effective_capability_model,
    map_run_output_to_response,
    parse_workflow_model,
)
from aios.harness.model_workflow import (
    HarvestedInference,
    ParkState,
    launch_model_workflow_park,
    take_pending_harvest,
)
from aios.harness.step_context import (
    compose_step_context,
    compute_step_prelude,
    prelude_overhead_local,
)
from aios.harness.sweep import find_sessions_needing_inference, session_has_pending_work
from aios.harness.tokens import approx_tokens, approx_tokens_by_class
from aios.harness.tool_dispatch import launch_mcp_tool_calls, launch_tool_calls
from aios.harness.tool_disposition import classify_tool_call
from aios.jobs.app import defer_run_wake, defer_wake
from aios.logging import get_logger
from aios.models.agents import (
    McpServerSpec,
    StepSurface,
    is_mcp_tool_name,
)
from aios.models.events import (
    ERRORED_LIFECYCLE_STATUS,
    ERRORED_LIFECYCLE_STOP_REASON,
    Event,
)
from aios.services import accounts as accounts_service
from aios.services import agents as agents_service
from aios.services import sessions as sessions_service
from aios.tools.workflow_completion import fail_all_open_requests

if TYPE_CHECKING:
    import asyncpg

    from aios.harness.inflight_tool_registry import InflightToolRegistry
    from aios.models.memory_stores import MemoryStoreResourceEcho

log = get_logger("aios.harness.loop")


_RETRY_BACKOFF_SECONDS: list[float] = [2, 8, 30, 120]

# ``HARNESS_STEP_TIMEOUT_S`` (imported from ``aios.config`` above) is the
# wall-clock cap on a single ``run_session_step`` call. The harness's
# zero-hang guarantee: per-call timeouts (LiteLLM, MCP, tool dispatch, etc.)
# are the precise instruments, but if any future code path bypasses them
# this cap fires and forces a clean rescheduling. Sized as the default 900s
# model-call deadline plus 60s of headroom for prologue, context-build, and
# epilogue work that no longer compete with the model budget.

# litellm's standardized ``finish_reason`` for a safety refusal. Anthropic's
# ``stop_reason: "refusal"`` maps here; OpenAI/Azure ``content_filter`` lands
# here too. A refusal is a *bricked* turn: the response is often truncated
# mid-generation (a tool call with a half-written argument the API closes into
# valid-but-wrong JSON) or empty (refused at token 1). Persisting it poisons
# subsequent turns and dispatching its tool calls is dangerous, so the step
# surfaces it as an errored turn instead of treating it as a normal completion.
# Keyed on the standardized value — NOT on any provider/model name — so it
# stays provider-agnostic.
REFUSAL_FINISH_REASON = "content_filter"

# Operator-facing message on the errored stop_reason. Renders behind the
# console's "Errored" pill (status idle + stop_reason.type == "error").
_REFUSAL_STOP_REASON_MESSAGE = (
    "Model returned a content_filter refusal; the turn was not dispatched. "
    "The model likely refused due to conversation content. To recover, post a "
    "message to the session (optionally after switching the agent's model or "
    "trimming the conversation that triggered the refusal)."
)
_SPEND_CAP_STOP_REASON_MESSAGE = (
    "This account has reached its spend limit, so no model calls can be made. "
    "The account operator can raise the limit (account config spend_limit_usd) to resume."
)

# litellm's already-typed terminal model-call errors: requests that will NEVER
# succeed on retry of the SAME prompt. Routing these straight to the errored
# latch skips the [2,8,30,120]s backoff ladder, which on a terminal error only
# burns ~160s + up to 4 doomed prompt-token round-trips. Kept as a small FAMILY
# rule over litellm's exception hierarchy (generalize-over-enumerate) — NOT a
# per-provider-string enum. ContextWindowExceededError and
# ContentPolicyViolationError both subclass BadRequestError, so they are already
# covered by isinstance against BadRequestError alone; they are listed only to
# DOCUMENT the covered 400-subclasses (redundant for matching, not required).
_TERMINAL_MODEL_ERRORS: tuple[type[Exception], ...] = (
    litellm_exceptions.BadRequestError,  # 400: malformed/invalid prompt
    litellm_exceptions.ContextWindowExceededError,  # 400 subclass: prompt too long
    litellm_exceptions.ContentPolicyViolationError,  # 400 subclass: policy block
    litellm_exceptions.AuthenticationError,  # 401: bad/expired key
    litellm_exceptions.PermissionDeniedError,  # 403: key lacks model access
    litellm_exceptions.NotFoundError,  # 404: unknown model id
    litellm_exceptions.UnprocessableEntityError,  # 422: schema-invalid request
)


# Cap the persisted provider-error message so a pathological multi-KB provider
# body can't bloat the span JSONB. The class + status are the high-signal fields;
# the message is a truncated human hint, not a full transcript.
_PROVIDER_ERROR_MESSAGE_MAX_CHARS = 2000


def _provider_error_detail(exc: BaseException) -> dict[str, Any]:
    """Distil a model-call exception into the durable diagnostic triple.

    Persists ``exception_class`` + ``http_status`` + ``message`` on the
    ``child_errored``/errored-turn span so a latched failure is diagnosable
    straight from Postgres/console — no more reconstructing 429/529/overload/
    timeout causes from span *timings* (#1442).

    ``http_status`` is litellm's ``status_code`` when present (RateLimitError →
    429, InternalServerError → 500/529, etc.); ``None`` for non-HTTP failures
    (connection drops, our own ``ModelCallDeadlineError``). The message is
    truncated to keep the span JSONB bounded.
    """
    message = str(exc)
    if len(message) > _PROVIDER_ERROR_MESSAGE_MAX_CHARS:
        message = message[:_PROVIDER_ERROR_MESSAGE_MAX_CHARS] + "…"
    status = getattr(exc, "status_code", None)
    http_status: int | None
    if isinstance(status, bool):  # bool is an int subclass — never a status code
        http_status = None
    elif isinstance(status, int):
        http_status = status
    else:
        try:
            http_status = int(status) if status is not None else None
        except (TypeError, ValueError):
            http_status = None
    return {
        "exception_class": type(exc).__name__,
        "http_status": http_status,
        "message": message,
    }


def _is_terminal_model_error(exc: BaseException) -> bool:
    """True iff ``exc`` is a litellm error that retrying the SAME prompt cannot fix.

    Fail-SAFE: anything NOT matched here (RateLimitError / APIConnectionError /
    InternalServerError / Timeout / ServiceUnavailableError, or any non-litellm
    Exception) falls through to the existing backoff ladder — current behavior.
    A misclassified *transient* therefore degrades to burning the ladder, never
    the reverse.
    """
    return isinstance(exc, _TERMINAL_MODEL_ERRORS)


_MODEL_TERMINAL_ERROR_STOP_REASON_MESSAGE = (
    "The model call failed with a terminal error that retrying cannot fix "
    "(e.g. invalid request, context-window exceeded, content policy, or auth). "
    "To recover, post a message to the session, optionally after switching the "
    "agent's model or trimming the conversation."
)


def _retry_delay_for_attempt(attempt: int) -> float | None:
    """Return the backoff delay for ``attempt``, or ``None`` if the budget is spent."""
    if attempt >= len(_RETRY_BACKOFF_SECONDS):
        return None
    return _RETRY_BACKOFF_SECONDS[attempt]


def _limit_to_microusd(limit_usd: float | None) -> int | None:
    if limit_usd is None:
        return None
    return round(limit_usd * 1_000_000)


def _resolve_cost_microusd(
    model: str, usage: dict[str, int], cost_usd: float | None, *, session_id: str
) -> int:
    """Cost of a model call in micro-USD; warn and return 0 if the model is cost-unmapped."""
    effective_cost_usd = cost_usd if cost_usd is not None else estimate_cost_usd(model, usage)
    if effective_cost_usd is None:
        log.warning("usage.model_cost_unmapped", model=model, session_id=session_id)
        return 0
    return round(effective_cost_usd * 1_000_000)


async def _resolve_capability_model(pool: asyncpg.Pool[Any], model: str, *, account_id: str) -> str:
    """Resolve the model the capability gates should key on for ``model`` (#1637).

    For a raw provider model this is ``model`` unchanged (the common case — no DB
    read). For a ``workflow:<id>[@version]`` binding it is the bound workflow's
    declared effective model (``output_model``), looked up from the workflow
    definition, so the vision / thinking / token-window gates resolve to the
    inner model instead of silently degrading on the opaque ``workflow:`` string.

    A binding whose workflow declares no ``output_model`` — or whose lookup fails
    (deleted workflow, malformed binding) — falls back to the raw ``workflow:``
    string: the pre-#1637 degraded posture, never an error (the gate resolution
    must not wedge the step; the binding's own dispatch path validates the ref).
    """
    ref = parse_workflow_model(model)
    if ref is None:
        return model
    from aios.services import workflows as wf_service

    output_model: str | None = None
    try:
        if ref.version is not None:
            version = await wf_service.get_workflow_version(
                pool, ref.workflow_id, ref.version, account_id=account_id
            )
            output_model = version.output_model
        else:
            workflow = await wf_service.get_workflow(pool, ref.workflow_id, account_id=account_id)
            output_model = workflow.output_model
    except Exception as err:
        # A missing/archived workflow or a transient read error must not wedge the
        # step here — the binding's dispatch path (the park) surfaces a real ref
        # error. Degrade the gates to the raw string (no worse than pre-#1637).
        log.warning(
            "step.capability_model_lookup_failed",
            model=model,
            workflow_id=ref.workflow_id,
            error=str(err),
        )
        return model
    return effective_capability_model(model, output_model=output_model)


def _crossed_spend_warning_threshold(
    new_spent_microusd: int, cost_microusd: int, limit_microusd: int | None
) -> bool:
    if limit_microusd is None or cost_microusd <= 0:
        return False
    threshold = 0.8 * limit_microusd
    return new_spent_microusd >= threshold > new_spent_microusd - cost_microusd


class _StepResult(NamedTuple):
    """What a step asks ``run_session_step`` to defer **after** ``step_end``.

    Every wake a step provokes is deferred here, never inside the body, so its
    ``wake_deferred`` span lands in step N+1's window — the "all wake_deferred since
    the previous step_end" pairing rule the profiler relies on (#132).

    * ``retry_delay`` — a model-error backoff reschedule (``defer_wake``).
    * ``nudge_session`` — the quiescence guard nudged an owed request; re-wake the
      session so the model gets another turn (``defer_wake``).
    * ``autoerror_caller_run_id`` — the guard auto-errored a request past its budget;
      wake that caller run to harvest the ``no_return`` (``defer_run_wake``).
    * ``autoerror_caller_session_ids`` — same, for session callers (#1127): wake each
      caller session so its parked ``invoke`` tool task harvests the ``no_return``
      (``defer_wake``).
    * ``archive_when_idle`` — the session was launched self-reclaiming; archive it
      (iff still idle) as the LAST write of the step, after ``step_end`` and the wakes
      (once archived, any further ``append_event`` hits the ``archived_at IS NULL`` fence).
    """

    retry_delay: float | None = None
    nudge_session: bool = False
    autoerror_caller_run_id: str | None = None
    autoerror_caller_session_ids: tuple[str, ...] = ()
    archive_when_idle: bool = False


async def refresh_session_mount_state(
    pool: asyncpg.Pool[Any], session_id: str, *, account_id: str
) -> list[MemoryStoreResourceEcho]:
    """Refresh the cached resource echoes and the sandbox drift check.

    Returns the memory echoes (used downstream for prompt augmentation).
    Github echoes and env-var credential echoes are fed into the registry's
    drift check but not returned — no current caller in the step body needs
    them. The env-var echoes carry ``updated_at`` so a credential rotation
    recycles the sandbox (#877), exactly like github token rotation.
    """
    from aios.db import queries

    async with pool.acquire() as conn:
        memory_echoes = await queries.list_session_memory_store_echoes(
            conn, session_id, account_id=account_id
        )
        github_echoes = await queries.list_session_github_repo_echoes(
            conn, session_id, account_id=account_id
        )
        env_var_echoes = await queries.list_session_env_var_credential_echoes(
            conn, session_id, account_id=account_id
        )
    runtime.set_session_memory_mounts(session_id, memory_echoes)
    if runtime.sandbox_registry is not None:
        await runtime.sandbox_registry.release_if_mounts_changed(
            session_id, memory_echoes, github_echoes, env_var_echoes
        )
    return memory_echoes


async def run_session_step(
    session_id: str,
    *,
    cause: str = "message",
) -> None:
    """Run one inference step for the session.

    Called by the procrastinate ``wake_session`` task. The procrastinate
    ``lock`` parameter guarantees only one step runs per session at a
    time.
    """
    pool = runtime.require_pool()
    # Entry guard (mirrors run_workflow_step's terminal early-return): a wake for a
    # session that has been archived or deleted is an idempotent no-op. Without it
    # the unconditional step_start append below would hit append_event's archived
    # guard and fail the job — e.g. a sweep wake racing an operator archive, or a
    # workflow child reclaimed after answering.
    account_id = await sessions_service.load_live_session_account_id(pool, session_id)
    if account_id is None:
        return
    # Bind tenant/session/cause onto structlog contextvars so every line this step
    # emits is attributable; cleared in the outer finally below (defensive hygiene —
    # see that block). Bound AFTER the gone-session guard — that path is an
    # idempotent no-op with no account to attribute.
    bind_contextvars(session_id=session_id, account_id=account_id, cause=cause)
    # Outer try/finally guarantees ``clear_contextvars`` is the VERY LAST thing the
    # step does: the post-step wakes and archive-reclaim below (and their nested
    # ``defer_wake`` logging) must still carry account_id/session_id/cause, so the
    # clear can only run once every per-step log line has been emitted. Clearing is
    # defensive hygiene — contextvars are task-scoped and procrastinate runs each job
    # in a fresh asyncio task, but dropping the bindings here keeps them from
    # outliving the step regardless of how the worker schedules tasks. The try opens
    # IMMEDIATELY after the bind so the still-fallible setup below (``step_start``
    # append, ``register_step``) can't escape with the contextvars still bound — e.g.
    # a DB drop or a session archived in the race window past the account_id guard.
    try:
        inflight_tool_registry = runtime.require_inflight_tool_registry()

        # Outermost span pair: brackets the entire step (issue #131).  Emitted
        # before the sweep guard so early-outs are also measured — a "wasted
        # wake" cost shows up as a ``step_start``/``step_end`` pair with no
        # ``context_build_*`` inside.  ``step_start_id`` backpointer on the
        # end event matches the ``context_build_start_id`` convention.
        step_start = await sessions_service.append_event(
            pool,
            session_id,
            "span",
            {"event": "step_start", "cause": cause},
            account_id=account_id,
        )
        current_task = asyncio.current_task()
        assert current_task is not None
        inflight_tool_registry.register_step(session_id, current_task)
        result = _StepResult()
        try:
            try:
                result = await asyncio.wait_for(
                    _run_session_step_body(
                        pool,
                        inflight_tool_registry,
                        session_id,
                        cause=cause,
                        account_id=account_id,
                    ),
                    timeout=HARNESS_STEP_TIMEOUT_S,
                )
            except TimeoutError:
                # Job-level safety net: a per-call timeout was missing or didn't
                # fire. Force a reschedulable error state so the next wake can
                # proceed (matches what the body's litellm-error handler does).
                log.exception(
                    "step.job_timeout", session_id=session_id, timeout=HARNESS_STEP_TIMEOUT_S
                )
                result = _StepResult(
                    retry_delay=await _handle_step_timeout(pool, session_id, account_id=account_id)
                )
            except Exception:
                # Unexpected harness error (not a model/tool error — those are caught
                # inside _run_session_step_body). Emit a span so the event log has a
                # record, then apply the retry-or-failure state machine identically to
                # the timeout path. Re-raise when the budget is exhausted.
                log.exception("step.harness_error", session_id=session_id)
                await sessions_service.append_event(
                    pool,
                    session_id,
                    "span",
                    {"event": "harness_error", "is_error": True},
                    account_id=account_id,
                )
                delay = await _apply_retry_or_failure(pool, session_id, account_id=account_id)
                result = _StepResult(retry_delay=delay)
                if delay is None:
                    raise
        finally:
            inflight_tool_registry.unregister_step(session_id)
            await sessions_service.append_event(
                pool,
                session_id,
                "span",
                {"event": "step_end", "step_start_id": step_start.id},
                account_id=account_id,
            )

        # Every wake fires AFTER ``step_end`` so its ``wake_deferred`` span lands in
        # step N+1's temporal window, not step N's — the "all wake_deferred since the
        # previous step_end" pairing rule the profiler keys off (#132). Emitting inside
        # the body would mis-attribute the gap. ``reschedule`` carries a known backoff
        # delay; the ``request_nudge`` re-wakes the session for another answer turn; the
        # auto-error wakes the caller run to harvest the ``no_return``.
        if result.retry_delay is not None:
            await defer_wake(
                pool,
                session_id,
                cause="reschedule",
                delay_seconds=result.retry_delay,
                account_id=account_id,
            )
        if result.nudge_session:
            await defer_wake(pool, session_id, cause="request_nudge", account_id=account_id)
        if result.autoerror_caller_run_id is not None:
            # batch: a no_return auto-error is a child completion like any other.
            await defer_run_wake(result.autoerror_caller_run_id, batch=True)
        for caller_session_id in result.autoerror_caller_session_ids:
            # Session caller (#1127): wake it so its parked invoke() tool task harvests
            # the no_return response (its await_session is also self-subscribed here).
            await defer_wake(
                pool, caller_session_id, cause="invoke_response", account_id=account_id
            )

        # Archive-on-quiescence, performed LAST: a session launched ``archive_when_idle``
        # self-archives the first time it goes idle. It runs after ``step_end`` and every
        # wake span so it is the step's final session write — once archived, ``append_event``
        # rejects writes (``archived_at IS NULL`` fence). ``reclaim_session_if_idle`` is
        # conditional on still-idle, so a nudge or a user message that re-activated the
        # session (its ``defer_wake`` span already appended just above) wins and the archive
        # no-ops. Only the clean end-of-turn return sets the flag (error/reschedule paths
        # leave it False), so a rescheduling session is never reclaimed out from under a retry.
        if result.archive_when_idle and await sessions_service.reclaim_session_if_idle(
            pool, session_id, account_id=account_id
        ):
            log.info("step.session_reclaimed", session_id=session_id, cause=cause)
    finally:
        clear_contextvars()


async def _run_session_step_body(
    pool: asyncpg.Pool[Any],
    inflight_tool_registry: InflightToolRegistry,
    session_id: str,
    *,
    cause: str,
    account_id: str,
) -> _StepResult:
    """Returns the wakes ``run_session_step`` should defer **after** ``step_end``
    (see :class:`_StepResult`): a model-error backoff reschedule, and/or the
    quiescence guard's nudge / auto-error wakes. Keeping every ``defer_wake`` out of
    the body is what makes each ``wake_deferred`` land in the next step's window."""
    # Sweep-based guard: does this session actually need work?
    # Prevents wasted DB/model calls from stale or duplicate wakes.
    #
    # Bracket with a ``sweep_start``/``sweep_end`` span pair (site="entry").
    # Only the wake-detection query runs here — no ghost repair, no
    # ``defer_wake`` — so ``repaired_ghosts`` is always 0. ``woken_sessions``
    # at ``site="entry"`` is 0 or 1: it records whether the guard determined
    # this specific session had work. 0 indicates a wasted wake.
    #
    # (A) Fast-path early-out (#1659). The full multi-CTE
    # ``find_sessions_needing_inference`` measured a worker-wide 3.8-5.2s median
    # on the per-turn entry path (worker-pool / event-loop contention, not query
    # cost). Replace it here with ``session_has_pending_work`` — a single
    # PK-scoped boolean that is a PROVEN OVER-APPROXIMATION of the full sweep
    # (TRUE whenever the full sweep could return this session). Safety by
    # construction (the wedge-class constraint): early-out ONLY on a provable
    # ``False`` ("no work"); on ``True`` ("maybe work") FALL THROUGH to the full
    # ``find_sessions_needing_inference``. A wrong predicate then costs at most an
    # extra full sweep, never a missed wake → never a wedged session.
    #
    # (C) Observability spans (#1659). Bracket the two guard sub-costs so the
    # currently-inferred split is measured: ``sweep.pool_acquire`` around the
    # connection acquire and ``sweep.query_exec`` around the SQL. Then
    # ``(sweep_end - sweep_start) - query_exec - pool_acquire ~= event-loop
    # time-share`` -- which gates whether (B) pool/concurrency tuning is the
    # load-bearing follow-up.
    sweep_start = await sessions_service.append_event(
        pool,
        session_id,
        "span",
        {"event": "sweep_start", "site": "entry"},
        account_id=account_id,
    )
    has_work = False
    try:
        pool_acquire_start = await sessions_service.append_event(
            pool,
            session_id,
            "span",
            {"event": "sweep.pool_acquire_start"},
            account_id=account_id,
        )
        async with pool.acquire() as fast_conn:
            await sessions_service.append_event(
                pool,
                session_id,
                "span",
                {
                    "event": "sweep.pool_acquire_end",
                    "start_id": pool_acquire_start.id,
                },
                account_id=account_id,
            )
            query_exec_start = await sessions_service.append_event(
                pool,
                session_id,
                "span",
                {"event": "sweep.query_exec_start"},
                account_id=account_id,
            )
            try:
                has_work = await session_has_pending_work(fast_conn, session_id)
            finally:
                await sessions_service.append_event(
                    pool,
                    session_id,
                    "span",
                    {
                        "event": "sweep.query_exec_end",
                        "start_id": query_exec_start.id,
                    },
                    account_id=account_id,
                )
    finally:
        await sessions_service.append_event(
            pool,
            session_id,
            "span",
            {
                "event": "sweep_end",
                "sweep_start_id": sweep_start.id,
                "repaired_ghosts": 0,
                "woken_sessions": 1 if has_work else 0,
                "fast_path": True,
            },
            account_id=account_id,
        )
    if not has_work:
        # Provably no work — the fast path is a proven over-approximation, so a
        # ``False`` here is authoritative. Skip the full sweep entirely.
        log.debug("step.early_out", session_id=session_id, cause=cause)
        return _StepResult()

    # Fast path said "maybe work" — fall through to the authoritative full sweep.
    # It can still find no work (e.g. an incomplete tool batch the scalar gate
    # can't see), in which case we early-out here; the cost of a wrong fast-path
    # predicate is bounded to this occasional extra full sweep, never a wedge.
    needs = await find_sessions_needing_inference(
        pool, inflight_tool_registry, session_id=session_id
    )
    if session_id not in needs:
        log.debug("step.early_out", session_id=session_id, cause=cause, via="full_sweep")
        return _StepResult()

    # Cancel leaf (cancel-design §4/§6): a session carrying an unharvested cancel-marker
    # answers each cancelled request + harvests the marker under its own lock, then skips
    # inference this turn — the cancelled request needs no model work. The C2 sweep clause
    # is what put a marked session into ``needs`` above, so this only runs when there's an
    # exit to apply; once harvested it no longer re-wakes (no hot-loop).
    if await sessions_service.harvest_session_cancel_markers(
        pool, session_id, account_id=account_id
    ):
        log.info("step.cancel_harvested", session_id=session_id)
        return _StepResult()

    session = await sessions_service.get_session_basic(pool, session_id, account_id=account_id)

    from aios.services.channels import list_session_channels

    agent, channels, memory_echoes = await asyncio.gather(
        agents_service.load_for_session(pool, session, account_id=account_id),
        list_session_channels(pool, session_id, account_id=account_id),
        refresh_session_mount_state(pool, session_id, account_id=account_id),
    )

    mcp_server_map: dict[str, McpServerSpec] = {s.name: s for s in agent.mcp_servers}

    # ── workflow: binding capability descriptor (#1637) ──
    #
    # The capability gates — vision (``supports_vision``), extended-thinking
    # continuity (``model_descriptor(...).supports_thinking``), and token-window
    # calibration (``read_windowed_events(model=...)``) — all key on the literal
    # model string. A ``workflow:<id>`` binding matches none, so a bound model
    # would silently degrade: images dropped, thinking-blocks stripped, token
    # counting under-counts. Resolve the binding to its declared EFFECTIVE model
    # (the bound workflow's ``output_model``) ONCE here and key every downstream
    # gate on that effective string instead of the opaque ``workflow:`` one. A
    # raw provider model resolves to itself, so the common case is unchanged.
    capability_model = await _resolve_capability_model(pool, agent.model, account_id=account_id)

    # Span the prelude + windowed-read work individually (issue #1658).
    #
    # This pair of reads runs in the window between ``step_start`` and
    # ``context_build_start`` and used to be UNSPANNED — the ``context_build_*``
    # pair only brackets ``compose_step_context`` (which measures ~0.00s), so a
    # multi-second pre-inference read cost was blind-spotted by every profiling
    # angle that keyed on ``context_build_*``. Bracketing each read individually
    # turns "where did the pre-inference seconds go?" into a query instead of a
    # manual full-window (``step_start`` → ``context_build_start``)
    # decomposition. On failure we still emit the end with ``is_error: True``
    # and re-raise, matching the ``context_build_*`` / ``model_request_*``
    # symmetry (no orphan starts).

    # Build the events-independent prelude (system prompt + tools)
    # before windowing so its overhead can be subtracted from the
    # window budget — otherwise the sent prompt can exceed window_max
    # by exactly that overhead.
    compute_prelude_start = await sessions_service.append_event(
        pool,
        session_id,
        "span",
        {"event": "compute_prelude_start"},
        account_id=account_id,
    )
    try:
        prelude = await compute_step_prelude(
            pool,
            session_id,
            account_id=account_id,
            session=session,
            agent=agent,
            channels=channels,
            memory_store_echoes=memory_echoes,
        )
    except Exception:
        await sessions_service.append_event(
            pool,
            session_id,
            "span",
            {
                "event": "compute_prelude_end",
                "compute_prelude_start_id": compute_prelude_start.id,
                "is_error": True,
            },
            account_id=account_id,
        )
        raise
    await sessions_service.append_event(
        pool,
        session_id,
        "span",
        {
            "event": "compute_prelude_end",
            "compute_prelude_start_id": compute_prelude_start.id,
            "is_error": False,
        },
        account_id=account_id,
    )

    # Read windowed message events for this session.
    read_window_start = await sessions_service.append_event(
        pool,
        session_id,
        "span",
        {"event": "read_window_start"},
        account_id=account_id,
    )
    try:
        windowed = await sessions_service.read_windowed_events(
            pool,
            session_id,
            window_min=agent.window_min,
            window_max=agent.window_max,
            model=capability_model,
            overhead_local=prelude_overhead_local(prelude),
            account_id=account_id,
        )
    except Exception:
        await sessions_service.append_event(
            pool,
            session_id,
            "span",
            {
                "event": "read_window_end",
                "read_window_start_id": read_window_start.id,
                "is_error": True,
            },
            account_id=account_id,
        )
        raise
    events = windowed.events
    await sessions_service.append_event(
        pool,
        session_id,
        "span",
        {
            "event": "read_window_end",
            "read_window_start_id": read_window_start.id,
            "is_error": False,
            "event_count_read": len(events),
        },
        account_id=account_id,
    )

    # Check for confirmed-but-undispatched tool calls (always_ask → allow).
    # The sweep's case (c) ensures we passed the guard above.
    pending = await _dispatch_confirmed_tools(
        pool,
        session_id,
        account_id=account_id,
        inflight_tool_registry=inflight_tool_registry,
    )
    if pending:
        pending_builtin = [tc for tc in pending if not is_mcp_tool_name(_tc_name(tc))]
        pending_mcp = [tc for tc in pending if is_mcp_tool_name(_tc_name(tc))]
        if pending_builtin:
            launch_tool_calls(pool, session_id, pending_builtin, account_id=account_id)
        if pending_mcp:
            launch_mcp_tool_calls(
                pool,
                session_id,
                pending_mcp,
                mcp_server_map,
                focal_channel=session.focal_channel,
                account_id=account_id,
            )
        log.info(
            "step.confirmed_tools_dispatched",
            session_id=session_id,
            count=len(pending),
        )
        return _StepResult()

    # Pre-flight admission against the rolled-up subtree envelope (#1279).
    #
    # The admission decision — refusing to *start* this step's model call
    # before dispatch — is made against the rolled-up subtree envelope (P1's
    # `get_account_subtree_spent_microusd`, #1296): the SUM of every meter at
    # or below this account, archived edges severed. The rollup includes the
    # account's own meter, so it subsumes the flat ceiling — a flat breach is
    # always a subtree breach — but it ALSO latches once descendants'
    # *cumulative* spend breaches an ancestor's effective limit, even when no
    # single account's flat meter has crossed on its own. A hard ceiling, not
    # an allocation market: dollars are an externally-checkable derived scalar.
    # (The post-call warning threshold below measures the freshly-charged flat
    # meter against this same effective limit — see `_charge_usage`.)
    (
        subtree_spent_microusd,
        spend_limit_usd,
    ) = await accounts_service.get_account_subtree_spend_state(pool, account_id)
    spend_limit_microusd = _limit_to_microusd(spend_limit_usd)
    if spend_limit_microusd is not None and subtree_spent_microusd >= spend_limit_microusd:
        assert spend_limit_usd is not None
        await _handle_spend_cap(
            pool,
            session_id,
            spent_microusd=subtree_spent_microusd,
            spend_limit_usd=spend_limit_usd,
            account_id=account_id,
        )
        return _StepResult()

    # Span the remainder of the prologue so "why is the step slow?"
    # can separate context-build cost from model-call cost (issue #78).
    # Bracketing starts AFTER the dispatch early-return so every start
    # has a matching end; on failure we still emit the end with
    # ``is_error: True`` and re-raise, matching the ``model_request_*``
    # symmetry.
    context_build_start = await sessions_service.append_event(
        pool,
        session_id,
        "span",
        {"event": "context_build_start"},
        account_id=account_id,
    )

    try:
        step_ctx = await compose_step_context(
            pool=pool,
            session=session,
            account_id=account_id,
            agent=agent,
            channels=channels,
            prelude=prelude,
            events=events,
            in_flight_tool_call_ids=frozenset(
                inflight_tool_registry.in_flight_tool_call_ids(session_id)
            ),
            omission=windowed.omission,
            capability_model=capability_model,
        )
    except Exception:
        await sessions_service.append_event(
            pool,
            session_id,
            "span",
            {
                "event": "context_build_end",
                "context_build_start_id": context_build_start.id,
                "is_error": True,
            },
            account_id=account_id,
        )
        raise

    messages = step_ctx.messages
    tools = step_ctx.tools

    # Provision skill files to workspace (idempotent, host-side writes).
    if step_ctx.skill_versions:
        from aios.harness.skills import provision_skill_files

        await provision_skill_files(session_id, step_ctx.skill_versions)

    await sessions_service.append_event(
        pool,
        session_id,
        "span",
        {
            "event": "context_build_end",
            "context_build_start_id": context_build_start.id,
            "is_error": False,
            "event_count_read": len(events),
            "message_count": len(messages),
            "tools_count": len(tools),
        },
        account_id=account_id,
    )

    # Dump the exact chat-completions payload we're about to send to LiteLLM
    # when AIOS_DUMP_CONTEXT is set — useful for debugging prompt construction
    # (header inlining, system-prompt augmentation, tool list shape).
    await _dump_context_if_enabled(session_id, agent.model, messages, tools)

    llm_request = LlmRequest(
        messages=messages,
        tools=tools if tools else None,
        params=agent.litellm_extra or None,
        session_id=session_id,
    )

    # ── workflow: model binding — async two-step model-dispatch + harvest (#1634) ──
    #
    # When ``agent.model`` is ``workflow:<wf_id>[@version]`` the inference is produced
    # by a workflow run, not a raw provider call. The whole step is under the 960s cap
    # (config.py) and "stay responsive while parked" is a tool-task property — so we do
    # NOT await the inner deliberation inline. Instead:
    #
    #   * HARVEST (step N+1): a prior step already parked owing an assistant message; the
    #     bound run has resolved, so fold its structured return into ``assistant_msg`` and
    #     run the existing append/charge/dispatch tail. NO re-charge — the inner inference
    #     charged once at its own ``call_llm`` site; we record only a span. ``reacting_to``
    #     was sealed at park (not recomputed here).
    #   * PARK (step N): no harvest pending → open an awaited run, journal the park (sealing
    #     ``reacting_to``), and end the step owing an assistant message. The run resolves
    #     async; its completion wakes the session, which lands on the harvest branch above.
    workflow_ref = parse_workflow_model(agent.model)
    # ``no_recharge`` / ``reacting_to_override`` are the two ways the harvest path
    # differs from the inline-model tail it shares: the inner inference already
    # charged at its own ``call_llm`` site (record a span, do NOT re-charge), and
    # the assistant turn carries the watermark sealed at park (NOT step_ctx's, which
    # may have advanced as stimuli arrived during the inner deliberation).
    no_recharge = False
    reacting_to_override: int | None = None
    harvested: HarvestedInference | None = None
    if workflow_ref is not None:
        disposition = await take_pending_harvest(pool, session_id, account_id=account_id)
        if disposition is ParkState.PARK_PENDING:
            # A park is OPEN and its run has not resolved yet. The park wrote a
            # ``span`` event, which does not advance ``last_stimulus_seq`` /
            # ``last_reacted_seq`` — so the unreacted-stimulus inequality that caused
            # the park still holds and the sweep keeps re-waking this session every
            # tick while the inner run deliberates. End the step WITHOUT launching a
            # second run (re-parking on nothing): exactly ONE inner awaited run runs
            # per turn regardless of how many sweep ticks elapse. The harvest task's
            # ``defer_wake`` (or a later sweep re-wake) re-enters and harvests.
            return _StepResult()
        if disposition is ParkState.NO_PARK:
            await launch_model_workflow_park(
                pool,
                session_id,
                ref=workflow_ref,
                request=llm_request,
                reacting_to=step_ctx.reacting_to,
                account_id=account_id,
            )
            # End the step owing an assistant message — a new step disposition. The run's
            # async resolution wakes the session for the harvest; no inference ran here, so
            # no model_request span, no charge, no assistant turn.
            return _StepResult()
        harvested = disposition

    if harvested is not None:
        # ── HARVEST: fold the resolved bound run into the shared dispatch tail ──
        # EVERY fold path below writes exactly one ``model_workflow_harvest_end``
        # span carrying this ``run_id``. That span is the durable *park-consumed*
        # marker: ``find_latest_model_workflow_park`` EXCLUDES any park whose run
        # already has it (see its docstring), so once a parked turn is folded the
        # park is un-returnable and the next stimulus launches a FRESH park instead
        # of re-folding this (stale) harvest forever. Writing it on the errored /
        # invalid-shape paths too — not just the ok path — is what stops an errored
        # workflow-model turn from re-erroring on the same stale park each time a
        # user message tries to recover the session.
        if harvested.outcome != "ok":
            # The bound run errored / was cancelled — no assistant turn to dispatch.
            # Latch errored so the session parks for recovery (a user message lifts it).
            log.warning(
                "step.model_workflow_errored",
                session_id=session_id,
                run_id=harvested.run_id,
                outcome=harvested.outcome,
                error=harvested.error,
            )
            await _latch_errored_turn(
                pool,
                session_id,
                error_kind="model_workflow_run_errored",
                stop_message=_MODEL_TERMINAL_ERROR_STOP_REASON_MESSAGE,
                account_id=account_id,
            )
            # Consume the park AFTER the errored latch lands: the session is now
            # ``errored`` (sweep-skipped) so nothing re-reads the park before this
            # marker is written; a later recovering user message reads NO_PARK.
            await _append_harvest_consumed_marker(
                pool,
                session_id,
                run_id=harvested.run_id,
                model=agent.model,
                is_error=True,
                account_id=account_id,
            )
            return _StepResult()
        try:
            llm_response = map_run_output_to_response(harvested.output)
        except BindingBoundaryError as exc:
            # The bound workflow returned a structurally invalid assistant shape — the
            # binding boundary fails loud. Latch the turn errored (same shape as a model
            # terminal error): the turn produced no usable inference.
            log.warning(
                "step.model_workflow_invalid_shape",
                session_id=session_id,
                run_id=harvested.run_id,
                error=str(exc),
            )
            await _latch_errored_turn(
                pool,
                session_id,
                error_kind="model_workflow_invalid_shape",
                stop_message=_MODEL_TERMINAL_ERROR_STOP_REASON_MESSAGE,
                account_id=account_id,
            )
            await _append_harvest_consumed_marker(
                pool,
                session_id,
                run_id=harvested.run_id,
                model=agent.model,
                is_error=True,
                account_id=account_id,
            )
            return _StepResult()
        # Record the harvest as the step's model span (NO re-charge); seal the
        # park-time watermark; fall through to the shared append/dispatch tail. This
        # span is ALSO the park-consumed marker for the ok path (see the block note
        # above) — it is the same event the errored paths write.
        no_recharge = True
        reacting_to_override = harvested.reacting_to
        start_event = await _append_harvest_consumed_marker(
            pool,
            session_id,
            run_id=harvested.run_id,
            model=agent.model,
            is_error=False,
            account_id=account_id,
        )
    else:
        # ── Inline model call: stream deltas via pg_notify only when an SSE
        # subscriber is attached (issue #81); otherwise run the faster
        # non-streaming path.  OpenRouter-style proxies can be 2-3x slower on
        # the streaming path when nobody is consuming the deltas.
        #
        # Emit span start so consumers can measure inference latency.
        start_event = await sessions_service.append_event(
            pool,
            session_id,
            "span",
            {"event": "model_request_start"},
            account_id=account_id,
        )
        subscribed = await has_subscriber(pool, session_id)
        try:
            if subscribed:
                llm_response = await stream_litellm(
                    llm_request,
                    model=agent.model,
                    pool=pool,
                )
            else:
                llm_response = await call_litellm(
                    llm_request,
                    model=agent.model,
                )
        except ModelCallDeadlineError as exc:
            log.exception(
                "step.model_call_deadline", session_id=session_id, chunks_seen=exc.chunks_seen
            )
            if exc.chunks_seen > 0:
                await _handle_streaming_model_deadline(
                    pool,
                    session_id,
                    start_event_id=start_event.id,
                    usage=exc.usage,
                    cost_usd=exc.cost_usd,
                    model=agent.model,
                    account_id=account_id,
                    provider_error=_provider_error_detail(exc),
                )
                return _StepResult()
            await _append_model_request_error_span(
                pool,
                session_id,
                start_event_id=start_event.id,
                account_id=account_id,
                provider_error=_provider_error_detail(exc),
            )
            return _StepResult(
                retry_delay=await _apply_retry_or_failure(pool, session_id, account_id=account_id)
            )
        except Exception as exc:
            if _is_terminal_model_error(exc):
                log.warning(
                    "step.model_terminal_error",
                    session_id=session_id,
                    error_class=type(exc).__name__,
                )
                await _append_model_request_error_span(
                    pool,
                    session_id,
                    start_event_id=start_event.id,
                    account_id=account_id,
                    provider_error=_provider_error_detail(exc),
                )
                await _latch_errored_turn(
                    pool,
                    session_id,
                    error_kind="model_terminal_error",
                    stop_message=_MODEL_TERMINAL_ERROR_STOP_REASON_MESSAGE,
                    account_id=account_id,
                )
                return _StepResult()  # no retry_delay → no defer_wake → session parks errored
            log.exception("step.litellm_failed", session_id=session_id)
            await _append_model_request_error_span(
                pool,
                session_id,
                start_event_id=start_event.id,
                account_id=account_id,
                provider_error=_provider_error_detail(exc),
            )
            return _StepResult(
                retry_delay=await _apply_retry_or_failure(pool, session_id, account_id=account_id)
            )

    # Project the named ``LlmResponse`` back to the locals the rest of the step
    # threads. ``assistant_msg`` is the opaque, normalized provider message dict
    # (retained on the response) that the harness persists intact — content,
    # tool_calls, thinking_blocks, and any provider extensions all survive.
    assistant_msg = llm_response.message
    usage = llm_response.usage
    cost_usd = llm_response.cost
    finish_reason = llm_response.finish_reason

    # ``local_tokens`` costs the full payload (messages + tools) so it
    # matches what the provider counts.  The error branch above stays
    # un-stamped; its ``is_error=True`` alone is enough to keep it out of
    # calibration reads (the partial index and the aggregate query both
    # filter on ``is_error=false``).
    # Per-class breakdown (issue #1609): the regression's training data.
    # ``local_tokens`` stays the model-neutral ``approx_tokens`` total so the
    # stored baseline is byte-identical to the pre-#1609 value — ``by_class``
    # only *attributes* that payload across content classes for the
    # per-(model, class) calibration.  The per-class split costs each class
    # slice in isolation, so its sum carries per-message framing overhead
    # more than once and must NOT be used as the baseline (it would skew the
    # stored ``local_tokens`` away from ``cumulative_tokens`` and from what
    # callers recompute via ``approx_tokens``).
    # Both counters re-tokenize the full slate every step; run the pair off
    # the event loop (issue #1744) — ``Encoding.encode`` is a stateless,
    # GIL-releasing call, and the per-message/per-payload memoization in
    # ``tokens.py`` makes the steady-state cost O(new tail) rather than
    # O(slate). Stamp order/content below is unchanged.
    def _compute_token_counts() -> tuple[int, dict[str, int]]:
        return (
            approx_tokens(messages, tools=tools),
            approx_tokens_by_class(messages, tools=tools),
        )

    local_tokens, by_class = await asyncio.to_thread(_compute_token_counts)
    cost_microusd = _resolve_cost_microusd(agent.model, usage, cost_usd, session_id=session_id)
    await sessions_service.append_event(
        pool,
        session_id,
        "span",
        {
            "event": "model_request_end",
            "model_request_start_id": start_event.id,
            "is_error": False,
            "model_usage": usage,
            "cost_usd": cost_usd,
            "local_tokens": local_tokens,
            "local_tokens_by_class": by_class,
            "model": agent.model,
        },
        account_id=account_id,
    )

    # Charge cumulative session-level usage AFTER the model response is durably
    # recorded — the assistant message persisted below, or the refusal span in
    # the branch. increment_usage commits in its own transaction, so charging
    # BEFORE the persist double-bills whenever the persist raises (a DB error
    # caught upstream → a retry that re-calls the model). Charging after fails
    # safe: a crash in the gap loses the charge rather than duplicating it. A
    # refusal still consumed tokens, so it charges too — after _handle_refusal
    # records its span and latches errored.
    async def _charge_usage() -> int:
        # Harvest path: the inner inference already charged at its own ``call_llm``
        # site inside the bound run, so re-charging here would double-bill. A
        # zero-delta increment is a pure read of the running account total — it
        # leaves spend untouched (so the warning-threshold check below sees delta 0)
        # while keeping a single return path.
        in_tok = 0 if no_recharge else usage.get("input_tokens", 0)
        out_tok = 0 if no_recharge else usage.get("output_tokens", 0)
        cache_r = 0 if no_recharge else usage.get("cache_read_input_tokens", 0)
        cache_c = 0 if no_recharge else usage.get("cache_creation_input_tokens", 0)
        charge = 0 if no_recharge else cost_microusd
        return await sessions_service.increment_usage(
            pool,
            session_id,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_read_input_tokens=cache_r,
            cache_creation_input_tokens=cache_c,
            cost_microusd=charge,
            account_id=account_id,
        )

    # A refusal bricks the turn: the assistant message is partial/empty and its
    # tool calls may be truncated. Do NOT persist it as a normal assistant turn
    # (it would poison subsequent context) and do NOT dispatch its tool calls
    # (a cut argument can hit a wrong-but-valid target). Record the refusal as a
    # span (excluded from build_messages replay) and latch the session into the
    # errored state, where it parks until a user message recovers it. End the
    # step cleanly — no tool dispatch, normal step_end/turn_ended bracketing.
    if finish_reason == REFUSAL_FINISH_REASON:
        await _handle_refusal(
            pool,
            session_id,
            assistant_msg,
            finish_reason=finish_reason,
            account_id=account_id,
        )
        await _charge_usage()
        log.warning(
            "step.model_refusal",
            session_id=session_id,
            finish_reason=finish_reason,
            had_tool_calls=bool(assistant_msg.get("tool_calls")),
        )
        # No ``archive_when_idle``: an errored session parks for recovery, exactly
        # like the litellm-error / retry-budget-exhausted paths, which never
        # self-archive. A user message lifts it back to pending.
        return _StepResult()

    if channels:
        from aios.harness.channels import apply_monologue_prefix

        assistant_msg = apply_monologue_prefix(assistant_msg)

    # Record the seq of the latest user/tool event in the context this
    # response was based on; events after this seq are "new" on the next
    # wake. ``find_sessions_needing_inference`` uses it as the watermark.
    # The harvest path uses the watermark sealed at PARK (``reacting_to_override``),
    # not step_ctx's — which may have advanced as stimuli arrived during the inner
    # deliberation, and must not retroactively widen what this turn "reacted to".
    assistant_msg["reacting_to"] = (
        reacting_to_override if reacting_to_override is not None else step_ctx.reacting_to
    )

    # Append assistant message to the session log (unfenced — procrastinate lock
    # provides mutual exclusion) and, in the SAME transaction, enforce request
    # totality: if this tool-call-free turn would leave the session idle while it
    # owes a request response, a nudge (or a no_return error once the budget is
    # spent) is written atomically with the idling event — so no reader ever
    # observes the session idle with an open request. A strict no-op for any
    # session that owes nothing (the common case).
    # The resulting wakes (nudge → re-wake the session; auto-error → wake the
    # caller run) are deferred by run_session_step AFTER step_end via _StepResult,
    # so their wake_deferred spans land in the next step's window (see the §wakes
    # block in run_session_step).
    guard_result = await sessions_service.append_assistant_and_guard_quiescence(
        pool,
        session_id,
        assistant_msg,
        account_id=account_id,
    )
    # Charge now that the assistant message is durably persisted (see _charge_usage).
    new_spent_microusd = await _charge_usage()
    if _crossed_spend_warning_threshold(new_spent_microusd, cost_microusd, spend_limit_microusd):
        log.warning(
            "account.spend_limit_warning",
            account_id=account_id,
            spent_usd=new_spent_microusd / 1_000_000,
            spend_limit_usd=spend_limit_usd,
        )
    nudged = guard_result.nudged
    autoerror_caller_run_id = guard_result.autoerror_caller_run_id
    autoerror_caller_session_ids = guard_result.autoerror_caller_session_ids
    # The appended assistant event's locked focal stamp — threaded into the
    # live tool dispatch so ``append_event`` skips the in-lock tool-parent
    # lookup for the results these calls produce (issue #862).
    parent_focal = guard_result.assistant_focal_at_arrival

    # Partition tool calls into dispatch buckets. Immediate builtin/MCP
    # launch now; ``needs_confirm`` and ``custom`` sit unresolved in the
    # log until an external POST lands the result — the session ends its
    # turn anyway and any stimulus can wake it (``Session.awaiting``
    # surfaces what's still pending).
    tool_calls: list[dict[str, Any]] = assistant_msg.get("tool_calls") or []

    if tool_calls:
        immediate: list[dict[str, Any]] = []
        mcp_immediate: list[dict[str, Any]] = []
        needs_confirm: list[dict[str, Any]] = []
        custom: list[dict[str, Any]] = []
        unknown_mcp: list[dict[str, Any]] = []

        for tc in tool_calls:
            kind = _classify_tool_call(tc, agent, mcp_server_map)
            if kind == "immediate":
                immediate.append(tc)
            elif kind == "mcp_immediate":
                mcp_immediate.append(tc)
            elif kind == "needs_confirm":
                needs_confirm.append(tc)
            elif kind == "custom":
                custom.append(tc)
            else:  # "unknown_mcp"
                unknown_mcp.append(tc)

        if immediate:
            launch_tool_calls(
                pool,
                session_id,
                immediate,
                account_id=account_id,
                parent_focal_at_arrival=parent_focal,
            )
            log.info(
                "step.tools_launched",
                session_id=session_id,
                count=len(immediate),
                tool_names=[_tc_name(tc) for tc in immediate],
            )

        # Unknown-MCP tools route through the regular MCP dispatcher,
        # bypassing the permission gate.  ``_execute_mcp_tool_async``
        # already detects unknown servers and appends a tool_error
        # event for them.  Routing them to immediate dispatch lets the
        # model see the error in the next step and self-correct.
        immediate_mcp = mcp_immediate + unknown_mcp
        if immediate_mcp:
            launch_mcp_tool_calls(
                pool,
                session_id,
                immediate_mcp,
                mcp_server_map,
                focal_channel=session.focal_channel,
                account_id=account_id,
                parent_focal_at_arrival=parent_focal,
            )
            log.info(
                "step.mcp_tools_launched",
                session_id=session_id,
                count=len(immediate_mcp),
                tool_names=[_tc_name(tc) for tc in immediate_mcp],
                unknown_count=len(unknown_mcp),
            )

        if needs_confirm or custom:
            log.info(
                "step.external_tools_pending",
                session_id=session_id,
                confirmations=[tc.get("id") for tc in needs_confirm if tc.get("id")],
                custom_tools=[tc.get("id") for tc in custom if tc.get("id")],
            )

    # End-of-turn is unconditional; the resulting ``status`` ({active, idle})
    # is derived from the event log per read — a session that just launched
    # background tools derives ``active`` until they resolve, without any
    # status write here (see queries._SESSION_STATUS_EXPR). We only record the
    # stop_reason of this step.
    await sessions_service.set_session_stop_reason(
        pool, session_id, {"type": "end_turn"}, account_id=account_id
    )
    await _append_lifecycle(
        pool, session_id, "turn_ended", "idle", "end_turn", account_id=account_id
    )
    log.info("step.turn_ended", session_id=session_id, cause=cause)
    # Hand the quiescence guard's wakes — and the archive-on-quiescence reclaim — to
    # run_session_step to perform AFTER step_end (the reclaim must be the step's last
    # session write; see _StepResult.archive_when_idle). The flag is the session's
    # immutable launch property, read once from the start-of-step ``session``.
    return _StepResult(
        nudge_session=nudged,
        autoerror_caller_run_id=autoerror_caller_run_id,
        autoerror_caller_session_ids=autoerror_caller_session_ids,
        archive_when_idle=session.archive_when_idle,
    )


def _switch_channel_tool_spec() -> dict[str, Any]:
    """Build the chat-completions tool entry for the ``switch_channel`` built-in.

    Injected unconditionally into the tool list when the session has
    any bound channels (see ``run_session_step``).  Agents don't need
    to list it in their ``tools`` declaration — it's focal-machinery
    scope, not agent scope.
    """
    from aios.tools.registry import openai_tool_entry
    from aios.tools.registry import registry as tool_registry

    return openai_tool_entry(tool_registry.get("switch_channel"))


async def _dump_context_if_enabled(
    session_id: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> None:
    """Write the chat-completions payload to disk when ``AIOS_DUMP_CONTEXT`` is set.

    Debug aid: inspect exactly what reaches LiteLLM (post header-inlining,
    post system-prompt augmentation, with the full tool list).
    """
    import os as _os

    if not _os.environ.get("AIOS_DUMP_CONTEXT"):
        return
    import asyncio as _asyncio
    import json as _json
    import time as _time
    from pathlib import Path as _Path

    dump_dir = _Path(_os.environ.get("AIOS_DUMP_CONTEXT_DIR", "/tmp/aios-context-dumps"))
    ts = int(_time.time() * 1000)
    path = dump_dir / f"{ts}_{session_id}.json"
    payload = {
        "session_id": session_id,
        "model": model,
        "messages": messages,
        "tools": tools,
    }

    def _write() -> None:
        dump_dir.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            _json.dump(payload, f, indent=2)

    await _asyncio.to_thread(_write)
    log.info("step.context_dumped", path=str(path))


def _tc_name(tc: dict[str, Any]) -> str:
    """Extract the function name from a tool_call dict."""
    name: str = (tc.get("function") or {}).get("name", "")
    return name


type ToolDispatchKind = Literal[
    "immediate", "mcp_immediate", "needs_confirm", "custom", "unknown_mcp"
]


def _classify_tool_call(
    tool_call: dict[str, Any],
    agent: Any,
    mcp_server_map: dict[str, McpServerSpec],
) -> ToolDispatchKind:
    """Classify a tool call into a dispatch bucket.

    Returns one of:

    * ``"immediate"`` — built-in tool, run synchronously.
    * ``"mcp_immediate"`` — known MCP tool, ``always_allow``.
    * ``"needs_confirm"`` — built-in or MCP tool gated on
      ``always_ask`` confirmation.
    * ``"custom"`` — client-executed custom tool (the harness holds
      the call until the client posts a tool-result).
    * ``"unknown_mcp"`` — MCP-namespaced tool whose server is not
      registered.  Routed to immediate tool-error so the model can
      self-correct rather than leaving the call unresolved.

    Thin projection of the single-source disposition classifier
    (:func:`aios.harness.tool_disposition.classify_tool_call`, #1076): the
    permission ladder is walked exactly once there; the three consumers
    (this dispatch path, the awaiting view, the recovery sweep) differ only
    in their terminal projection.  ``ToolDispatchKind``'s literals are the
    :class:`~aios.harness.tool_disposition.ToolDisposition` values verbatim.
    """
    # Thin projection of the single-source disposition classifier (#1076).
    # A fresh dispatch is never pre-confirmed, so confirmation_resolved=False;
    # the loop carries the mcp_server_map and so distinguishes unknown_mcp.
    function = tool_call.get("function") or {}
    name: str = function.get("name") or ""

    disposition = classify_tool_call(
        name,
        function.get("arguments"),
        agent,
        confirmation_resolved=False,
        mcp_server_map=mcp_server_map,
    )
    return disposition.value


async def discover_session_mcp_tools(
    pool: Any,
    session_id: str,
    agent: StepSurface,
    *,
    account_id: str,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Discover MCP tools from agent-declared servers, filtered by enabled
    ``mcp_toolset`` entries.

    Returns ``(tools, instructions_by_server)`` where the second element
    maps ``server_name`` → the server's ``InitializeResult.instructions``
    string.  Servers that supplied no instructions (or ``""``) are
    omitted from the dict.
    """
    from aios.mcp.client import discover_mcp_tools, resolve_auth_for_target_url
    from aios.tools.registry import effective_transport

    enabled_server_names: set[str] = set()
    for tool_spec in agent.tools:
        if tool_spec.type == "mcp_toolset" and tool_spec.enabled and tool_spec.mcp_server_name:
            enabled_server_names.add(tool_spec.mcp_server_name)
    servers: list[McpServerSpec] = [s for s in agent.mcp_servers if s.name in enabled_server_names]
    if not servers:
        return [], {}

    crypto_box = runtime.require_crypto_box()

    # Binding identity for the discovery result cache (#1391): the agent id +
    # version. The tool set is agent-definition-level and immutable within a
    # version, so caching on this key is correct — a version bump lands on a
    # fresh key (re-discovers), while a static binding serves from cache and
    # pays no per-step ``list_tools()`` RPC. Frozen-surface / version-pinned
    # sessions keep one (id, version) for the session lifetime, so they serve
    # from cache for the whole session with no rediscovery.
    binding_id = agents_service.tool_cache_binding_id(agent)

    # Circuit breaker (#1391): skip a server whose discovery recently timed out /
    # failed so one unresponsive server can't re-stall the prelude every step —
    # the agent proceeds degraded on the healthy servers' (possibly cached) tools.
    from aios.mcp.client import _DISCOVERY_UNHEALTHY_BACKOFF_S

    async def _discover_one(spec: McpServerSpec) -> tuple[list[dict[str, Any]], str | None]:
        vault_id, headers = await resolve_auth_for_target_url(
            pool, crypto_box, session_id, spec.url, account_id=account_id
        )
        _pool = runtime.mcp_session_pool
        if _pool is not None:
            from aios.mcp.client import _headers_key

            hkey = _headers_key(spec.headers)
            # A cached result is always served (cheap, no RPC) even while the
            # circuit is open; only an uncached discovery is short-circuited.
            if _pool.get_cached_tools(spec.url, vault_id, hkey, binding_id) is None and (
                _pool.is_unhealthy(spec.url, vault_id, hkey)
            ):
                raise TimeoutError(
                    f"MCP server {spec.name!r} discovery skipped: "
                    f"in backoff ({_DISCOVERY_UNHEALTHY_BACKOFF_S:.0f}s) after a prior timeout"
                )
        return await discover_mcp_tools(
            spec.url, vault_id, headers, spec.name, spec_headers=spec.headers, binding_id=binding_id
        )

    # Discovery runs as part of the step prelude — a process the model
    # didn't consciously initiate — so a single server's transport
    # failure is logged at WARN for ops visibility but does NOT surface
    # as a model-visible event. The failed server contributes neither
    # tools nor an instructions entry, so the system prompt's
    # mcp_servers_block reflects only servers the model can actually
    # use. Healthy servers' discoveries proceed unaffected.
    raw_results = await asyncio.gather(*[_discover_one(s) for s in servers], return_exceptions=True)
    tools: list[dict[str, Any]] = []
    instructions_by_server: dict[str, str] = {}
    for spec, result in zip(servers, raw_results, strict=True):
        name = spec.name
        if isinstance(result, BaseException):
            log.warning(
                "mcp.discovery_failed",
                server_name=name,
                url=spec.url,
                error=f"{type(result).__name__}: {result}",
            )
            continue
        tool_list, instructions = result
        # Filter out ``cli``-only MCP tools — the model can't see them.
        # Per-tool transport overrides via the agent's ``mcp_toolset``
        # config (default_config / configs) are resolved via the shared
        # ``effective_transport`` helper.
        for td in tool_list:
            qualified = td.get("function", {}).get("name", "")
            if effective_transport(qualified, agent.tools) == "cli":
                continue
            tools.append(td)
        if instructions:
            instructions_by_server[name] = instructions

    # Observability (#1698 (e)): emit exactly one durable ``mcp_server_unavailable``
    # session event per breaker DOWN transition. The breaker arms in the pool
    # (``acquire``/discovery) which has no session context, so it records the
    # edge and we drain + stamp it here where session_id is in scope. Deduped on
    # the breaker edge (drain empties the queue), ``is_error:false`` keeps it out
    # of error-filtered views but queryable — mirrors the ``step_timeout``
    # telemetry precedent. Gives the ops-agent an external artifact to detect a
    # session running degraded.
    _pool = runtime.mcp_session_pool
    if _pool is not None:
        down_urls = _pool.drain_degraded_events()
        if down_urls:
            url_to_name = {s.url: s.name for s in agent.mcp_servers}
            for down_url in down_urls:
                await sessions_service.append_event(
                    pool,
                    session_id,
                    "span",
                    {
                        "event": "mcp_server_unavailable",
                        "server": url_to_name.get(down_url, down_url),
                        "url": down_url,
                        "is_error": False,
                    },
                    account_id=account_id,
                )
    return tools, instructions_by_server


async def _dispatch_confirmed_tools(
    pool: Any,
    session_id: str,
    *,
    account_id: str,
    inflight_tool_registry: InflightToolRegistry,
) -> list[dict[str, Any]]:
    """Find tool calls that have been confirmed (allow) but not yet dispatched.

    Returns the original tool call dicts ready for ``launch_tool_calls``,
    or an empty list if nothing to dispatch.

    Resolution is UNWINDOWED and mirrors the sweep's case-(c) wake predicate
    (``sweep.CONFIRMED_ROWS_SQL``): a ``tool_confirmed``/``allow`` lifecycle
    event whose ``tool_call_id`` has no ``tool_result``.  Sourcing the
    dispatchable calls from the windowed message slice — as an earlier version
    did — dropped a confirmed ``always_ask`` tool whose parent assistant had
    scrolled past ``window_max`` (#737, the window-edge sibling of the
    lifecycle-``LIMIT`` bug #155).  Detection (the sweep) and dispatch (here)
    now resolve the identical predicate, so they can't disagree at any edge.

    Skips ``tool_call_id``s whose asyncio task is still in flight per
    *inflight_tool_registry*: procrastinate releases the per-session lock when step N's
    job body returns, but the fire-and-forget tool task outlives the body —
    any wake firing step N+1 before the task appends its result would otherwise
    re-launch the same tool and write a second ``tool_result`` event (violates
    CLAUDE.md invariant #4).  ``list_confirmed_unresolved_tool_calls`` applies
    the complementary already-has-a-result guard.
    """
    dispatchable = await sessions_service.list_confirmed_unresolved_tool_calls(
        pool, session_id, account_id=account_id
    )
    if not dispatchable:
        return []
    in_flight = inflight_tool_registry.in_flight_tool_call_ids(session_id)
    return [tc for tc in dispatchable if tc.get("id") not in in_flight]


async def _append_model_request_error_span(
    pool: Any,
    session_id: str,
    *,
    start_event_id: str,
    account_id: str,
    provider_error: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "event": "model_request_end",
        "model_request_start_id": start_event_id,
        "is_error": True,
        "model_usage": {},
        "cost_usd": None,
    }
    # Persist the provider failure detail (exception class + HTTP status +
    # message) so a child_errored / errored-turn span is diagnosable straight
    # from the DB — not by reconstructing the cause from span timings (#1442).
    if provider_error is not None:
        payload["provider_error"] = provider_error
    await sessions_service.append_event(
        pool,
        session_id,
        "span",
        payload,
        account_id=account_id,
    )


async def _append_harvest_consumed_marker(
    pool: Any,
    session_id: str,
    *,
    run_id: str,
    model: str,
    is_error: bool,
    account_id: str,
) -> Event:
    """Append the ``model_workflow_harvest_end`` span that marks a park *consumed* (#1634).

    Written on EVERY fold path (ok / errored / invalid-shape), exactly once per
    ``run_id`` per turn — the fold step runs once and ends owing no further park.
    :func:`aios.db.queries.events.find_latest_model_workflow_park` excludes any park
    whose run already has this span, so a folded park is un-returnable and the next
    stimulus opens a fresh park instead of re-folding the stale harvest forever. On
    the ok path the returned span doubles as the step's model-request start span
    (its id pairs the downstream ``model_request_end``).
    """
    return await sessions_service.append_event(
        pool,
        session_id,
        "span",
        {
            "event": "model_workflow_harvest_end",
            "run_id": run_id,
            "model": model,
            "is_error": is_error,
        },
        account_id=account_id,
    )


async def _latch_errored_turn(
    pool: Any,
    session_id: str,
    *,
    error_kind: str,
    stop_message: str | None = None,
    finish_reason: str | None = None,
    account_id: str,
) -> None:
    """Land a session in the terminal ``errored`` state (#353).

    Runs the fixed three-step latch in order: fail every open request on the
    session's behalf, set the ``error`` stop_reason, then append the
    ``turn_ended``/``errored`` lifecycle event. The lifecycle event puts the
    session in the derived ``errored`` state, which the sweep skips (see
    ``sweep.ERRORED_SESSIONS_SQL``); any in-flight tool task that completes
    after this point sits unreaped until a user message recovers the session
    (its seq overtakes the error event).

    The ordering is load-bearing. ``fail_all_open_requests`` MUST land BEFORE
    the lifecycle latch. A workflow child whose turn errored can no longer
    answer the requests it was invoked with, so each is failed with a monotonic
    response and its invoking run resolves (and can raise ``AgentError``)
    instead of hanging forever on a dead child (no-op for a non-child / nothing
    owed). Because the latch makes the sweep skip the session, a crash after
    latching-but-before-responding would strand the child errored-and-unanswered
    with no path to ever resolve its callers. Responses-before-latch means a
    crash instead leaves the child un-latched and recoverable — the sweep
    re-wakes it, it re-errors, and the now-written responses no-op.
    """
    await fail_all_open_requests(
        pool, session_id, account_id=account_id, error={"kind": error_kind}
    )
    stop_reason: dict[str, Any] = {"type": "error"}
    if stop_message is not None:
        stop_reason["message"] = stop_message
    if finish_reason is not None:
        stop_reason["finish_reason"] = finish_reason
    await sessions_service.set_session_stop_reason(
        pool, session_id, stop_reason, account_id=account_id
    )
    # The lifecycle ``status``/``stop_reason`` here are the SAME shared constants
    # ``append_event`` reads to bump ``last_error_seq`` and park the errored
    # session. Routing both write and read through ``aios.models.events`` is the
    # binding that survives the JSONB ``Any`` boundary (#1084); the coupling is
    # pinned by ``test_errored_lifecycle_coupling.py``.
    await _append_lifecycle(
        pool,
        session_id,
        "turn_ended",
        ERRORED_LIFECYCLE_STATUS,
        ERRORED_LIFECYCLE_STOP_REASON,
        account_id=account_id,
    )


async def _handle_streaming_model_deadline(
    pool: Any,
    session_id: str,
    *,
    start_event_id: str,
    usage: dict[str, int],
    cost_usd: float | None,
    model: str,
    account_id: str,
    provider_error: dict[str, Any] | None = None,
) -> None:
    span_payload: dict[str, Any] = {
        "event": "model_request_end",
        "model_request_start_id": start_event_id,
        "is_error": True,
        "model_usage": usage,
        "cost_usd": cost_usd,
    }
    # This errored-turn span also carries the provider failure detail so the
    # streaming-deadline cause is diagnosable from the DB directly (#1442).
    if provider_error is not None:
        span_payload["provider_error"] = provider_error
    await sessions_service.append_event(
        pool,
        session_id,
        "span",
        span_payload,
        account_id=account_id,
    )
    cost_microusd = _resolve_cost_microusd(model, usage, cost_usd, session_id=session_id)
    await sessions_service.increment_usage(
        pool,
        session_id,
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
        cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
        cost_microusd=cost_microusd,
        account_id=account_id,
    )
    deadline_s = get_settings().model_call_deadline_s
    await _latch_errored_turn(
        pool,
        session_id,
        error_kind="model_call_deadline",
        stop_message=(
            f"model call exceeded its {deadline_s:.0f}s total deadline while still "
            "streaming; partial token usage was recorded"
        ),
        account_id=account_id,
    )


async def _apply_retry_or_failure(pool: Any, session_id: str, *, account_id: str) -> float | None:
    """Apply the rescheduling state when backoff budget allows; otherwise
    mark a terminal error.

    Returns the retry delay (seconds) when a retry will be deferred, or
    ``None`` when the budget is spent and the session ends in error
    state.  Both branches advance the session's lifecycle and status;
    the caller decides whether to also propagate an exception.
    """
    attempt = await _count_consecutive_rescheduling(pool, session_id, account_id=account_id)
    delay = _retry_delay_for_attempt(attempt)
    if delay is not None:
        await sessions_service.set_session_stop_reason(
            pool, session_id, {"type": "rescheduling"}, account_id=account_id
        )
        await _append_lifecycle(
            pool,
            session_id,
            "turn_ended",
            "rescheduling",
            "rescheduling",
            account_id=account_id,
        )
        return delay
    # Terminal landing pad (#353): the retry budget is spent, so land the
    # session in the errored state. See ``_latch_errored_turn`` for the
    # responses-before-latch ordering invariant this relies on.
    await _latch_errored_turn(pool, session_id, error_kind="child_errored", account_id=account_id)
    return None


async def _handle_spend_cap(
    pool: Any,
    session_id: str,
    *,
    spent_microusd: int,
    spend_limit_usd: float,
    account_id: str,
) -> None:
    """Latch a session that has reached its account spend limit."""
    await sessions_service.append_event(
        pool,
        session_id,
        "span",
        {
            "event": "spend_cap_exceeded",
            "is_error": True,
            "spent_microusd": spent_microusd,
            "spend_limit_usd": spend_limit_usd,
        },
        account_id=account_id,
    )
    await _latch_errored_turn(
        pool,
        session_id,
        error_kind="spend_cap_exceeded",
        stop_message=_SPEND_CAP_STOP_REASON_MESSAGE,
        account_id=account_id,
    )


async def _handle_refusal(
    pool: Any,
    session_id: str,
    assistant_msg: dict[str, Any],
    *,
    finish_reason: str,
    account_id: str,
) -> None:
    """Latch a model refusal (``finish_reason == content_filter``) as an errored turn.

    The refused assistant message is NOT persisted as a normal turn (it would
    replay as poison) and its tool calls are NOT dispatched (they may be
    truncated). The partial content/tool_calls are stashed on a ``span`` event
    for debugging — ``build_messages`` only replays ``kind == "message"`` events,
    so the span never re-enters context. The session then lands in the terminal
    ``errored`` state via the shared latch used by every terminal-error handler
    (see ``_latch_errored_turn``): a workflow child's open requests are
    failed so its callers resolve, the stop_reason latches to ``error`` (drives
    the console "Errored" pill), and the error lifecycle event bumps
    ``last_error_seq`` so the sweep parks the session until a user message
    recovers it.
    """
    await sessions_service.append_event(
        pool,
        session_id,
        "span",
        {
            "event": "model_refusal",
            "is_error": True,
            "finish_reason": finish_reason,
            # Kept for debugging only — a span is never replayed by build_messages.
            "partial_content": assistant_msg.get("content") or "",
            "partial_tool_calls": assistant_msg.get("tool_calls") or [],
        },
        account_id=account_id,
    )
    await _latch_errored_turn(
        pool,
        session_id,
        error_kind="model_refusal",
        stop_message=_REFUSAL_STOP_REASON_MESSAGE,
        finish_reason=finish_reason,
        account_id=account_id,
    )


async def _handle_step_timeout(pool: Any, session_id: str, *, account_id: str) -> float | None:
    """Synthesize a reschedulable error state when the job-level cap fires."""
    await sessions_service.append_event(
        pool,
        session_id,
        "span",
        {"event": "step_timeout", "timeout_seconds": HARNESS_STEP_TIMEOUT_S, "is_error": True},
        account_id=account_id,
    )
    return await _apply_retry_or_failure(pool, session_id, account_id=account_id)


async def _count_consecutive_rescheduling(pool: Any, session_id: str, *, account_id: str) -> int:
    """Count consecutive rescheduling lifecycle events at the tail of the log.

    Returns the number of consecutive ``turn_ended`` lifecycle events
    with ``stop_reason == "rescheduling"`` at the end of the lifecycle
    event sequence. A non-rescheduling event breaks the streak.
    """
    # Only the tail matters; reading ASC with the default LIMIT would miss the
    # recent streak entirely on a session with >limit lifecycle events.
    lifecycle_events = await sessions_service.read_events(
        pool,
        session_id,
        kind="lifecycle",
        newest_first=True,
        limit=len(_RETRY_BACKOFF_SECONDS) + 1,
        account_id=account_id,
    )
    count = 0
    for e in lifecycle_events:
        if e.data.get("event") == "turn_ended" and e.data.get("stop_reason") == "rescheduling":
            count += 1
        else:
            break
    return count


async def _append_lifecycle(
    pool: Any,
    session_id: str,
    event: str,
    status: str,
    stop_reason: str,
    *,
    account_id: str,
) -> None:
    """Append a lifecycle event."""
    await sessions_service.append_event(
        pool,
        session_id,
        "lifecycle",
        {"event": event, "status": status, "stop_reason": stop_reason},
        account_id=account_id,
    )
