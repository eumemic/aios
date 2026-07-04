"""Run-side park + harvest for the ``workflow:`` model binding (issue #1634).

Part of the **Workflows-as-Models** epic. The session-side counterpart of the
binding boundary (:mod:`aios.harness.model_binding`): the durable park/harvest
plumbing that lets a step's inference be produced by a workflow run **without
awaiting inline** (the whole step is under the 960s cap, and "stay responsive
while parked" is a tool-task property — so the inner deliberation must span
wakes, never block one step).

**Async two-step.**

* **Park (step N).** :func:`launch_model_workflow_park` opens an **awaited** run
  of the bound workflow (reusing ``launch_awaited_run`` — the run side ships) and
  journals a ``model_workflow_park`` event carrying the run id and the sealed
  ``reacting_to`` watermark. It then spawns a fire-and-forget task that parks on
  the run (exactly like the ``call_*`` builtins) and, on resolution, writes a
  ``model_workflow_harvest`` event with the run's structured output + wakes the
  session. The step ends **owing an assistant message** — it does NOT await.

* **Harvest (step N+1).** :func:`take_pending_harvest` reads the latest
  *un-consumed* park and its matching harvest (if the run has resolved). When
  present it returns a :class:`HarvestedInference` the step folds into
  ``assistant_msg`` and runs the existing append/charge/dispatch tail — **no
  re-charge** (the inner inference charged once at its own ``call_llm`` site; the
  harvest records only a span) and the ``reacting_to`` it sealed at park (never
  recomputed at harvest). Folding the harvest writes a ``model_workflow_harvest_end``
  span for the run id, which marks the park **consumed**: the next read excludes it
  (:func:`aios.db.queries.events.find_latest_model_workflow_park`), so a later
  stimulus opens a fresh park rather than re-folding the stale harvest.

The park/harvest pair keys off the run id, so a session can only owe one parked
inference at a time (a step that parks ends the turn; the next inference is the
harvest). The harvest event carries the raw run outcome — the binding boundary
(:func:`aios.harness.model_binding.map_run_output_to_response`) projects it into
an ``LlmResponse`` at harvest time, where a malformed shape fails loud.
"""

from __future__ import annotations

import asyncio
import enum
from dataclasses import dataclass
from typing import Any

import asyncpg

from aios.harness.completion import LlmRequest
from aios.harness.model_binding import WorkflowModelRef
from aios.logging import get_logger
from aios.services import sessions as sessions_service
from aios.services import tasks as tasks_service
from aios.services import workflows as wf_service

log = get_logger("aios.harness.model_workflow")

# Strong refs to in-flight park tasks — without this the event loop only holds a
# weak reference and may GC a fire-and-forget task mid-await (RUF006). The task
# discards itself on completion; the sweep re-park is the crash backstop.
_PARK_TASKS: set[asyncio.Task[None]] = set()

# The ``(session_id, run_id)`` keys of harvest tasks currently in-flight in THIS
# worker — the model-dispatch analog of ``InflightToolRegistry`` for ``call_*``
# parks (#1635). The crash-recovery sweep re-parks a session whose unharvested
# model-dispatch park has NO live harvest task; this set lets a re-park skip a key
# already serviced in-process (the harvest write is idempotent, so a double-launch
# is safe — but checking spares the redundant ``await_task`` poll). Cleared at
# worker boot via :func:`reset_inflight_harvests`, before the boot sweep runs, so a
# stale key from a previous loop can never mask a genuinely lost task.
_INFLIGHT_HARVESTS: set[tuple[str, str]] = set()


def reset_inflight_harvests() -> None:
    """Drop all in-flight-harvest keys (worker boot, before the crash-recovery sweep).

    At boot every harvest task from the previous loop is provably gone (the worker
    died), so the set must start empty — otherwise a stale key would make the boot
    sweep treat a genuinely lost park as still-serviced and skip its re-park (#1635).
    """
    _INFLIGHT_HARVESTS.clear()


def inflight_harvest_keys() -> frozenset[tuple[str, str]]:
    """The ``(session_id, run_id)`` keys of harvest tasks live in THIS worker (#1635)."""
    return frozenset(_INFLIGHT_HARVESTS)


# Event ``kind`` is ``"span"`` (excluded from the message-replay window like the
# other harness bookkeeping events); the ``event`` discriminator names the role.
PARK_EVENT = "model_workflow_park"
HARVEST_EVENT = "model_workflow_harvest"

# Per-park await poll budget — bounded so an unbounded park can't pin a connection
# forever between polls (mirrors ``invoke_session._AWAIT_POLL_SECONDS``).
_AWAIT_POLL_SECONDS = 300.0


class ParkState(enum.Enum):
    """Three-way disposition of a session's parked workflow-model inference.

    Distinguishing :data:`PARK_PENDING` from "no park" is the load-bearing fix for
    the multi-dispatch / multi-billing defect (#1634 review): a parked step writes
    a ``span`` event, which does NOT advance ``last_stimulus_seq`` /
    ``last_reacted_seq`` — so the unreacted-stimulus inequality that caused the park
    STILL holds afterwards and the periodic sweep re-wakes the session every tick
    while the inner run deliberates. If the harvest read collapses "park open &
    unresolved" into the same ``None`` as "no park", the caller re-parks on every
    re-wake → N parallel paid inner runs per turn. Surfacing :data:`PARK_PENDING`
    lets the caller end the step WITHOUT launching a second run, so **exactly one
    inner awaited run is launched per workflow-model turn** regardless of how many
    sweep ticks elapse during deliberation.
    """

    #: No open park — the caller should launch a fresh awaited run (the park branch).
    NO_PARK = "no_park"
    #: A park is open but its run has not resolved — the caller ends the step owing
    #: the assistant message again, re-parking on NOTHING (no new run). The harvest
    #: task's ``defer_wake`` (or a sweep re-wake) lands the resolved harvest later.
    PARK_PENDING = "park_pending"


@dataclass(frozen=True, slots=True)
class HarvestedInference:
    """A resolved bound-workflow inference ready to fold into the dispatch tail.

    * ``outcome`` — the awaiter outcome (``"ok"`` / ``"errored"`` / ``"cancelled"``).
    * ``output`` — the run's structured return on ``ok`` (the binding boundary maps
      it to an ``LlmResponse``); ``None`` otherwise.
    * ``error`` — the ``{kind, message, …}`` detail on a non-``ok`` outcome.
    * ``reacting_to`` — the watermark sealed at park, re-applied to the harvested
      assistant turn (NOT recomputed at harvest).
    * ``run_id`` — the bound run that produced this inference (for the span).
    """

    outcome: str
    output: Any
    error: dict[str, Any] | None
    reacting_to: int
    run_id: str


async def launch_model_workflow_park(
    pool: asyncpg.Pool[Any],
    session_id: str,
    *,
    ref: WorkflowModelRef,
    request: LlmRequest,
    reacting_to: int,
    account_id: str,
) -> str:
    """Open an awaited run of the bound workflow and park owing an assistant message.

    Journals the ``model_workflow_park`` event (run id + sealed ``reacting_to``),
    then spawns the fire-and-forget harvest task. Returns the bound run id. The
    caller ends the step after this — the inner deliberation resolves async and a
    later wake harvests it.
    """
    session = await sessions_service.get_session_basic(pool, session_id, account_id=account_id)
    # The inference payload is delivered as the run's ``input`` — a bound workflow
    # receives the same named ``LlmRequest`` shape ``call_llm`` consumes, so it can
    # forward it to its own ``call_llm`` leaf or deliberate over it.
    run_input = {
        "messages": request.messages,
        "tools": request.tools,
        "params": request.params,
        "session_id": request.session_id,
    }
    run, _request_id = await wf_service.launch_awaited_run(
        pool,
        account_id=account_id,
        workflow_id=ref.workflow_id,
        version=ref.version,
        environment_id=session.environment_id,
        input=run_input,
        # ``purpose='model_dispatch'`` is the tool_call_id-less servicer discriminant
        # (#1635): a ``call_*`` park stamps a ``tool_call_id`` on its caller edge, but
        # a model-dispatch park has none (the assistant message is the run's OUTPUT,
        # produced only after the park resolves). This marker is what
        # ``find_unharvested_model_dispatch_parks`` keys on so the crash-recovery sweep
        # can re-derive the park's run from the durable edge alone.
        caller={"kind": "session", "id": session_id, "purpose": "model_dispatch"},
        launcher_session_id=session_id,
        parent_run_id=session.parent_run_id,
    )
    # Seal ``reacting_to`` at park — the harvest re-applies this exact watermark to
    # the assistant turn (it is NOT recomputed when the run resolves, so a stimulus
    # that arrives mid-deliberation does not retroactively widen what the turn
    # "reacted to").
    await sessions_service.append_event(
        pool,
        session_id,
        "span",
        {
            "event": PARK_EVENT,
            "run_id": run.id,
            "reacting_to": reacting_to,
        },
        account_id=account_id,
    )
    _launch_harvest_task(pool, session_id, run_id=run.id, account_id=account_id)
    log.info(
        "model_workflow.parked",
        session_id=session_id,
        run_id=run.id,
        workflow_id=ref.workflow_id,
        version=ref.version,
    )
    return run.id


def _launch_harvest_task(
    pool: asyncpg.Pool[Any], session_id: str, *, run_id: str, account_id: str
) -> None:
    """Spawn the fire-and-forget park task (held in ``_PARK_TASKS``; the sweep is the backstop).

    The ``(session_id, run_id)`` key is registered in ``_INFLIGHT_HARVESTS`` for the task's
    lifetime so the crash-recovery sweep (#1635) does not re-park a key already serviced in
    this worker — and cleared in the done-callback (it fires on success, error, AND cancel,
    so a key is never leaked).
    """
    key = (session_id, run_id)
    _INFLIGHT_HARVESTS.add(key)
    task = asyncio.create_task(
        _park_and_signal(pool, session_id, run_id=run_id, account_id=account_id),
        name=f"model_workflow_park:{session_id}:{run_id}",
    )
    _PARK_TASKS.add(task)

    def _done(t: asyncio.Task[None]) -> None:
        _PARK_TASKS.discard(t)
        _INFLIGHT_HARVESTS.discard(key)

    task.add_done_callback(_done)


def relaunch_model_dispatch_park(
    pool: asyncpg.Pool[Any], session_id: str, *, run_id: str, account_id: str
) -> bool:
    """Re-park a stranded model-dispatch harvest task whose in-memory task was lost (#1635).

    The model-dispatch analog of :func:`aios.harness.tool_dispatch.relaunch_parked_task`.
    A worker crash takes the fire-and-forget harvest task down with it; on restart the
    crash-recovery sweep re-derives the unharvested park
    (:func:`aios.db.queries.find_unharvested_model_dispatch_parks`) and calls this to spawn
    a fresh harvest task. The task is a pure read of the run's durable terminal state
    followed by one idempotent harvest append (dedup-guarded on ``run_id``), so a re-park
    that races a real harvest is safe — first write wins.

    Returns ``True`` if a task was launched, ``False`` if one is already in-flight in THIS
    worker for ``(session_id, run_id)`` (the live task will write the harvest; re-parking
    would only duplicate the ``await_task`` poll). The sweep counts launches.
    """
    if (session_id, run_id) in _INFLIGHT_HARVESTS:
        return False
    _launch_harvest_task(pool, session_id, run_id=run_id, account_id=account_id)
    log.info("model_workflow.reparked", session_id=session_id, run_id=run_id)
    return True


async def _park_and_signal(
    pool: asyncpg.Pool[Any], session_id: str, *, run_id: str, account_id: str
) -> None:
    """Park on the bound run; on resolution write the harvest event + wake the session.

    A pure read of the run's durable terminal state followed by one harvest-event
    append — re-entrant, so a re-park after a worker crash (the sweep re-derives
    the unharvested park) lands the same harvest with no double effect (the
    harvest append is dedup-guarded on the run id).
    """
    from aios.config import get_settings
    from aios.jobs.app import defer_wake

    db_url = get_settings().db_url
    try:
        while True:
            resp = await tasks_service.await_task(
                pool,
                db_url,
                servicer_kind="run",
                servicer_id=run_id,
                request_id=None,
                account_id=account_id,
                timeout_seconds=_AWAIT_POLL_SECONDS,
            )
            if resp.outcome is not None:
                break
        await write_harvest_event(
            pool,
            session_id,
            run_id=run_id,
            outcome=resp.outcome,
            output=resp.result,
            error=resp.error,
            account_id=account_id,
        )
        await defer_wake(pool, session_id, cause="model_workflow_harvest", account_id=account_id)
        log.info(
            "model_workflow.harvest_signalled",
            session_id=session_id,
            run_id=run_id,
            outcome=resp.outcome,
        )
    except asyncio.CancelledError:
        # Worker shutdown: no harvest written. The crash-recovery sweep (#1635)
        # re-derives the unharvested park (``find_unharvested_model_dispatch_parks``)
        # and re-parks via ``relaunch_model_dispatch_park`` — the durable backstop that
        # keeps the turn from stranding when this in-process task dies with the worker.
        raise
    except Exception:
        log.exception("model_workflow.park_failed", session_id=session_id, run_id=run_id)


async def write_harvest_event(
    pool: asyncpg.Pool[Any],
    session_id: str,
    *,
    run_id: str,
    outcome: str,
    output: Any,
    error: dict[str, Any] | None,
    account_id: str,
) -> None:
    """Append the ``model_workflow_harvest`` event for a resolved bound run.

    Idempotent on ``run_id``: a re-park that races the first harvest (or a sweep
    re-drive) does not append a second harvest for the same run — the existing
    harvest is the one the next step folds in.
    """
    if await _harvest_exists(pool, session_id, run_id=run_id, account_id=account_id):
        return
    await sessions_service.append_event(
        pool,
        session_id,
        "span",
        {
            "event": HARVEST_EVENT,
            "run_id": run_id,
            "outcome": outcome,
            "output": output,
            "error": error,
        },
        account_id=account_id,
    )


async def _harvest_exists(
    pool: asyncpg.Pool[Any], session_id: str, *, run_id: str, account_id: str
) -> bool:
    from aios.db import queries

    async with pool.acquire() as conn:
        harvest = await queries.find_model_workflow_harvest(
            conn, session_id, run_id=run_id, account_id=account_id
        )
    return harvest is not None


async def take_pending_harvest(
    pool: asyncpg.Pool[Any], session_id: str, *, account_id: str
) -> HarvestedInference | ParkState:
    """Disposition the latest open park: harvest, still-pending, or none.

    Returns one of three values (the three-way distinction is the multi-dispatch
    fix — see :class:`ParkState`):

    * :data:`ParkState.NO_PARK` — the session owes no parked inference whose run is
      still unharvested. The caller launches a fresh awaited run (the park branch).
    * :data:`ParkState.PARK_PENDING` — a park is open but its run has NOT resolved
      yet (no harvest event). The caller ends the step owing the assistant message
      again WITHOUT launching a second run; a later wake (the harvest task's
      ``defer_wake``, or a sweep re-wake) re-enters and harvests.
    * :class:`HarvestedInference` — the open park's run has resolved; folded into the
      dispatch tail with the park's sealed ``reacting_to``.

    A malformed park (no usable run id) is treated as :data:`ParkState.NO_PARK` — it
    cannot be harvested and must not wedge the turn; the caller re-parks.
    """
    from aios.db import queries

    async with pool.acquire() as conn:
        # The latest park that has NOT yet been consumed. A park is consumed once
        # its harvest is folded (any fold path writes a ``model_workflow_harvest_end``
        # span for its run id); ``find_latest_model_workflow_park`` excludes parks
        # with that marker, so once a turn folds its harvest this returns ``None``
        # and a fresh stimulus re-enters the NO_PARK / launch branch instead of
        # re-folding the same (now stale) harvest forever.
        park = await queries.find_latest_model_workflow_park(
            conn, session_id, account_id=account_id
        )
        if park is None:
            return ParkState.NO_PARK
        run_id = park.get("run_id")
        if not isinstance(run_id, str):
            return ParkState.NO_PARK
        harvest = await queries.find_model_workflow_harvest(
            conn, session_id, run_id=run_id, account_id=account_id
        )
    if harvest is None:
        # Park open, run unresolved: do NOT re-park (no new run) — end the step owing
        # the message; the harvest's ``defer_wake`` (or a sweep re-wake) re-enters.
        return ParkState.PARK_PENDING
    return HarvestedInference(
        outcome=str(harvest.get("outcome")),
        output=harvest.get("output"),
        error=harvest.get("error"),
        reacting_to=int(park.get("reacting_to") or 0),
        run_id=run_id,
    )
