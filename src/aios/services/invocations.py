"""``await_invocation`` — the one awaiter over the invocation tree (§3.2).

Merges the session and run completion long-polls into a single dispatch over the
edge handle ``(servicer_kind, servicer_id, request_id)``. One
:class:`~aios.models.invocations.AwaitResponse` ``{outcome, result, error}`` out;
``outcome=None`` means still pending (the caller re-polls).

The **awaiter's** terminal rule lives here:

* a **run**'s outcome is read off its terminal row — ``completed → ok`` (output on
  the row), ``errored → errored`` (``error.kind`` from the ``run_completed`` event
  via :func:`resolve_run_error`), ``cancelled → cancelled``. Status and ``is_error``
  never diverge (``workflows/step.py`` derives one from the other), so the row is
  authoritative — this retires the hand-rolled copy that used to live in ``await_run``.
* a **session**'s outcome is read off its ``request_response`` event via
  :func:`derive_response` (the same monotonic resolver the run harvest uses).

The 3-valued ``outcome`` *is* the trace's ``TerminalState`` minus liveness (folded to
``None``), so this awaiter and the **display** normalizer
(:mod:`aios.services.trace_normalizer`) classify the same states — they must stay in
lockstep. They are deliberately NOT merged: the awaiter returns the full ``error``
payload + ``outcome=None`` for pending and reads lazily (no event fetch on the ``ok``
path), whereas the normalizer returns ``(TerminalState, error_kind)`` for the trace
view; a shared core would be an adapter over a 6-line function (net complexity up).

Both arms subscribe BEFORE the first predicate read (LISTEN-before-read) and drive
the shared :func:`await_completion` loop. The watermark/quiescence session-await
(``await_session``) has no run analog and stays an orthogonal session-only alias.
"""

from __future__ import annotations

from typing import Any, Literal

import asyncpg

from aios.db import queries
from aios.db.listen import open_listen_for_events, open_listen_for_run_events
from aios.db.queries import trace as trace_q
from aios.db.queries import workflows as wf_queries
from aios.errors import NotFoundError, ValidationError
from aios.models.invocations import AwaitResponse, OpenInvocation
from aios.models.workflows import TERMINAL_RUN_STATUSES, WfRun
from aios.services.await_completion import await_completion

ServicerKind = Literal["session", "run"]


async def await_invocation(
    pool: asyncpg.Pool[Any],
    db_url: str,
    *,
    servicer_kind: ServicerKind,
    servicer_id: str,
    request_id: str | None,
    account_id: str,
    timeout_seconds: float,
) -> AwaitResponse:
    """Block until the servicer reaches a terminal state, or ``timeout_seconds``.

    Dispatches on ``servicer_kind``: a ``session`` resolves via its
    ``request_response`` event (so ``request_id`` is required), a ``run`` resolves
    off its terminal row (``request_id`` ignored). A cross-tenant/missing
    ``servicer_id`` 404s before any LISTEN opens. On timeout returns
    ``outcome=None`` so the caller re-polls.
    """
    if servicer_kind == "session":
        if request_id is None:
            raise ValidationError("await of a session servicer requires request_id")
        return await _await_session(
            pool,
            db_url,
            servicer_id,
            account_id=account_id,
            request_id=request_id,
            timeout_seconds=timeout_seconds,
        )
    return await _await_run(
        pool, db_url, servicer_id, account_id=account_id, timeout_seconds=timeout_seconds
    )


async def _await_session(
    pool: asyncpg.Pool[Any],
    db_url: str,
    session_id: str,
    *,
    account_id: str,
    request_id: str,
    timeout_seconds: float,
) -> AwaitResponse:
    # Scope-check FIRST (404s cross-tenant/missing before any LISTEN opens): the
    # watermark read enforces ``WHERE id = $1 AND account_id = $2`` and returns
    # None when the row is missing or cross-tenant — the same guarantee as get_session.
    async with pool.acquire() as conn:
        if await queries.read_session_watermarks(conn, session_id, account_id=account_id) is None:
            raise NotFoundError(f"session {session_id} not found", detail={"id": session_id})

    async def _read() -> dict[str, Any] | None:
        async with pool.acquire() as conn:
            return await queries.derive_response(
                conn, session_id, account_id=account_id, request_id=request_id
            )

    # on_connected=None omits the #81 subscriber lock: an await poller consumes
    # only the terminal completion state, never the deltas, so it must not force
    # the awaited session's worker onto the streaming path.
    subscription = await open_listen_for_events(db_url, session_id, on_connected=None)
    try:
        resolved = await await_completion(
            subscription.queue,
            read_state=_read,
            is_done=lambda state: state is not None,
            timeout_seconds=timeout_seconds,
        )
    finally:
        subscription.terminate()
    return _response_to_await(resolved)


async def _await_run(
    pool: asyncpg.Pool[Any],
    db_url: str,
    run_id: str,
    *,
    account_id: str,
    timeout_seconds: float,
) -> AwaitResponse:
    async with pool.acquire() as conn:
        await wf_queries.get_wf_run(conn, run_id, account_id=account_id)  # 404s cross-tenant

    async def _read() -> WfRun:
        async with pool.acquire() as conn:
            return await wf_queries.get_wf_run(conn, run_id, account_id=account_id)

    subscription = await open_listen_for_run_events(db_url, run_id)
    try:
        run = await await_completion(
            subscription.queue,
            read_state=_read,
            is_done=lambda r: r.status in TERMINAL_RUN_STATUSES,
            timeout_seconds=timeout_seconds,
        )
    finally:
        subscription.terminate()

    # Row-derived terminal rule — the awaiter twin of trace_normalizer.normalize_run_root
    # (which returns TerminalState + error_kind for display); keep the two in lockstep.
    if run.status not in TERMINAL_RUN_STATUSES:
        return AwaitResponse(outcome=None)  # timed out, still running
    if run.status == "completed":
        return AwaitResponse(outcome="ok", result=run.output)
    if run.status == "cancelled":
        # A cancelled run deliberately writes no request_response; surface it as a
        # clean ``cancelled`` outcome (error.kind for a model caller that branches).
        return AwaitResponse(outcome="cancelled", error={"kind": "cancelled"})
    # errored — error.kind lives only in the run_completed payload, not on the row.
    async with pool.acquire() as conn:
        error = await wf_queries.resolve_run_error(conn, run_id)
    return AwaitResponse(outcome="errored", error=error)


def _response_to_await(resolved: dict[str, Any] | None) -> AwaitResponse:
    """Map a :func:`derive_response` ``{result, is_error, error}`` (or ``None``) to an envelope.

    The one place the resolver's ``is_error`` bit becomes the 3-valued ``outcome``:
    a ``cancelled``-kinded error surfaces as ``cancelled`` (uniform child-death
    notify), any other error as ``errored``, a clean response as ``ok``. Keep the
    error→outcome classification in lockstep with
    :func:`aios.services.trace_normalizer.normalize_response` (the display twin).
    """
    if resolved is None:
        return AwaitResponse(outcome=None)
    if resolved.get("is_error"):
        error = resolved.get("error")
        kind = error.get("kind") if isinstance(error, dict) else None
        outcome: Literal["errored", "cancelled"] = "cancelled" if kind == "cancelled" else "errored"
        return AwaitResponse(outcome=outcome, error=error)
    return AwaitResponse(outcome="ok", result=resolved.get("result"))


async def cancel_invocation(
    pool: asyncpg.Pool[Any],
    *,
    servicer_kind: ServicerKind,
    servicer_id: str,
    request_id: str | None,
    account_id: str,
    canceller_session_id: str | None = None,
) -> None:
    """Cancel an invocation by its edge handle (cancel-design §2) — the supervisor seed.

    Seeds the exit on the servicer so it harvests the cancel under its OWN single-writer
    step lock: a **run** via the existing ``wf_run_signals kind='cancel'`` + harvest
    (``cancel_run``); a **session** via a ``session_cancel_markers`` row that the C2 sweep
    wakes into its leaf. Account-scoped (cross-tenant/missing 404s before any write);
    idempotent (the marker is an ON CONFLICT no-op, so a re-cancel is a no-op).

    Seeds the ROOT only; the recursion is each marked node's own leaf job. A **session**
    root cascades DOWN its subtree: ``harvest_session_cancel_markers`` answers ``cancelled``
    and re-seeds markers on its awaited children (§2.3 — built). The **run-down** cascade —
    a cancelled run re-seeding its own children — is NOT built: a run root finalizes as a
    single node (#788/#1152), and the §9 quiescence accounting rides with it.

    ``request_id`` is ``None`` for a **run** servicer (it cancels off its terminal row, not a
    request edge — same dispatch asymmetry as :func:`await_invocation`); a **session** servicer
    still requires it (the cancel-marker keys on the request edge). ``canceller_session_id``
    scopes a **model**-initiated cancel (the ``stop_task`` tool): the run arm threads it into
    ``cancel_run``'s launcher guard so a session may cancel only runs it launched. The operator
    HTTP path leaves it ``None`` — account-scoped, unguarded — turning the run-cancel launcher
    guard from prose-held into construction-held for the model plane.
    """
    if servicer_kind == "run":
        from aios.services import workflows as wf_service

        # ``cancel_run`` 404s cross-tenant, seeds the run cancel signal, and wakes the run.
        # ``canceller_session_id`` (set on the model plane) gates it to runs that session
        # launched; the operator path passes None (account-scoped).
        await wf_service.cancel_run(
            pool,
            run_id=servicer_id,
            account_id=account_id,
            canceller_session_id=canceller_session_id,
        )
        return

    # session: scope-check, seed the exit-marker in one txn, then wake the leaf.
    if request_id is None:
        raise ValidationError("cancel of a session servicer requires request_id")
    async with pool.acquire() as conn, conn.transaction():
        if await queries.read_session_watermarks(conn, servicer_id, account_id=account_id) is None:
            raise NotFoundError(f"session {servicer_id} not found", detail={"id": servicer_id})
        await queries.insert_session_cancel_marker(
            conn, session_id=servicer_id, request_id=request_id, account_id=account_id
        )
    from aios.services.wake import defer_wake

    await defer_wake(pool, servicer_id, cause="cancel", account_id=account_id)


async def list_open_invocations(
    pool: asyncpg.Pool[Any],
    *,
    session_id: str,
    account_id: str,
) -> list[OpenInvocation]:
    """A session's still-open outbound ``call_*`` invocations — backs the ``list_tasks`` tool (#1428).

    Reads the caller's whole edge roster (:func:`aios.db.queries.trace.list_caller_invocations`)
    under one ``REPEATABLE READ`` readonly snapshot, then resolves each edge's liveness with the
    same #1126 resolvers the trace/await paths use — ``derive_response`` for a session servicer,
    ``derive_run_response`` for a run — and keeps only the **open** ones (``resp is None``, i.e.
    still pending). One snapshot so the roster and the per-edge liveness can't tear (the
    ``services.trace.get_trace`` pattern). Returns the open invocations keyed by ``tool_call_id``
    (the handle ``stop_task`` takes), oldest-first. A point-in-time snapshot, independent of the
    in-context ``_PENDING`` placeholders ``build_messages`` synthesizes — both converge next step.
    """
    async with (
        pool.acquire() as conn,
        conn.transaction(isolation="repeatable_read", readonly=True),
    ):
        edges = await trace_q.list_caller_invocations(
            conn, caller_session_id=session_id, account_id=account_id
        )
        open_invocations: list[OpenInvocation] = []
        for edge in edges:
            if edge.servicer_kind == "session":
                # A session servicer always carries a request_id (the request edge it answers).
                assert edge.request_id is not None
                resp = await queries.derive_response(
                    conn, edge.servicer_id, account_id=account_id, request_id=edge.request_id
                )
            else:
                resp = await wf_queries.derive_run_response(
                    conn, edge.servicer_id, account_id=account_id
                )
            if resp is None:  # alive and unanswered → an open task
                open_invocations.append(
                    OpenInvocation(
                        tool_call_id=edge.tool_call_id,
                        kind=edge.servicer_kind,
                        target=edge.servicer_id,
                        opened_at=edge.opened_at,
                    )
                )
    return open_invocations
