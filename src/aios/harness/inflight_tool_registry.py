"""Registry of in-flight asyncio tool tasks per session.

Tracks ``asyncio.Task`` objects spawned by the async tool dispatcher so they
can be cancelled (by the pg-notify interrupt listener, or at worker shutdown)
and so the worker knows which sessions have background work in progress.

One instance per worker process, stashed on :mod:`aios.harness.runtime`.
"""

from __future__ import annotations

import asyncio

from aios.logging import get_logger

log = get_logger("aios.harness.inflight_tool_registry")


class InflightToolRegistry:
    """Maps ``session_id`` → ``{tool_call_id → asyncio.Task}``."""

    def __init__(self) -> None:
        self._tasks: dict[str, dict[str, asyncio.Task[None]]] = {}
        self._dispatch_seq: dict[str, dict[str, int | None]] = {}
        self._step_tasks: dict[str, asyncio.Task[None]] = {}
        self._step_start_seq: dict[str, int | None] = {}

    def add(self, session_id: str, tool_call_id: str, task: asyncio.Task[None]) -> None:
        """Register a newly-launched tool task.

        Auto-captures the CURRENT ``start_seq`` of *session_id*'s registered
        step (from :meth:`register_step`), if any, as this task's
        ``dispatch_seq`` (#1756) — no call-site change needed: every live
        dispatch (``launch_tool_calls``/``launch_mcp_tool_calls``) runs
        synchronously inside the step body, AFTER that step has already
        called :meth:`register_step`, so the lookup always sees the
        launching step's seq. A cold dispatch outside any registered step for
        this session (the periodic sweep's ghost re-park) captures ``None`` —
        deliberately: a re-parked task is recovering pre-crash work with no
        current step to attribute it to, so :meth:`cancel_session`'s
        ``min_start_seq`` treats a ``None`` dispatch_seq as always-cancellable
        (conservative in the direction of "an interrupt should win").
        """
        session_tasks = self._tasks.setdefault(session_id, {})
        session_tasks[tool_call_id] = task
        session_seqs = self._dispatch_seq.setdefault(session_id, {})
        session_seqs[tool_call_id] = self._step_start_seq.get(session_id)

    def remove(self, session_id: str, tool_call_id: str) -> None:
        session_tasks = self._tasks.get(session_id)
        if session_tasks is not None:
            session_tasks.pop(tool_call_id, None)
            if not session_tasks:
                del self._tasks[session_id]
        session_seqs = self._dispatch_seq.get(session_id)
        if session_seqs is not None:
            session_seqs.pop(tool_call_id, None)
            if not session_seqs:
                del self._dispatch_seq[session_id]

    def cancel_session(self, session_id: str, *, min_start_seq: int | None = None) -> int:
        """Cancel in-flight tool tasks for a session. Returns count cancelled.

        Callers: the LIVE pg-notify interrupt listener (``worker.py``), which
        runs from a non-registered listener task — so there is no in-set task
        to skip (the prior self-skip existed only for the now-deleted in-band
        ``cancel`` tool, #1458) — and passes no ``min_start_seq`` (unconditional
        cancel: an interrupt landing on a live LISTEN connection has no
        ambiguity — every currently in-flight task predates it).

        ``min_start_seq`` (#1756), when given, is the RECONNECT RE-DRIVE's
        guard: only a task whose ``dispatch_seq`` (captured at :meth:`add`
        time, from the launching step's ``start_seq``) is STRICTLY LESS than
        ``min_start_seq`` — or has no recorded ``dispatch_seq`` at all (a cold
        ghost re-park) — is cancelled. A task dispatched by a step that began
        AT OR AFTER ``min_start_seq`` (a legitimate post-interrupt follow-up
        step's own fresh dispatch) is left alone — the same non-cancel-the-
        follow-up guard :meth:`cancel_step` applies to the step task itself.
        """
        session_tasks = self._tasks.get(session_id)
        if not session_tasks:
            return 0
        session_seqs = self._dispatch_seq.get(session_id, {})
        count = 0
        for tool_call_id, task in list(session_tasks.items()):
            if task.done():
                continue
            if min_start_seq is not None:
                dispatch_seq = session_seqs.get(tool_call_id)
                if dispatch_seq is not None and dispatch_seq >= min_start_seq:
                    continue
            task.cancel()
            count += 1
        if count:
            log.info("session.tasks_cancelled", session_id=session_id, count=count)
        return count

    def in_flight_tool_call_ids(self, session_id: str) -> set[str]:
        """Return tool_call_ids with in-flight tasks for a session."""
        session_tasks = self._tasks.get(session_id)
        if not session_tasks:
            return set()
        return {tcid for tcid, task in session_tasks.items() if not task.done()}

    async def wait_for_session_tools(self, session_id: str) -> None:
        """Wait until every tool task currently running for a session is done.

        The task map is re-read after each drain. This catches sibling tool
        dispatches added while the first batch was completing and gives
        lifecycle operations a point after which no active sandbox command is
        still using the session's container.
        """
        while True:
            tasks = [task for task in self._tasks.get(session_id, {}).values() if not task.done()]
            if not tasks:
                return
            await asyncio.gather(*tasks, return_exceptions=True)

    def all_in_flight_tool_call_ids(self) -> dict[str, set[str]]:
        """Return ``{session_id: {tool_call_ids}}`` for all sessions with in-flight tasks."""
        result: dict[str, set[str]] = {}
        for sid, tasks in self._tasks.items():
            active = {tcid for tcid, t in tasks.items() if not t.done()}
            if active:
                result[sid] = active
        return result

    def tracked_session_ids(self) -> set[str]:
        """Return every ``session_id`` with a live in-flight tool task and/or a
        registered step task on THIS worker.

        Consumed by the interrupt-listener reconnect re-drive (#1756,
        ``worker._run_interrupt_listener``): the re-drive only needs to
        re-check sessions this worker could actually still be running work
        for — a cheap local-memory scope, no cross-worker DB scan.
        """
        live_task_sids = {
            sid for sid, tasks in self._tasks.items() if any(not t.done() for t in tasks.values())
        }
        live_step_sids = {sid for sid, t in self._step_tasks.items() if not t.done()}
        return live_task_sids | live_step_sids

    def register_step(
        self, session_id: str, task: asyncio.Task[None], *, start_seq: int | None = None
    ) -> None:
        """Register the currently-running step task for *session_id*.

        ``start_seq`` (#1756) is the seq of the step's own ``step_start`` span
        event — the event-log position the step began at. It lets a
        seq-bounded caller (:meth:`cancel_step`'s ``min_start_seq``) tell a
        STALE step (one that began before a given event) from a step that
        began AFTER it — e.g. a legitimate post-interrupt follow-up step,
        which must never be cancelled by a re-drive of an OLDER interrupt.
        ``None`` (the default) opts a caller out of that comparison — every
        pre-#1756 call site that doesn't pass it keeps unconditional-cancel
        behavior.
        """
        self._step_tasks[session_id] = task
        self._step_start_seq[session_id] = start_seq

    def unregister_step(self, session_id: str) -> None:
        self._step_tasks.pop(session_id, None)
        self._step_start_seq.pop(session_id, None)

    def cancel_step(self, session_id: str, *, min_start_seq: int | None = None) -> bool:
        """Cancel the registered step task for *session_id*.

        ``min_start_seq`` (#1756), when given, gates the cancel: it only
        fires if the registered step's ``start_seq`` (from
        :meth:`register_step`) is STRICTLY LESS than ``min_start_seq`` — i.e.
        the step began before whatever event ``min_start_seq`` denotes (an
        interrupt seq, for the reconnect re-drive). A step registered with no
        ``start_seq``, or one that began AT OR AFTER ``min_start_seq``, is
        left alone — the load-bearing guard against a re-drive cancelling a
        legitimate post-interrupt follow-up step. ``None`` (the default)
        preserves the unconditional-cancel behavior the live pg-notify
        listener call site relies on (an interrupt landing during a LIVE
        LISTEN connection has no ambiguity to resolve: any step running at
        that moment predates the interrupt that just arrived).
        """
        task = self._step_tasks.get(session_id)
        if task is None or task.done():
            return False
        if min_start_seq is not None:
            start_seq = self._step_start_seq.get(session_id)
            if start_seq is not None and start_seq >= min_start_seq:
                return False
        task.cancel()
        log.info("step.cancelled", session_id=session_id)
        return True

    async def shutdown(self) -> None:
        """Cancel all tasks and await them for cleanup (finally blocks)."""
        all_tasks: list[asyncio.Task[None]] = []
        for session_tasks in self._tasks.values():
            for task in session_tasks.values():
                if not task.done():
                    task.cancel()
                    all_tasks.append(task)
        if all_tasks:
            log.info("inflight_tool_registry.shutdown", count=len(all_tasks))
            await asyncio.gather(*all_tasks, return_exceptions=True)
        self._tasks.clear()
