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
        self._step_tasks: dict[str, asyncio.Task[None]] = {}

    def add(self, session_id: str, tool_call_id: str, task: asyncio.Task[None]) -> None:
        session_tasks = self._tasks.setdefault(session_id, {})
        session_tasks[tool_call_id] = task

    def remove(self, session_id: str, tool_call_id: str) -> None:
        session_tasks = self._tasks.get(session_id)
        if session_tasks is not None:
            session_tasks.pop(tool_call_id, None)
            if not session_tasks:
                del self._tasks[session_id]

    def cancel_session(self, session_id: str) -> int:
        """Cancel all in-flight tool tasks for a session. Returns count cancelled.

        Sole caller: the pg-notify interrupt listener (``worker.py``), which runs from a
        non-registered listener task — so there is no in-set task to skip (the prior
        self-skip existed only for the now-deleted in-band ``cancel`` tool, #1458).
        """
        session_tasks = self._tasks.get(session_id)
        if not session_tasks:
            return 0
        count = 0
        for _tool_call_id, task in list(session_tasks.items()):
            if not task.done():
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

    def all_in_flight_tool_call_ids(self) -> dict[str, set[str]]:
        """Return ``{session_id: {tool_call_ids}}`` for all sessions with in-flight tasks."""
        result: dict[str, set[str]] = {}
        for sid, tasks in self._tasks.items():
            active = {tcid for tcid, t in tasks.items() if not t.done()}
            if active:
                result[sid] = active
        return result

    def register_step(self, session_id: str, task: asyncio.Task[None]) -> None:
        self._step_tasks[session_id] = task

    def unregister_step(self, session_id: str) -> None:
        self._step_tasks.pop(session_id, None)

    def cancel_step(self, session_id: str) -> bool:
        task = self._step_tasks.get(session_id)
        if task is None or task.done():
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
