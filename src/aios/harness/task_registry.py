"""Registry of in-flight asyncio tool tasks per session.

Tracks ``asyncio.Task`` objects spawned by the async tool dispatcher so
they can be cancelled (by the model via the ``cancel`` tool, or at
worker shutdown) and so the worker knows which sessions have background
work in progress.

One instance per worker process, stashed on :mod:`aios.harness.runtime`.
"""

from __future__ import annotations

import asyncio

from aios.logging import get_logger

log = get_logger("aios.harness.task_registry")


class TaskRegistry:
    """Maps ``session_id`` → ``{tool_call_id → asyncio.Task}``."""

    def __init__(self) -> None:
        self._tasks: dict[str, dict[str, asyncio.Task[None]]] = {}

    def add(self, session_id: str, tool_call_id: str, task: asyncio.Task[None]) -> None:
        session_tasks = self._tasks.setdefault(session_id, {})
        session_tasks[tool_call_id] = task

    def remove(self, session_id: str, tool_call_id: str) -> None:
        session_tasks = self._tasks.get(session_id)
        if session_tasks is not None:
            session_tasks.pop(tool_call_id, None)
            if not session_tasks:
                del self._tasks[session_id]

    def cancel_task(self, session_id: str, tool_call_id: str) -> bool:
        """Cancel one tool task. Returns True if the task was found and cancelled."""
        session_tasks = self._tasks.get(session_id)
        if session_tasks is None:
            return False
        task = session_tasks.get(tool_call_id)
        if task is None or task.done():
            return False
        task.cancel()
        log.info("task.cancelled", session_id=session_id, tool_call_id=tool_call_id)
        return True

    def cancel_session(self, session_id: str) -> int:
        """Cancel all in-flight tool tasks for a session. Returns count cancelled."""
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

    def in_flight_count(self, session_id: str) -> int:
        session_tasks = self._tasks.get(session_id)
        if not session_tasks:
            return 0
        return sum(1 for t in session_tasks.values() if not t.done())

    async def shutdown(self) -> None:
        """Cancel all tasks and await them for cleanup (finally blocks)."""
        all_tasks: list[asyncio.Task[None]] = []
        for session_tasks in self._tasks.values():
            for task in session_tasks.values():
                if not task.done():
                    task.cancel()
                    all_tasks.append(task)
        if all_tasks:
            log.info("task_registry.shutdown", count=len(all_tasks))
            await asyncio.gather(*all_tasks, return_exceptions=True)
        self._tasks.clear()
