"""Per-worker in-memory registry of live session containers.

The registry makes container provisioning lazy: a tool handler asks for
a session's :class:`ContainerHandle`; the registry either returns the
cached handle or calls :func:`provision_for_session` to create a fresh
one.

One registry instance per worker process, stashed on
:mod:`aios.harness.runtime`. The procrastinate ``lock`` parameter
ensures only one step runs per session at a time.

Container lifecycle is decoupled from step lifecycle via an idle-TTL
reaper. Containers stay alive across consecutive steps for the same
session and are released when idle for longer than
``container_idle_timeout_seconds``. Worker shutdown calls
:meth:`release_all` to clean up everything.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from aios.logging import get_logger
from aios.sandbox.container import ContainerHandle
from aios.sandbox.provisioner import force_remove, list_managed_containers, provision_for_session
from aios.sandbox.provisioner import release as provisioner_release

if TYPE_CHECKING:
    import asyncpg

log = get_logger("aios.sandbox.registry")


class SandboxRegistry:
    """Maps session_id to a live :class:`ContainerHandle` with idle-TTL."""

    def __init__(self) -> None:
        self._handles: dict[str, ContainerHandle] = {}
        self._last_used: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._reaper_task: asyncio.Task[None] | None = None

    def _lock_for(self, session_id: str) -> asyncio.Lock:
        lock = self._locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[session_id] = lock
        return lock

    async def get_or_provision(
        self,
        session_id: str,
        *,
        pool: asyncpg.Pool[Any] | None = None,
    ) -> ContainerHandle:
        """Return the cached handle, or provision a new container.

        Passing ``pool`` emits a ``sandbox_provision_*`` span pair on
        the cold-start path only (issue #78) — warm hits stay
        zero-observable-cost.
        """
        handle = self._handles.get(session_id)
        if handle is not None:
            self._last_used[session_id] = time.monotonic()
            return handle

        async with self._lock_for(session_id):
            handle = self._handles.get(session_id)
            if handle is not None:
                self._last_used[session_id] = time.monotonic()
                return handle
            handle = await self._provision_with_span(session_id, pool=pool)
            self._handles[session_id] = handle
            self._last_used[session_id] = time.monotonic()
            return handle

    async def _provision_with_span(
        self, session_id: str, *, pool: asyncpg.Pool[Any] | None
    ) -> ContainerHandle:
        if pool is None:
            return await provision_for_session(session_id)

        from aios.services import sessions as sessions_service

        span_start = await sessions_service.append_event(
            pool, session_id, "span", {"event": "sandbox_provision_start"}
        )
        is_error = False
        handle: ContainerHandle | None = None
        try:
            handle = await provision_for_session(session_id)
            return handle
        except Exception:
            is_error = True
            raise
        finally:
            end_payload: dict[str, Any] = {
                "event": "sandbox_provision_end",
                "sandbox_provision_start_id": span_start.id,
                "is_error": is_error,
            }
            if handle is not None:
                end_payload["container_id"] = handle.container_id[:12]
            await sessions_service.append_event(pool, session_id, "span", end_payload)

    async def release(self, session_id: str) -> None:
        """Tear down one session's container. No-op if not cached."""
        handle = self._handles.pop(session_id, None)
        self._last_used.pop(session_id, None)
        if handle is None:
            return
        await provisioner_release(handle)

    def evict(self, session_id: str) -> None:
        """Drop the cache entry without docker teardown (container is dead)."""
        self._last_used.pop(session_id, None)
        if self._handles.pop(session_id, None) is not None:
            log.info("sandbox.evicted", session_id=session_id)

    async def release_all(self) -> None:
        """Tear down every container. Called at worker shutdown."""
        handles = list(self._handles.values())
        self._handles.clear()
        self._last_used.clear()
        if not handles:
            return
        log.info("sandbox.release_all", count=len(handles))
        results = await asyncio.gather(
            *(provisioner_release(h) for h in handles), return_exceptions=True
        )
        for h, result in zip(handles, results, strict=True):
            if isinstance(result, BaseException):
                log.warning(
                    "sandbox.release_all_error",
                    session_id=h.session_id,
                    container_id=h.container_id[:12],
                    error=str(result),
                )

    async def reap_orphans(self, active_session_ids: Iterable[str]) -> int:
        """At startup, remove containers not matching an active session."""
        try:
            managed = await list_managed_containers()
        except Exception as err:
            log.warning("sandbox.reap_list_failed", error=str(err))
            return 0
        if not managed:
            return 0

        active = set(active_session_ids)
        removed = 0
        for container_id, session_id in managed:
            if session_id and session_id in active:
                continue
            log.info(
                "sandbox.reap_orphan",
                container_id=container_id[:12],
                session_id=session_id or "<no-label>",
            )
            try:
                await force_remove(container_id)
                removed += 1
            except Exception as err:
                log.warning(
                    "sandbox.reap_remove_failed",
                    container_id=container_id[:12],
                    error=str(err),
                )
        return removed

    # ── idle-TTL reaper ──────────────────────────────────────────────────

    async def _reap_idle_loop(self, idle_timeout: float, interval: float = 60.0) -> None:
        """Background loop: release containers idle > idle_timeout seconds."""
        while True:
            await asyncio.sleep(interval)
            now = time.monotonic()
            to_release: list[str] = []
            for sid, last in list(self._last_used.items()):
                if now - last > idle_timeout:
                    to_release.append(sid)
            for sid in to_release:
                log.info("sandbox.idle_release", session_id=sid)
                await self.release(sid)

    def start_reaper(self, idle_timeout: float = 300.0) -> None:
        """Start the idle-TTL reaper background task."""
        if self._reaper_task is not None:
            return
        self._reaper_task = asyncio.create_task(
            self._reap_idle_loop(idle_timeout),
            name="sandbox-idle-reaper",
        )

    def stop_reaper(self) -> None:
        """Cancel the idle-TTL reaper."""
        if self._reaper_task is not None:
            self._reaper_task.cancel()
            self._reaper_task = None
