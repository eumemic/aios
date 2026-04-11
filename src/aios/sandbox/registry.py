"""Per-worker in-memory registry of live session containers.

The registry is the glue that makes container provisioning lazy. A tool
handler asks the registry for a session's :class:`ContainerHandle`;
the registry either returns the cached handle (second or later tool call
in the same turn) or calls :func:`provision_for_session` to create a
fresh one (first tool call in a session that doesn't have a container
yet).

One registry instance per worker process, stashed on
:mod:`aios.harness.runtime`. Workers do not share state across processes
— two workers running the same session would both have their own
container, but the DB lease ensures only one worker runs a given session
at a time.

Thread-safety: all methods are ``async`` but the internal dict is
manipulated from a single asyncio event loop. Two tool calls in the same
session can't run concurrently (the harness loop dispatches them
sequentially), but two different sessions running on the same worker
can. The per-session lock prevents a provision race if both sessions'
first tool call lands in the same tick.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable

from aios.logging import get_logger
from aios.sandbox.container import ContainerHandle
from aios.sandbox.provisioner import force_remove, list_managed_containers, provision_for_session
from aios.sandbox.provisioner import release as provisioner_release

log = get_logger("aios.sandbox.registry")


class SandboxRegistry:
    """Maps session_id to a live :class:`ContainerHandle`.

    Public API:

    * :meth:`get_or_provision` — return the cached handle or create one
    * :meth:`release` — tear down one session's container (turn ended)
    * :meth:`release_all` — tear down every container (worker shutdown)
    * :meth:`reap_orphans` — at startup, remove any leftover
      ``aios.managed=true`` containers that don't match an active lease
    * :meth:`evict` — drop the cache entry for a session without running
      teardown, used after a container-death error when the registry
      should forget the dead container and let the next tool call
      provision a fresh one
    """

    def __init__(self) -> None:
        self._handles: dict[str, ContainerHandle] = {}
        # One lock per session_id to serialize provisioning. Entries are
        # created on first touch and never removed (the worker's session
        # count is bounded by worker_concurrency, so the leak is minor
        # and simplifies the code).
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock_for(self, session_id: str) -> asyncio.Lock:
        lock = self._locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[session_id] = lock
        return lock

    async def get_or_provision(self, session_id: str) -> ContainerHandle:
        """Return the cached handle for ``session_id``, or create one.

        Serialized per session so concurrent tool calls don't race to
        provision two containers. In practice the harness dispatches
        tool calls sequentially within a turn, so contention is zero on
        the hot path.
        """
        # Fast path: already cached.
        handle = self._handles.get(session_id)
        if handle is not None:
            return handle

        async with self._lock_for(session_id):
            # Re-check under the lock — another tool call might have
            # provisioned while we were waiting.
            handle = self._handles.get(session_id)
            if handle is not None:
                return handle
            handle = await provision_for_session(session_id)
            self._handles[session_id] = handle
            return handle

    async def release(self, session_id: str) -> None:
        """Tear down the container for ``session_id`` if one exists.

        Called from the harness loop's finally block after the turn ends
        (lease released). No-op if the session never provisioned a
        container (chat-only turn).
        """
        handle = self._handles.pop(session_id, None)
        if handle is None:
            return
        await provisioner_release(handle)

    def evict(self, session_id: str) -> None:
        """Drop the cache entry without running teardown.

        Used by the tool dispatcher after a container-death error: the
        container is already gone or unusable, so we just forget about
        it and let the next tool call provision a fresh one. Does NOT
        shell out to ``docker rm`` — the caller should not block on
        that on the error path.
        """
        if self._handles.pop(session_id, None) is not None:
            log.info("sandbox.evicted", session_id=session_id)

    async def release_all(self) -> None:
        """Tear down every container in the registry.

        Called at worker shutdown. Runs the teardowns concurrently —
        all of them need to happen, none of them depend on each other.
        """
        handles = list(self._handles.values())
        self._handles.clear()
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
        """Remove any leftover ``aios.managed=true`` containers.

        At worker startup, list every container with the managed label
        and compare against ``active_session_ids`` (sessions the DB says
        have live leases). Force-remove anything else — those are
        corpses from a previous worker that crashed.

        Returns the number of containers removed. Logs but does not
        raise on individual removal errors; the orphan reaper is
        best-effort.
        """
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
