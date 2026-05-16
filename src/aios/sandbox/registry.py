"""Per-worker in-memory registry of live session sandboxes.

The registry makes sandbox provisioning lazy: a tool handler asks for
a session's :class:`SandboxHandle`; the registry either returns the
cached handle or builds a :class:`ProvisioningPlan` (DB queries +
GitProxy + clone materialization) and asks its :class:`SandboxBackend`
to bring a sandbox up.

One registry instance per worker process, stashed on
:mod:`aios.harness.runtime`. The procrastinate ``lock`` parameter
ensures only one step runs per session at a time; the registry's own
per-session ``asyncio.Lock`` serializes provisioning against drift
detection so the registry never returns a handle that's about to be
torn down.

Sandbox lifecycle is decoupled from step lifecycle via an idle-TTL
reaper. Sandboxes stay alive across consecutive steps for the same
session and are released when idle for longer than
``container_idle_timeout_seconds``. Worker shutdown calls
:meth:`release_all` to clean up everything.

The registry owns the per-session :class:`GitProxy`, not the handle —
the proxy is a host-side process whose lifetime tracks the session, not
the sandbox container. Releasing a session stops its proxy (if any) in
addition to destroying the sandbox.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from aios.logging import get_logger
from aios.sandbox.backends.base import CommandResult, SandboxBackend, SandboxHandle
from aios.sandbox.git_proxy import GitProxy
from aios.sandbox.network import WORKER_NETWORK_ALIAS
from aios.sandbox.setup import (
    apply_network_lockdown,
    ensure_workspace_runtime_dirs,
    install_packages,
)
from aios.sandbox.spec import (
    ProvisioningPlan,
    build_spec_from_session,
    mount_snapshot_from_echoes,
)

if TYPE_CHECKING:
    import asyncpg

    from aios.models.github_repositories import GithubRepositoryResourceEcho
    from aios.models.memory_stores import MemoryStoreResourceEcho

log = get_logger("aios.sandbox.registry")


class SandboxRegistry:
    """Maps session_id to a live :class:`SandboxHandle` with idle-TTL.

    Construct with the chosen :class:`SandboxBackend`; the registry holds
    it for the worker's lifetime and dispatches all provision/exec/destroy
    calls through it. Backend swaps require restarting the worker (the
    registry doesn't re-discover existing sandboxes from a different
    backend).
    """

    def __init__(self, backend: SandboxBackend) -> None:
        self._backend = backend
        self._handles: dict[str, SandboxHandle] = {}
        self._git_proxies: dict[str, GitProxy] = {}
        self._last_used: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._reaper_task: asyncio.Task[None] | None = None

    @property
    def backend(self) -> SandboxBackend:
        return self._backend

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
    ) -> SandboxHandle:
        """Return the cached handle, or provision a new sandbox.

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
    ) -> SandboxHandle:
        if pool is None:
            return await self._provision(session_id)

        from aios.services import sessions as sessions_service

        account_id = await sessions_service.load_session_account_id(pool, session_id)
        span_start = await sessions_service.append_event(
            pool, session_id, "span", {"event": "sandbox_provision_start"}, account_id=account_id
        )
        is_error = False
        handle: SandboxHandle | None = None
        try:
            handle = await self._provision(session_id)
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
                end_payload["container_id"] = handle.sandbox_id[:12]
            await sessions_service.append_event(
                pool, session_id, "span", end_payload, account_id=account_id
            )

    async def _provision(self, session_id: str) -> SandboxHandle:
        """Build the plan, ask the backend to create the sandbox, run setup."""
        plan = await build_spec_from_session(session_id)

        # Record the GitProxy before backend.create so a failure midway
        # has us in a state where release() will find and stop it.
        if plan.git_proxy is not None:
            self._git_proxies[session_id] = plan.git_proxy

        try:
            handle = await self._backend.create(plan.spec)
        except BaseException:
            # Spec was built (proxy may be running, broker secret is
            # registered) but the backend couldn't create the sandbox.
            # Drop both so their port + token map + secret map don't leak.
            self._git_proxies.pop(session_id, None)
            if plan.git_proxy is not None:
                await self._stop_proxy_silently(plan.git_proxy, session_id)
            self._release_mcp_broker_secret(session_id)
            raise

        # Setup steps after create. If any of these raise, tear the
        # sandbox down so we don't leak an empty container alongside
        # the proxy.
        try:
            await ensure_workspace_runtime_dirs(self._backend, handle)
            await install_packages(self._backend, handle, plan.env_config)
            await self._maybe_apply_lockdown(handle, plan)
        except BaseException:
            await self._destroy_quietly(handle, session_id)
            raise

        log.info(
            "sandbox.provisioned",
            session_id=session_id,
            container_id=handle.sandbox_id[:12],
            workspace_path=str(handle.workspace_path),
            backend=self._backend.name,
            networking=type(plan.spec.network_policy).__name__,
        )
        return handle

    async def _maybe_apply_lockdown(self, handle: SandboxHandle, plan: ProvisioningPlan) -> None:
        """Apply network lockdown if the plan calls for it."""
        from aios.harness import runtime
        from aios.models.environments import LimitedNetworking

        networking = plan.env_config.networking if plan.env_config else None
        if not isinstance(networking, LimitedNetworking):
            return
        extra_host_ports: list[tuple[str, int]] = [
            (WORKER_NETWORK_ALIAS, runtime.require_mcp_broker().port),
        ]
        if plan.git_proxy is not None:
            extra_host_ports.append((WORKER_NETWORK_ALIAS, plan.git_proxy.port))
        await apply_network_lockdown(
            self._backend,
            handle,
            networking,
            extra_host_ports=extra_host_ports,
        )

    async def _stop_proxy_silently(self, proxy: GitProxy, session_id: str) -> None:
        """Stop ``proxy``, log + swallow any error.

        Used by every cleanup path; a stuck proxy must never block
        sandbox teardown or propagate a secondary exception over a
        primary one.
        """
        try:
            await proxy.stop()
        except Exception as err:
            log.warning(
                "sandbox.git_proxy_stop_failed",
                session_id=session_id,
                error=str(err),
            )

    def _release_mcp_broker_secret(self, session_id: str) -> None:
        """Drop the per-session entry from the MCP broker's secret map.

        Idempotent: a missing entry is silently ignored. Called at every
        sandbox teardown site so the broker doesn't accumulate dangling
        secrets for sessions whose sandboxes are gone.
        """
        from aios.harness import runtime

        runtime.require_mcp_broker().unregister_session(session_id)

    async def _destroy_quietly(self, handle: SandboxHandle, session_id: str) -> None:
        """Tear down the handle's sandbox and stop the session's proxy.

        Used by the partial-failure cleanup path during provisioning, so
        cleanup errors are warnings (not raises). Caller's exception is
        propagating.
        """
        try:
            await self._backend.destroy(handle)
        except Exception as err:
            log.warning(
                "sandbox.destroy_during_cleanup_failed",
                session_id=session_id,
                container_id=handle.sandbox_id[:12],
                error=str(err),
            )
        proxy = self._git_proxies.pop(session_id, None)
        if proxy is not None:
            await self._stop_proxy_silently(proxy, session_id)
        self._release_mcp_broker_secret(session_id)

    async def exec(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        timeout_seconds: int,
        max_output_bytes: int,
        cwd: str = "/workspace",
    ) -> CommandResult:
        """Run ``command`` inside ``handle``'s sandbox via the backend."""
        return await self._backend.exec(
            handle,
            command,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
            cwd=cwd,
        )

    async def release(self, session_id: str) -> None:
        """Tear down one session's sandbox + proxy. No-op if not cached.

        Also clears worker-process runtime caches keyed on this session
        (``_session_memory_mounts``, ``_session_read_shas`` in
        :mod:`aios.harness.runtime`). Without this cleanup, every
        session that ran a step on this worker left an entry that
        persisted for the worker's process lifetime — slow unbounded
        growth across long-running workers handling many sessions. The
        caches are documented as "after session unload" but had no
        production caller until this hook. Re-populated naturally by
        the next step if the session wakes back up.

        Deferred import of ``aios.harness.runtime`` matches the existing
        pattern at :meth:`_provision_with_span` (line 191) and
        :meth:`_release_mcp_broker_secret`-adjacent paths — the runtime
        module lists ``aios.sandbox.registry`` under ``TYPE_CHECKING``,
        so a top-level import here would create a runtime cycle.

        Cleanup is sequenced BEFORE ``backend.destroy`` so the
        "registry says unloaded ⇒ runtime caches cleared" invariant
        holds even when ``destroy`` raises (Docker hiccup): a
        half-finished release shouldn't leave stale per-session caches
        pretending the session is still live.
        """
        from aios.harness import runtime

        handle = self._handles.pop(session_id, None)
        self._last_used.pop(session_id, None)
        self._locks.pop(session_id, None)
        proxy = self._git_proxies.pop(session_id, None)

        if proxy is not None:
            await self._stop_proxy_silently(proxy, session_id)
        self._release_mcp_broker_secret(session_id)
        runtime.clear_session_memory_mounts(session_id)
        runtime.clear_session_read_shas(session_id)
        if handle is None:
            return
        await self._backend.destroy(handle)

    async def release_if_mounts_changed(
        self,
        session_id: str,
        memory_echoes: list[MemoryStoreResourceEcho],
        github_echoes: list[GithubRepositoryResourceEcho],
    ) -> None:
        """Release the cached sandbox if its mount snapshot has drifted.

        Acquires the per-session lock so a tool task from a prior step
        can't race with the release via ``get_or_provision``.

        Drift includes: memory store attachments added/removed, github
        repos added/removed, github token rotated (the ``updated_at``
        timestamp on the github echo is part of the snapshot key, see
        :func:`mount_snapshot_from_echoes`).
        """
        async with self._lock_for(session_id):
            handle = self._handles.get(session_id)
            if handle is None:
                return
            if handle.mount_snapshot == mount_snapshot_from_echoes(memory_echoes, github_echoes):
                return
            log.info(
                "sandbox.released_for_mount_change",
                session_id=session_id,
                container_id=handle.sandbox_id[:12],
            )
            # Delegate to release() so the runtime-cache cleanup and any
            # future unload-event side effects stay in one place. Holding
            # the per-session lock guards against a tool task from a prior
            # step racing with the release via get_or_provision (which
            # acquires the same lock).
            await self.release(session_id)

    def peek(self, session_id: str) -> SandboxHandle | None:
        """Return the cached handle without provisioning. ``None`` if not cached."""
        return self._handles.get(session_id)

    def evict(self, session_id: str) -> None:
        """Drop the cache entry without backend teardown (sandbox is dead).

        Used by tool_dispatch when it detects a sandbox-side failure that
        suggests the sandbox itself is unhealthy; the next tool call will
        cold-start a fresh one.
        """
        from aios.harness import runtime

        self._last_used.pop(session_id, None)
        self._locks.pop(session_id, None)
        if self._handles.pop(session_id, None) is not None:
            log.info("sandbox.evicted", session_id=session_id)
        # Drop the proxy too — a fresh sandbox will get a fresh proxy
        # from the next provision pass.
        proxy = self._git_proxies.pop(session_id, None)
        if proxy is not None:
            # Fire-and-forget — eviction is best-effort cleanup; worker
            # shutdown's ``release_all`` is the authoritative final stop.
            asyncio.create_task(self._stop_proxy_silently(proxy, session_id))  # noqa: RUF006
        # Same story for the MCP broker secret: drop it immediately; a
        # fresh sandbox will get a fresh secret from the next provision.
        self._release_mcp_broker_secret(session_id)
        # Drop the runtime read-sha cache so a fresh sandbox doesn't
        # serve a stale precondition match against a different
        # container's file state. Memory mounts are re-populated at the
        # top of every step (loop.py:87) so leaving them is harmless,
        # but symmetry argues for clearing both.
        runtime.clear_session_memory_mounts(session_id)
        runtime.clear_session_read_shas(session_id)

    async def release_all(self) -> None:
        """Tear down every sandbox + proxy. Called at worker shutdown."""
        handles = list(self._handles.values())
        proxies = list(self._git_proxies.values())
        self._handles.clear()
        self._last_used.clear()
        self._locks.clear()
        self._git_proxies.clear()

        if proxies:
            proxy_results = await asyncio.gather(
                *(p.stop() for p in proxies), return_exceptions=True
            )
            for _p, result in zip(proxies, proxy_results, strict=True):
                if isinstance(result, BaseException):
                    log.warning(
                        "sandbox.release_all_proxy_error",
                        error=str(result),
                    )

        if not handles:
            return
        log.info("sandbox.release_all", count=len(handles))
        results = await asyncio.gather(
            *(self._backend.destroy(h) for h in handles), return_exceptions=True
        )
        for h, result in zip(handles, results, strict=True):
            if isinstance(result, BaseException):
                log.warning(
                    "sandbox.release_all_error",
                    session_id=h.session_id,
                    container_id=h.sandbox_id[:12],
                    error=str(result),
                )

    async def reap_orphans(self, active_session_ids: Iterable[str]) -> int:
        """At startup, remove sandboxes not matching an active session."""
        from aios.config import get_settings

        instance_id = get_settings().instance_id
        try:
            managed = await self._backend.list_managed(instance_id=instance_id)
        except Exception as err:
            log.warning("sandbox.reap_list_failed", error=str(err))
            return 0
        if not managed:
            return 0

        active = set(active_session_ids)
        removed = 0
        for ref in managed:
            if ref.session_id and ref.session_id in active:
                continue
            log.info(
                "sandbox.reap_orphan",
                container_id=ref.sandbox_id[:12],
                session_id=ref.session_id or "<no-label>",
            )
            try:
                await self._backend.force_remove(ref.sandbox_id)
                removed += 1
            except Exception as err:
                log.warning(
                    "sandbox.reap_remove_failed",
                    container_id=ref.sandbox_id[:12],
                    error=str(err),
                )
        return removed

    # ── idle-TTL reaper ──────────────────────────────────────────────────

    async def _reap_idle_loop(self, idle_timeout: float, interval: float = 60.0) -> None:
        """Background loop: release sandboxes idle > idle_timeout seconds.

        The try/except is nested INSIDE ``while True`` (mirroring
        :func:`aios.harness.worker._periodic_sweep` and the PR #443 fix
        for ``_run_interrupt_listener``): a Docker daemon hiccup raising
        ``SandboxBackendError`` from ``release()`` must not silently
        disable the reaper for the worker's lifetime — that would
        leak every subsequent idle sandbox until process exit.
        ``CancelledError`` is not an :class:`Exception` subclass, so
        ``stop_reaper()``'s cancel still exits the loop cleanly.
        """
        while True:
            try:
                await asyncio.sleep(interval)
                now = time.monotonic()
                to_release: list[str] = []
                for sid, last in list(self._last_used.items()):
                    if now - last > idle_timeout:
                        to_release.append(sid)
                for sid in to_release:
                    log.info("sandbox.idle_release", session_id=sid)
                    await self.release(sid)
            except Exception:
                log.exception("sandbox.reap_idle_loop_failed")

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
