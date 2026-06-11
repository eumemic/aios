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
from typing import TYPE_CHECKING, Any

from aios.db import queries
from aios.logging import get_logger
from aios.sandbox.backends.base import CommandResult, SandboxBackend, SandboxHandle
from aios.sandbox.git_proxy import GitProxy
from aios.sandbox.network import WORKER_NETWORK_ALIAS
from aios.sandbox.setup import (
    apply_network_lockdown,
    ensure_workspace_runtime_dirs,
    install_egress_ca,
    install_packages,
)
from aios.sandbox.spec import (
    ProvisioningPlan,
    build_spec_from_session,
    cleanup_session_secret_file,
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
        # Strong refs for evict()'s fire-and-forget proxy-stop tasks.
        # asyncio only weak-refs tasks, so without this the task can be
        # GC'd before stop() unwinds the proxy's uvicorn server, httpx
        # client and bound TCP port.
        self._evict_proxy_stop_tasks: set[asyncio.Task[None]] = set()

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
        the cold-start path only (issue #78).

        Every warm hit probes ``backend.is_alive(handle)`` before
        returning. Issue #691: ``--rm`` containers that died between
        provisions (entrypoint exit, OOM kill, daemon-side cleanup)
        leave the cached handle dangling — the registry can't observe
        the loss until the next backend call fails. The probe costs one
        round-trip per warm hit and closes the common case: a not-alive
        answer is treated as a cache miss; the dead container's host-side
        resources are recycled and a fresh sandbox provisioned under the
        per-session lock.

        A second warm-hit probe re-reads ``sessions.spec_version`` and
        recycles the (still-alive) sandbox when it has drifted past the
        snapshot stamped on the handle (issue #713). A memory store or
        github repo attached/detached between steps bumps ``spec_version``
        via a Postgres trigger; the next step would otherwise reuse a
        sandbox whose mounts no longer match
        :func:`build_spec_from_session`. The version probe only runs when
        a ``pool`` is supplied (the cold-start span path already passes
        one) and is best-effort — a transient DB error returns the live
        handle rather than churning a healthy sandbox. A drifted version
        triggers a recycle: the alive container is explicitly destroyed
        via ``_destroy_quietly`` before re-provisioning so it doesn't
        run until the next worker restart the way a dead-container evict
        would.

        Best-effort, not a hard guarantee. The probe runs lock-free, so a
        container can still die between the probe and the caller's use,
        and a concurrent ``release``/reaper (which DO hold the per-session
        lock) can tear down a handle this warm probe just validated. Those
        callers cold-fail one backend call and self-correct on the next
        step — same as before this fix. The probe removes the *steady-
        state* staleness (a container that died turns sweeps earlier),
        not every TOCTOU window.
        """
        handle = self._handles.get(session_id)
        spec_version_drifted = False
        if handle is not None and await self._backend.is_alive(handle):
            if pool is None or not await self._spec_version_changed(session_id, handle, pool):
                self._last_used[session_id] = time.monotonic()
                return handle
            spec_version_drifted = True
            log.info(
                "sandbox.spec_version_drift_recycling",
                session_id=session_id,
                container_id=handle.sandbox_id[:12],
                handle_spec_version=handle.spec_version,
            )

        # ``handle`` is the dead or spec-version-drifted reference we just
        # probed (or None on a cold miss). Capture it: under the lock, a
        # *different* cached handle means a concurrent caller already
        # recycled+reprovisioned while we waited, and we trust theirs without
        # a second probe (they validated it micro-seconds ago via
        # backend.create). Only when the cached handle is still the same one
        # do we recycle it.
        stale = handle
        async with self._lock_for(session_id):
            current = self._handles.get(session_id)
            if current is not None and current is not stale:
                self._last_used[session_id] = time.monotonic()
                return current
            if current is not None:
                if spec_version_drifted:
                    # Container is still alive — must destroy it, not just
                    # evict (evict drops the cache entry but skips
                    # backend.destroy, which is correct for dead containers
                    # but would leak a live one). _destroy_quietly is
                    # best-effort so a Docker hiccup doesn't block
                    # re-provisioning (#713).
                    self.evict(session_id, unload_session_caches=False)
                    await self._destroy_quietly(current, session_id)
                else:
                    log.warning(
                        "sandbox.stale_handle_recycling",
                        session_id=session_id,
                        container_id=current.sandbox_id[:12],
                    )
                    # Recycle the dead container's host-side resources (proxy,
                    # broker secret) but DELIBERATELY keep the session-level
                    # runtime caches: we're mid-step and about to hand a fresh
                    # sandbox back to the same step. ``_session_memory_mounts``
                    # in particular is consumed by the bash memory-reconcile
                    # that runs right after this returns (bash.py snapshots it
                    # *after* get_or_provision); clearing it here would make
                    # the before/after diff empty and silently drop the step's
                    # memory writes.
                    self.evict(session_id, unload_session_caches=False)
            handle = await self._provision_with_span(session_id, pool=pool)
            self._handles[session_id] = handle
            self._last_used[session_id] = time.monotonic()
            return handle

    async def _spec_version_changed(
        self, session_id: str, handle: SandboxHandle, pool: asyncpg.Pool[Any]
    ) -> bool:
        """Return True iff ``sessions.spec_version`` drifted past the handle's
        snapshot (issue #713).

        Best-effort: any exception (transient DB error, session vanished)
        returns ``False`` so a healthy sandbox isn't recycled over a blip.
        One unscoped round-trip — this runs on every warm hit (every tool
        call reusing a live sandbox), so it must stay a single query.
        """
        try:
            async with pool.acquire() as conn:
                current = await queries.unscoped_get_session_spec_version(conn, session_id)
        except Exception as err:
            log.warning(
                "sandbox.spec_version_probe_failed",
                session_id=session_id,
                error=str(err),
            )
            return False
        return current != handle.spec_version

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
            self._release_tool_broker_secret(session_id)
            raise

        # Setup steps after create. If any of these raise, tear the
        # sandbox down so we don't leak an empty container alongside
        # the proxy.
        try:
            await ensure_workspace_runtime_dirs(self._backend, handle)
            await install_egress_ca(self._backend, handle)
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
            (WORKER_NETWORK_ALIAS, runtime.require_tool_broker().port),
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

    def _release_tool_broker_secret(self, session_id: str) -> None:
        """Drop the per-session entry from the tool broker's secret map.

        Idempotent: a missing entry is silently ignored. Called at every
        sandbox teardown site so the broker doesn't accumulate dangling
        secrets for sessions whose sandboxes are gone.

        Also removes the per-session secret file written by the spec
        builder when UDS transport is in use (issue #698). Reads the
        socket path from settings; if no UDS is configured the cleanup
        is a no-op.
        """
        from aios.config import get_settings
        from aios.harness import runtime

        # Skip broker-side unregistration when worker_main never ran (e.g.
        # e2e tests that wire the registry directly). The cleanup of the
        # on-disk .secret file is independent and still runs. Going through
        # ``require_tool_broker()`` (rather than reading the attribute
        # directly) keeps the function-level indirection that unit tests
        # patch — ``runtime.tool_broker`` itself is a module global that
        # can't be cleanly mocked per-test.
        try:
            broker = runtime.require_tool_broker()
        except RuntimeError:
            broker = None
        if broker is not None:
            broker.unregister_session(session_id)
        settings = get_settings()
        cleanup_session_secret_file(session_id, settings.tool_broker_socket_path)

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
        self._release_tool_broker_secret(session_id)

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
        :meth:`_release_tool_broker_secret`-adjacent paths — the runtime
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
        # NOTE: do NOT pop self._locks[session_id] here.  The two
        # release()-callers (``release_if_mounts_changed`` and the
        # idle reaper) wrap this call in ``async with
        # self._lock_for(session_id)``; popping the entry mid-release
        # would leave the caller holding a lock that's no longer
        # findable in the dict, so a concurrent ``get_or_provision``
        # would call ``_lock_for()``, see no entry, create a new
        # lock, and race with the in-progress release.  The race
        # leaks: ``_release_tool_broker_secret`` below would
        # unregister the new provision's broker secret, wedging the
        # new sandbox.  The accumulation of one ``asyncio.Lock`` per
        # ever-touched session is a bounded leak that the worker's
        # eventual restart clears; the race is unacceptable.
        # ``release_all()`` clears the whole dict at teardown.
        proxy = self._git_proxies.pop(session_id, None)

        if proxy is not None:
            await self._stop_proxy_silently(proxy, session_id)
        self._release_tool_broker_secret(session_id)
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

    def evict(self, session_id: str, *, unload_session_caches: bool = True) -> None:
        """Drop the cache entry without backend teardown (sandbox is dead).

        Used by tool_dispatch when it detects a sandbox-side failure that
        suggests the sandbox itself is unhealthy; the next tool call will
        cold-start a fresh one.

        The per-session :class:`GitProxy` is stopped in the background:
        the caller is on a retry path and can't block on stop()'s up-to-5s
        graceful drain. Skipping stop() entirely would strand the proxy's
        uvicorn server, httpx client and bound TCP port — ``release_all``
        only sees ``_git_proxies`` (which we've popped here), so without
        the background stop the proxy leaks for the worker's lifetime.

        ``unload_session_caches`` (default True) clears the per-session
        runtime caches (``_session_memory_mounts``, ``_session_read_shas``).
        That is correct when the session is being unloaded or its sandbox
        recycled *between* steps (the next step repopulates). The stale-
        handle recycle inside :meth:`get_or_provision` passes False: it
        happens *mid-step* and immediately re-provisions the same session,
        so the caches are still valid and — for ``_session_memory_mounts``
        — load-bearing for the bash memory-reconcile that runs right after
        provisioning. Clearing them there would silently drop the step's
        memory-store writes (the before/after diff would see no mounts).
        """
        from aios.harness import runtime

        self._last_used.pop(session_id, None)
        # NOTE: do NOT pop self._locks[session_id] here either — same
        # reason as ``release()``.  A concurrent ``get_or_provision``
        # that is already inside ``async with self._lock_for(sid)``
        # (awaiting ``_provision``) would have its lock disappear
        # from the dict; a subsequent third task arriving via
        # ``_lock_for(sid)`` would see no entry, create a new lock,
        # and race with the in-flight provision.  ``_release_tool_broker_secret``
        # below would then unregister that in-flight provision's
        # broker secret, wedging the new sandbox.
        if self._handles.pop(session_id, None) is not None:
            log.info("sandbox.evicted", session_id=session_id)
        # Drop the proxy too — a fresh sandbox will get a fresh proxy
        # from the next provision pass.
        proxy = self._git_proxies.pop(session_id, None)
        if proxy is not None:
            task = asyncio.create_task(
                self._stop_proxy_silently(proxy, session_id),
                name=f"sandbox-evict-proxy-stop:{session_id}",
            )
            self._evict_proxy_stop_tasks.add(task)
            task.add_done_callback(self._evict_proxy_stop_tasks.discard)
        # Same story for the tool broker secret: drop it immediately; a
        # fresh sandbox will get a fresh secret from the next provision.
        self._release_tool_broker_secret(session_id)
        # Drop the runtime caches so a fresh sandbox doesn't serve a stale
        # read-sha precondition match against a different container's file
        # state. Skipped on the mid-step recycle path (see the
        # ``unload_session_caches`` note in the docstring) where the same
        # session keeps running and the caches remain valid.
        if unload_session_caches:
            runtime.clear_session_memory_mounts(session_id)
            runtime.clear_session_read_shas(session_id)

    async def release_all(self) -> None:
        """Tear down every sandbox + proxy. Called at worker shutdown."""
        handles = list(self._handles.values())
        proxies = list(self._git_proxies.values())
        # Drop the per-session tool broker secret (and its on-disk
        # ``.secret`` file when UDS transport is in use) for every active
        # session, matching the cleanup path in ``release()``/``evict()``.
        # Without this, ``.secret`` files leak on every clean worker
        # shutdown.
        for h in handles:
            self._release_tool_broker_secret(h.session_id)
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

    async def reap_orphans(self) -> int:
        """At startup, remove every sandbox we manage.

        Called once per worker boot, before any step or tool task runs;
        the worker's ``task_registry`` is empty by construction, so every
        managed container is a corpse from a prior run. Sessions that
        resume on this worker will spawn fresh containers on their next
        step.
        """
        from aios.config import get_settings

        instance_id = get_settings().instance_id
        try:
            managed = await self._backend.list_managed(instance_id=instance_id)
        except Exception as err:
            log.warning("sandbox.reap_list_failed", error=str(err))
            return 0
        if not managed:
            return 0

        removed = 0
        for ref in managed:
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

    async def _reap_idle_once(self, idle_timeout: float) -> None:
        """One reap pass: release every session idle past ``idle_timeout``.

        Extracted from :meth:`_reap_idle_loop` for deterministic testing.
        Driven once per tick by the loop; never called directly in
        production.
        """
        now = time.monotonic()
        to_release: list[str] = []
        for sid, last in list(self._last_used.items()):
            if now - last > idle_timeout:
                to_release.append(sid)
        for sid in to_release:
            async with self._lock_for(sid):
                # Re-check under the lock: a concurrent get_or_provision
                # warm hit could have freshened _last_used[sid] between
                # the scan above and now. The warm path takes no lock (it
                # only does a liveness probe, not the per-session lock —
                # see get_or_provision), so this re-check is the only place
                # we serialize against it. #566 added the lock, which closes
                # the cold-path race; the missing piece was that the reaper
                # trusted its stale to_release snapshot. ``evict`` can also
                # pop _last_used out from under us, so absence means skip.
                current = self._last_used.get(sid)
                if current is not None and time.monotonic() - current > idle_timeout:
                    log.info("sandbox.idle_release", session_id=sid)
                    await self.release(sid)

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
                await self._reap_idle_once(idle_timeout)
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
