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
``container_idle_timeout_seconds`` (which now snapshots the rootfs before
removing — durable session sandboxes). Worker shutdown calls
:meth:`stop_all`, which STOPS every container (leaving its filesystem in a
stopped corpse for the next worker's GC tick to salvage) rather than
destroying them.

The registry owns the per-session :class:`GitProxy`, not the handle —
the proxy is a host-side process whose lifetime tracks the session, not
the sandbox container. Releasing a session stops its proxy (if any) in
addition to destroying the sandbox.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from aios.config import get_settings
from aios.db import queries
from aios.logging import get_logger
from aios.sandbox.backends.base import (
    BASE_IMAGE_LABEL_KEY,
    FLATTENED_LABEL_KEY,
    FLATTENED_LABEL_VALUE,
    SESSION_LABEL_KEY,
    CommandResult,
    ManagedImage,
    SandboxBackend,
    SandboxBackendError,
    SandboxHandle,
    SandboxSpec,
)
from aios.sandbox.git_proxy import GitProxy
from aios.sandbox.network import WORKER_NETWORK_ALIAS
from aios.sandbox.setup import (
    apply_network_lockdown,
    ensure_workspace_runtime_dirs,
    install_egress_ca,
    install_packages,
)
from aios.sandbox.snapshot_store import LocalDaemonStore, SnapshotStore
from aios.sandbox.spec import (
    ProvisioningPlan,
    build_spec_from_session,
    cleanup_session_secret_file,
    mount_snapshot_from_echoes,
    snapshot_tag,
)

if TYPE_CHECKING:
    import asyncpg

    from aios.db.queries import EnvVarCredentialEcho
    from aios.models.github_repositories import GithubRepositoryResourceEcho
    from aios.models.memory_stores import MemoryStoreResourceEcho

    # Type-only: a runtime top-level import would pull in ``aios.tools`` (via
    # ``secret_egress_proxy`` → ``url_safety``), whose package ``__init__``
    # imports ``bash`` → back into ``aios.sandbox.spec`` — a cycle. The
    # registry only ever annotates with this type (it constructs the proxy in
    # ``spec.build_spec_from_session`` and receives it on the plan), so a
    # TYPE_CHECKING import is sufficient.
    from aios.sandbox.secret_egress_proxy import SecretEgressProxy

log = get_logger("aios.sandbox.registry")

# Model-visible FS-loss lifecycle events (durable session sandboxes, §5.9).
# Appended append-only and NON-stimulus-bearing (a GC append never wakes a
# session); rendered by ``harness.context`` as bracketed user-role notices.
SANDBOX_FS_RESET_EVENT = "sandbox_fs_reset"
SANDBOX_FS_EXPIRED_EVENT = "sandbox_fs_expired"
SANDBOX_FS_OVER_LIMIT_EVENT = "sandbox_fs_over_limit"

# Overall deadline for the worker-shutdown ``stop_all`` (durable session
# sandboxes, §5.4): a hung daemon must not eat the SIGTERM grace.
_STOP_ALL_TIMEOUT_S = 8.0

# GC reconciler tick interval (durable session sandboxes, §5.5): hourly, with
# an immediate first tick at boot (replacing the old boot-time orphan reap).
_GC_INTERVAL_SECONDS = 3600.0


@dataclass(frozen=True, slots=True)
class SessionSnapshotState:
    """The GC's per-session decision inputs (one existing session row).

    A session *absent* from the GC's state map is **deleted** — its snapshot
    is collectible without an event (the model is gone). ``last_event_at`` is
    the dormancy probe (the ``created_at`` of the event at ``last_event_seq``);
    ``None`` only on the should-not-happen no-events edge, treated as *not*
    dormant (conservative — never wipe on a missing probe).
    """

    session_id: str
    account_id: str
    archived: bool
    last_event_at: datetime | None
    snapshot_ref: str | None
    snapshot_host: str | None
    snapshot_bytes: int | None


_GcVerdict = Literal["retain", "remove"]
_GcReason = Literal["live", "retention_ttl", "deleted", "residue"]


@dataclass(frozen=True, slots=True)
class GcImageVerdict:
    """A classified managed image and what the GC should do with it."""

    image: ManagedImage
    session_id: str | None
    is_canonical: bool
    removal_ref: str  # the tag for a canonical image (cascade-deletes the chain), else the image id
    verdict: _GcVerdict
    reason: _GcReason


def _is_session_dormant(state: SessionSnapshotState, now: datetime, ttl_seconds: int) -> bool:
    """A session is dormant iff its last activity is older than the TTL.

    Archived sessions follow the same rule (unarchive exists; immediate
    deletion would strand it). A missing dormancy probe reads as *not* dormant.
    """
    if state.last_event_at is None:
        return False
    return (now - state.last_event_at).total_seconds() > ttl_seconds


def _classify_images(
    images: list[ManagedImage],
    states: dict[str, SessionSnapshotState],
    *,
    now: datetime,
    ttl_seconds: int,
    this_host: str,
) -> list[GcImageVerdict]:
    """Pure retain-rule classifier for the GC image pass (§5.5), table-driven.

    The single rule: **an image is retained iff it is the canonical tag of an
    existing session whose last activity is within the TTL.** Everything else
    managed-and-mine is removed — crash residue, flatten leftovers, deleted
    sessions (the delete hook), and dormant sessions (the latter flagged
    ``retention_ttl`` so the caller emits ``sandbox_fs_expired``).

    Untagged interiors of *live* chains are skipped **structurally** — any
    image that is the ``.Parent`` of another listed image is excluded (the
    leaf's removal cascade-deletes the chain; removing an interior directly
    would be refused anyway). On the containerd store ``.Parent`` may be empty;
    the structural skip then no-ops and the ``remove_image`` refusal is the
    safety net.
    """
    parent_ids = {img.parent_id for img in images if img.parent_id}
    verdicts: list[GcImageVerdict] = []
    for img in images:
        # Structural skip: interior of a (possibly live) chain.
        if img.image_id in parent_ids:
            continue
        sid = img.labels.get(SESSION_LABEL_KEY)
        canonical_tag = snapshot_tag(this_host, sid) if sid else None
        is_canonical = canonical_tag is not None and canonical_tag in img.repo_tags
        removal_ref = canonical_tag if (is_canonical and canonical_tag) else img.image_id
        state = states.get(sid) if sid else None

        if is_canonical and state is not None and not _is_session_dormant(state, now, ttl_seconds):
            verdict: _GcVerdict = "retain"
            reason: _GcReason = "live"
        elif is_canonical and state is not None:
            verdict, reason = "remove", "retention_ttl"  # dormant → emit expired event
        elif is_canonical and state is None:
            verdict, reason = "remove", "deleted"  # session deleted → no event
        else:
            verdict, reason = "remove", "residue"  # crash/flatten leftover, non-canonical
        verdicts.append(
            GcImageVerdict(
                image=img,
                session_id=sid,
                is_canonical=is_canonical,
                removal_ref=removal_ref,
                verdict=verdict,
                reason=reason,
            )
        )
    return verdicts


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
        # Snapshot transport seam (durable session sandboxes). v1 is the
        # identity store over the local daemon; multi-host is a drop-in
        # replacement here with no lifecycle changes.
        self._store: SnapshotStore = LocalDaemonStore(backend)
        self._handles: dict[str, SandboxHandle] = {}
        self._git_proxies: dict[str, GitProxy] = {}
        self._secret_proxies: dict[str, SecretEgressProxy] = {}
        self._last_used: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._reaper_task: asyncio.Task[None] | None = None
        self._gc_task: asyncio.Task[None] | None = None
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
        triggers a recycle: the alive container is SNAPSHOT then removed
        via ``_snapshot_and_remove`` before re-provisioning (durable session
        sandboxes — the FS survives the recycle so the immediately-following
        provision resolves the just-written tag, an FS-preserving reboot onto
        the current mounts).

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
                    # Container is still alive — SNAPSHOT then remove (durable
                    # session sandboxes, §5.4): the FS must survive the recycle
                    # so the immediately-following provision resolves the
                    # just-written snapshot tag — an FS-preserving reboot onto
                    # the current mounts. ``evict`` drops the cache/proxy/secret
                    # (the snapshot reads the container's labels, not the cache);
                    # ``_snapshot_and_remove`` stops + commits + writes the
                    # pointer + removes. On snapshot failure the corpse is
                    # retained and the following provision's salvage preamble
                    # recovers it.
                    self.evict(session_id, unload_session_caches=False)
                    await self._snapshot_and_remove(session_id, current)
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
        """Salvage corpses, resolve the snapshot, create the sandbox, run setup.

        Runs under the per-session lock (held by :meth:`get_or_provision`), so
        the salvage preamble's stop/commit/remove is serialized against
        release and the idle reaper.
        """
        # Provision preamble (durable session sandboxes, §5.4): salvage any
        # crash corpse for this session BEFORE provisioning — container death
        # no longer loses data. Salvage failure fails the provision loud (raw
        # error into the tool result, model-actionable); never a silent resume
        # from a stale tag.
        await self._salvage_session_corpses(session_id)
        # First-commit crash heal (§5.3): a commit that succeeded but whose
        # pointer write crashed leaves a canonical tag with a NULL pointer.
        # Reconcile it from local truth so resolution below can't miss a local
        # snapshot (the GC tick covers the same case out-of-band).
        await self._reconcile_pointer_from_local(session_id)

        plan = await build_spec_from_session(session_id)
        # Record the proxies before anything else fallible runs: once
        # build_spec_from_session returns, its proxies are RUNNING and the
        # broker secret is REGISTERED, so a raise from _resolve_snapshot (below)
        # would otherwise escape with the proxies absent from these dicts —
        # release/evict/stop_all could never find them. Storing here keeps the
        # whole span through backend.create leak-safe via the cleanup below.
        if plan.git_proxy is not None:
            self._git_proxies[session_id] = plan.git_proxy
        if plan.secret_proxy is not None:
            self._secret_proxies[session_id] = plan.secret_proxy

        try:
            # Resolve the snapshot pointer through the store: verified-negative
            # existence, base-image drift, missing-snapshot detection. Returns
            # the spec to run from — snapshot tag resolved to a local image, or
            # cleared to None (cold start) on a detected reset.
            spec = await self._resolve_snapshot(session_id, plan.spec)
            handle = await self._backend.create(spec)
        except BaseException:
            # Spec was built (proxies are running, broker secret is registered)
            # but resolution or backend.create failed before the sandbox
            # exists. Drop all so their ports + token/secret maps don't leak.
            # No backend.destroy here — there is no sandbox yet.
            self._git_proxies.pop(session_id, None)
            if plan.git_proxy is not None:
                await self._stop_proxy_silently(plan.git_proxy, session_id, kind="git_proxy")
            self._secret_proxies.pop(session_id, None)
            if plan.secret_proxy is not None:
                await self._stop_proxy_silently(plan.secret_proxy, session_id, kind="secret_proxy")
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
            # Whether this was a snapshot resume or a cold start — so
            # "why did this session cold-start" is answerable after the fact
            # (consecutive provisions flipping snapshot→base expose even
            # operator-caused wipes). §5.9.
            resumed_from_snapshot=handle.snapshot_image is not None,
        )
        return handle

    async def _maybe_apply_lockdown(self, handle: SandboxHandle, plan: ProvisioningPlan) -> None:
        """Apply network lockdown if the plan calls for it."""
        from aios.harness import runtime
        from aios.models.environments import LimitedNetworking
        from aios.models.vaults import parse_allowed_host_entry

        networking = plan.env_config.networking if plan.env_config else None
        if not isinstance(networking, LimitedNetworking):
            return
        extra_host_ports: list[tuple[str, int]] = [
            (WORKER_NETWORK_ALIAS, runtime.require_tool_broker().port),
        ]
        if plan.git_proxy is not None:
            extra_host_ports.append((WORKER_NETWORK_ALIAS, plan.git_proxy.port))
        dnat_hosts: list[str] = []
        dnat_target: tuple[str, int] | None = None
        if plan.secret_proxy is not None:
            # Open the filter OUTPUT for the rewritten (post-DNAT) flow to the
            # proxy endpoint — mirrors the git_proxy precedent above. (#878)
            extra_host_ports.append((WORKER_NETWORK_ALIAS, plan.secret_proxy.port))
            dnat_target = (WORKER_NETWORK_ALIAS, plan.secret_proxy.port)
            # Each credential's allowed_hosts holds canonical entries (host or
            # host/path-prefix); DNAT keys on the bare host only.
            seen: set[str] = set()
            for cred in plan.env_var_credentials:
                for entry in cred.allowed_hosts:
                    host, _prefix = parse_allowed_host_entry(entry)
                    if host not in seen:
                        seen.add(host)
                        dnat_hosts.append(host)
        await apply_network_lockdown(
            self._backend,
            handle,
            networking,
            extra_host_ports=extra_host_ports,
            dnat_hosts=dnat_hosts,
            dnat_target=dnat_target,
        )

    async def _stop_proxy_silently(
        self, proxy: GitProxy | SecretEgressProxy, session_id: str, *, kind: str
    ) -> None:
        """Stop ``proxy``, log + swallow any error.

        Used by every cleanup path; a stuck proxy must never block
        sandbox teardown or propagate a secondary exception over a
        primary one. ``kind`` (``"git_proxy"`` / ``"secret_proxy"``)
        names the proxy type in the failure log so a secret-egress stop
        failure isn't mis-attributed to the git proxy in logs/alerts.
        """
        try:
            await proxy.stop()
        except Exception as err:
            log.warning(
                "sandbox.proxy_stop_failed",
                kind=kind,
                session_id=session_id,
                error=str(err),
            )

    def _spawn_evict_proxy_stop(
        self, proxy: GitProxy | SecretEgressProxy, session_id: str, *, kind: str
    ) -> None:
        """Fire-and-forget stop of an evicted proxy, holding a strong ref.

        ``evict`` is on a retry path and can't block on ``stop()``'s up-to-5s
        graceful drain, so the stop runs as a detached task. asyncio only
        weak-refs tasks, so the task is parked in ``_evict_proxy_stop_tasks``
        (cleared via a done-callback) to keep it alive until ``stop()`` unwinds
        the proxy's uvicorn server, httpx client and bound TCP port.
        """
        task = asyncio.create_task(
            self._stop_proxy_silently(proxy, session_id, kind=kind),
            name=f"sandbox-evict-{kind}-stop:{session_id}",
        )
        self._evict_proxy_stop_tasks.add(task)
        task.add_done_callback(self._evict_proxy_stop_tasks.discard)

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
            await self._stop_proxy_silently(proxy, session_id, kind="git_proxy")
        secret_proxy = self._secret_proxies.pop(session_id, None)
        if secret_proxy is not None:
            await self._stop_proxy_silently(secret_proxy, session_id, kind="secret_proxy")
        self._release_tool_broker_secret(session_id)

    # ── durable-session-sandbox lifecycle helpers (§5.2-§5.4) ───────────────

    async def _snapshot_and_remove(self, session_id: str, handle: SandboxHandle) -> None:
        """Snapshot ``handle``'s rootfs → write the pointer → remove the container.

        **Ordering invariant**: the container is removed only after the
        snapshot verb AND the pointer write both succeed. On failure the
        stopped corpse is retained (no ``rm``) — the next provision's salvage
        preamble or the GC tick converges. Best-effort by contract: a snapshot
        failure here must not propagate (release/recycle callers continue).
        """
        if await self._snapshot_and_record(
            session_id, handle.sandbox_id, disk_limit_bytes=handle.disk_limit_bytes
        ):
            await self._backend.destroy(handle)

    async def _snapshot_and_record(
        self, session_id: str, sandbox_id: str, *, disk_limit_bytes: int | None
    ) -> bool:
        """Run the snapshot verb and write the DB pointer for ``sandbox_id``.

        Returns ``True`` iff the container may now be removed (snapshot verb
        and pointer write both succeeded). A failed pointer write is treated
        identically to a failed snapshot verb (§5.2): corpse retained, return
        ``False``, converge via salvage.
        """
        settings = get_settings()
        tag = snapshot_tag(settings.instance_id, session_id)
        try:
            outcome = await self._backend.snapshot(
                sandbox_id,
                tag,
                empty_floor_bytes=settings.sandbox_snapshot_empty_floor_bytes,
                flatten_if_unique_bytes_over=disk_limit_bytes,
            )
        except Exception as err:
            log.warning(
                "sandbox.snapshot_failed_corpse_retained",
                session_id=session_id,
                container_id=sandbox_id[:12],
                error=str(err),
            )
            return False

        # ``image_id is None`` ⇒ skipped_empty with no prior tag (a session
        # that never wrote): nothing to point at, leave the pointer NULL.
        if outcome.image_id is not None:
            # Edge-trigger the over-limit notice only on the crossing — read
            # the prior bytes only when we're actually over budget (rare).
            prev_bytes: int | None = None
            over_now = disk_limit_bytes is not None and outcome.unique_bytes > disk_limit_bytes
            if over_now:
                prev_bytes = await self._read_snapshot_bytes(session_id)
            try:
                await self._write_snapshot_pointer(session_id, tag, outcome.unique_bytes)
            except Exception as err:
                log.warning(
                    "sandbox.snapshot_pointer_write_failed_corpse_retained",
                    session_id=session_id,
                    container_id=sandbox_id[:12],
                    error=str(err),
                )
                return False
            if (
                disk_limit_bytes is not None
                and over_now
                and (prev_bytes is None or prev_bytes <= disk_limit_bytes)
            ):
                await self._append_fs_event(
                    session_id,
                    SANDBOX_FS_OVER_LIMIT_EVENT,
                    {"unique_bytes": outcome.unique_bytes, "limit_bytes": disk_limit_bytes},
                )

        log.info(
            "sandbox.snapshot",
            session_id=session_id,
            container_id=sandbox_id[:12],
            kind=outcome.kind,
            unique_bytes=outcome.unique_bytes,
            depth=outcome.depth,
        )
        return True

    async def _salvage_session_corpses(self, session_id: str) -> None:
        """Provision preamble: snapshot + remove every corpse of ``session_id``.

        Container death no longer loses data — a crash/OOM/daemon-restart
        corpse is committed here before the session's next container starts.
        **Salvage failure fails the provision** (raw error, model-actionable):
        a retained corpse must never be silently bypassed into a stale-tag
        resume. Per-corpse budget falls back to the global default (a crash
        corpse has no spec/handle to carry a per-env override).
        """
        settings = get_settings()
        refs = await self._backend.list_managed(
            instance_id=settings.instance_id, session_id=session_id
        )
        for ref in refs:
            removable = await self._snapshot_and_record(
                session_id,
                ref.sandbox_id,
                disk_limit_bytes=settings.sandbox_snapshot_budget_bytes,
            )
            if not removable:
                raise SandboxBackendError(
                    f"salvage of corpse {ref.sandbox_id[:12]} for session {session_id} "
                    "failed (snapshot or pointer write); refusing to provision over "
                    "unrecovered state"
                )
            await self._backend.force_remove(ref.sandbox_id)

    async def _reconcile_pointer_from_local(self, session_id: str) -> None:
        """First-commit crash heal (§5.3): if the canonical tag exists locally,
        ensure the pointer points at it.

        Closes the window where a commit succeeded but its pointer write
        crashed (and the container was already removed), leaving a tag with a
        NULL pointer that resolution would otherwise miss. Idempotent — a
        no-op when the pointer already matches. ``snapshot_bytes`` is set to
        the full image size here (reporting-only; the next real commit writes
        the accurate unique figure).
        """
        tag = snapshot_tag(get_settings().instance_id, session_id)
        try:
            if not await self._store.exists(tag):
                return
            size = await self._store.size(tag)
        except SandboxBackendError as err:
            log.warning(
                "sandbox.pointer_reconcile_probe_failed", session_id=session_id, error=str(err)
            )
            return
        await self._write_snapshot_pointer(session_id, tag, size)

    async def _resolve_snapshot(self, session_id: str, spec: SandboxSpec) -> SandboxSpec:
        """Resolve the DB snapshot pointer to a runnable spec (§5.3).

        ``spec.snapshot_image`` arrives as the raw pointer ref. Returns a spec
        whose ``snapshot_image`` is the locally-runnable tag for a valid
        resume, or ``None`` (cold start) on a detected reset. Verified-negative
        throughout: an indeterminate store probe raises and fails the provision
        rather than silently cold-starting (which the next idle's lineage gate
        would then punish as ``skipped_stale``).
        """
        ref = spec.snapshot_image
        if ref is None:
            return spec  # cold start — no pointer

        # Verified-negative existence through the store (raises on indeterminate).
        if not await self._store.exists(ref):
            # Pointer set + store verified-not-found ⇒ external mutation
            # (operator rmi, image-store loss, host replacement w/o transport).
            await self._reset_snapshot(session_id, reason="snapshot_missing")
            return dataclasses.replace(spec, snapshot_image=None)

        # Base-image drift: the snapshot's recorded base vs the currently
        # resolved env image. A mismatch means the operator deliberately
        # redefined the environment image.
        snap_labels = await self._backend.image_labels(ref)
        if snap_labels is None:
            # The image was removed between the existence probe and here (an
            # operator rmi racing resume). That's the snapshot-missing case, not
            # base drift — record the right reason and cold-start; nothing to
            # remove (it's already gone).
            await self._reset_snapshot(session_id, reason="snapshot_missing")
            return dataclasses.replace(spec, snapshot_image=None)
        snap_base = snap_labels.get(BASE_IMAGE_LABEL_KEY)
        if snap_base != spec.image:
            # Discard — the artifact must actually be GONE: a surviving tag
            # would be re-pointered by GC pass 4, and the next idle's lineage
            # gate would see a corpse rooted on the new base against the old
            # tag and discard live post-drift work as skipped_stale. remove +
            # clear + event, in the same step.
            await self._store.remove(ref)
            await self._reset_snapshot(session_id, reason="environment_image_changed")
            return dataclasses.replace(spec, snapshot_image=None)

        # Valid resume: make the ref locally runnable (identity for v1).
        local_tag = await self._store.get(ref)
        return dataclasses.replace(spec, snapshot_image=local_tag)

    async def _reset_snapshot(self, session_id: str, *, reason: str) -> None:
        """Clear the snapshot pointer and append a model-visible reset notice."""
        from aios.harness import runtime

        pool = runtime.require_pool()
        async with pool.acquire() as conn:
            await queries.unscoped_clear_session_snapshot(conn, session_id)
        await self._append_fs_event(session_id, SANDBOX_FS_RESET_EVENT, {"reason": reason})
        log.info("sandbox.fs_reset", session_id=session_id, reason=reason)

    async def _write_snapshot_pointer(self, session_id: str, ref: str, unique_bytes: int) -> None:
        """Write the DB snapshot pointer under the deployment's host id.

        ``snapshot_host`` is ``settings.instance_id`` in v1 (one worker), kept
        distinct from the deployment namespace that derives ``ref`` so a future
        multi-host deployment never changes a session's ref on handoff (§5.11).
        """
        from aios.harness import runtime

        pool = runtime.require_pool()
        async with pool.acquire() as conn:
            await queries.unscoped_set_session_snapshot(
                conn,
                session_id,
                ref=ref,
                host=get_settings().instance_id,
                snapshot_bytes=unique_bytes,
            )

    async def _read_snapshot_bytes(self, session_id: str) -> int | None:
        from aios.harness import runtime

        pool = runtime.require_pool()
        async with pool.acquire() as conn:
            return await queries.unscoped_get_session_snapshot_bytes(conn, session_id)

    async def _append_fs_event(self, session_id: str, event: str, payload: dict[str, Any]) -> None:
        """Append a model-visible FS-loss lifecycle event (§5.9).

        Append-only and **not** stimulus-bearing — ``append_event`` only
        advances ``last_stimulus_seq`` for user/tool roles, and a lifecycle
        event carries no role, so a GC/reset append never wakes the session or
        costs a model call; the notice is read at the next genuine wake.
        """
        from aios.harness import runtime
        from aios.services import sessions as sessions_service

        pool = runtime.require_pool()
        account_id = await sessions_service.load_session_account_id(pool, session_id)
        await sessions_service.append_event(
            pool,
            session_id,
            "lifecycle",
            {"event": event, **payload},
            account_id=account_id,
        )

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
        # ``stop_all()`` clears the whole dict at teardown.
        proxy = self._git_proxies.pop(session_id, None)
        secret_proxy = self._secret_proxies.pop(session_id, None)

        if proxy is not None:
            await self._stop_proxy_silently(proxy, session_id, kind="git_proxy")
        if secret_proxy is not None:
            await self._stop_proxy_silently(secret_proxy, session_id, kind="secret_proxy")
        self._release_tool_broker_secret(session_id)
        runtime.clear_session_memory_mounts(session_id)
        runtime.clear_session_read_shas(session_id)
        if handle is None:
            return
        # Durable session sandboxes: snapshot the rootfs → write the DB
        # pointer → remove (instead of a bare destroy). On snapshot/pointer
        # failure the corpse is retained and converges via the next
        # provision's salvage preamble or the GC tick. Host-side cleanup above
        # already ran, so a retained corpse never leaves a stale proxy/secret.
        await self._snapshot_and_remove(session_id, handle)

    async def release_if_mounts_changed(
        self,
        session_id: str,
        memory_echoes: list[MemoryStoreResourceEcho],
        github_echoes: list[GithubRepositoryResourceEcho],
        env_var_credential_echoes: list[EnvVarCredentialEcho],
    ) -> None:
        """Release the cached sandbox if its mount snapshot has drifted.

        Acquires the per-session lock so a tool task from a prior step
        can't race with the release via ``get_or_provision``.

        Drift includes: memory store attachments added/removed, github
        repos added/removed, github token rotated, and env-var credentials
        added/removed/rotated (the ``updated_at`` timestamp on the github
        echo and the env-var echo are part of the snapshot key, see
        :func:`mount_snapshot_from_echoes`). The secret-egress proxy is
        stopped via the delegated ``release()`` call.
        """
        async with self._lock_for(session_id):
            handle = self._handles.get(session_id)
            if handle is None:
                return
            if handle.mount_snapshot == mount_snapshot_from_echoes(
                memory_echoes, github_echoes, env_var_credential_echoes
            ):
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
        uvicorn server, httpx client and bound TCP port — ``stop_all``
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
        # Drop the proxies too — a fresh sandbox will get fresh proxies
        # from the next provision pass. Both stops are fire-and-forget
        # (the caller is on a retry path) and share the strong-ref set so
        # neither task is GC'd before stop() unwinds its server/port.
        git_proxy = self._git_proxies.pop(session_id, None)
        if git_proxy is not None:
            self._spawn_evict_proxy_stop(git_proxy, session_id, kind="git_proxy")
        secret_proxy = self._secret_proxies.pop(session_id, None)
        if secret_proxy is not None:
            self._spawn_evict_proxy_stop(secret_proxy, session_id, kind="secret_proxy")
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

    async def stop_all(self) -> None:
        """Worker shutdown: bounded parallel STOP (no commits) + host cleanup.

        Under durable persistence, shutdown does NOT commit — a commit per
        container is unbounded and could eat the SIGTERM grace. It stops the
        containers under one overall ``wait_for`` so a hung daemon can't wedge
        teardown, leaving stopped corpses (``--restart no``) that the next
        worker's immediate first GC tick salvages. Per-container stop failures
        are ops-log noise; the boot tick converges regardless.

        Replaces the old destroy-everything ``release_all`` — destroying at
        shutdown would lose every active session's filesystem.
        """
        handles = list(self._handles.values())
        proxies = list(self._git_proxies.values())
        secret_proxies = list(self._secret_proxies.values())
        # Drop the per-session tool broker secret (and its on-disk ``.secret``
        # file when UDS transport is in use) for every active session, matching
        # the cleanup path in ``release()``/``evict()``.
        for h in handles:
            self._release_tool_broker_secret(h.session_id)
        self._handles.clear()
        self._last_used.clear()
        self._locks.clear()
        self._git_proxies.clear()
        self._secret_proxies.clear()

        kinded_proxies: list[tuple[str, GitProxy | SecretEgressProxy]] = [
            *(("git_proxy", p) for p in proxies),
            *(("secret_proxy", p) for p in secret_proxies),
        ]
        if kinded_proxies:
            proxy_results = await asyncio.gather(
                *(p.stop() for _kind, p in kinded_proxies), return_exceptions=True
            )
            for (kind, _p), result in zip(kinded_proxies, proxy_results, strict=True):
                if isinstance(result, BaseException):
                    log.warning("sandbox.stop_all_proxy_error", kind=kind, error=str(result))

        # Drain any evict-triggered fire-and-forget proxy-stop tasks still in
        # flight: evict() pops the proxy out of ``_git_proxies`` /
        # ``_secret_proxies`` (so the gather above can't see it) and stops it in
        # the background. Without awaiting them here, shutdown would abandon an
        # in-progress teardown, leaking the proxy's server/port past worker
        # exit. Snapshot first — each task removes itself from the set via its
        # done-callback. These tasks already log+swallow internally, so
        # ``return_exceptions=True`` is belt-and-suspenders.
        evict_tasks = list(self._evict_proxy_stop_tasks)
        if evict_tasks:
            await asyncio.gather(*evict_tasks, return_exceptions=True)

        if not handles:
            return
        log.info("sandbox.stop_all", count=len(handles))
        try:
            results = await asyncio.wait_for(
                asyncio.gather(
                    *(self._backend.stop(h.sandbox_id) for h in handles),
                    return_exceptions=True,
                ),
                timeout=_STOP_ALL_TIMEOUT_S,
            )
        except TimeoutError:
            log.warning("sandbox.stop_all_timeout", count=len(handles), timeout=_STOP_ALL_TIMEOUT_S)
            return
        for h, result in zip(handles, results, strict=True):
            if isinstance(result, BaseException):
                log.warning(
                    "sandbox.stop_all_error",
                    session_id=h.session_id,
                    container_id=h.sandbox_id[:12],
                    error=str(result),
                )

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

    # ── GC: one retain-rule reconciler (§5.5) ───────────────────────────────

    def start_gc(self, pool: asyncpg.Pool[Any]) -> None:
        """Start the snapshot GC reconciler (hourly, immediate first tick).

        Replaces the old boot-time ``reap_orphans``: instead of removing every
        managed container at boot, the first tick salvages crash corpses and
        reconciles images/pointers against store truth, then repeats hourly.
        Boot is not blocked — a session waking mid-reconcile salvages its own
        corpse inline under its own lock.
        """
        if self._gc_task is not None:
            return
        self._gc_task = asyncio.create_task(self._gc_loop(pool), name="sandbox-snapshot-gc")

    def stop_gc(self) -> None:
        """Cancel the GC reconciler."""
        if self._gc_task is not None:
            self._gc_task.cancel()
            self._gc_task = None

    async def _gc_loop(self, pool: asyncpg.Pool[Any]) -> None:
        """Background loop: immediate first tick, then hourly.

        The try/except is nested INSIDE the loop (mirroring the idle reaper):
        a Docker/DB hiccup in one tick must not silently disable the GC for the
        worker's lifetime. ``CancelledError`` is not an ``Exception``, so
        ``stop_gc()`` still exits cleanly.
        """
        first = True
        while True:
            try:
                if not first:
                    await asyncio.sleep(_GC_INTERVAL_SECONDS)
                first = False
                await self._gc_once(pool)
            except Exception:
                log.exception("sandbox.gc_tick_failed")

    async def _gc_once(self, pool: asyncpg.Pool[Any]) -> None:
        """One GC tick: corpse pass, image pass, pool-budget pass, pointer reconcile."""
        settings = get_settings()
        instance_id = settings.instance_id
        now = datetime.now(UTC)

        # Pass 1 — corpses (may commit + remove, so it runs before the image enum).
        containers = await self._backend.list_managed(instance_id=instance_id)
        corpse_states = await self._load_gc_states(
            pool, {c.session_id for c in containers if c.session_id}
        )
        await self._gc_corpse_pass(containers, corpse_states, now, settings, instance_id)

        # Pass 2 — images (enumerated AFTER the corpse pass settled).
        images = await self._backend.list_managed_images(instance_id=instance_id)
        image_states = await self._load_gc_states(
            pool,
            {sid for img in images if (sid := img.labels.get(SESSION_LABEL_KEY)) is not None},
        )
        verdicts = _classify_images(
            images,
            image_states,
            now=now,
            ttl_seconds=settings.sandbox_snapshot_ttl_seconds,
            this_host=instance_id,
        )
        retained = await self._gc_image_pass(
            verdicts, image_states, now, settings.sandbox_snapshot_ttl_seconds, instance_id
        )

        # Pass 3 — per-host pool budget.
        await self._gc_pool_budget_pass(
            retained, image_states, settings.sandbox_snapshot_pool_bytes, instance_id
        )

        # Pass 4 — pointer reconciliation against local store truth.
        await self._gc_reconcile_pointers(retained, image_states, instance_id)

        log.info(
            "sandbox.gc_tick",
            corpses=len(containers),
            images=len(images),
            retained=len(retained),
            removed=len(verdicts) - len(retained),
        )

    async def _load_gc_states(
        self, pool: asyncpg.Pool[Any], session_ids: set[str]
    ) -> dict[str, SessionSnapshotState]:
        """Batch-load per-session GC inputs; absent ⇒ deleted (collectible)."""
        if not session_ids:
            return {}
        async with pool.acquire() as conn:
            rows = await queries.gc_snapshot_session_states(conn, list(session_ids))
        return {
            row["id"]: SessionSnapshotState(
                session_id=row["id"],
                account_id=row["account_id"],
                archived=row["archived_at"] is not None,
                last_event_at=row["last_event_at"],
                snapshot_ref=row["snapshot_ref"],
                snapshot_host=row["snapshot_host"],
                snapshot_bytes=row["snapshot_bytes"],
            )
            for row in rows
        }

    async def _fresh_session_state(self, session_id: str) -> SessionSnapshotState | None:
        """Re-read one session's GC state — the **condition re-verify** the
        retain rule needs under the per-session lock (§5.5).

        The tick-start ``states`` snapshot can be stale for a session that woke
        (or crossed back under the TTL) between the load and a drop decision;
        re-deriving dormancy from a fresh single-row read under the lock is what
        keeps the GC from force-removing a just-woke session's corpse without
        salvage, or removing its canonical snapshot a tick too early. ``None``
        ⇒ the session is genuinely gone (deleted), still collectible.
        """
        from aios.harness import runtime

        states = await self._load_gc_states(runtime.require_pool(), {session_id})
        return states.get(session_id)

    async def _gc_corpse_pass(
        self,
        containers: list[Any],
        states: dict[str, SessionSnapshotState],
        now: datetime,
        settings: Any,
        instance_id: str,
    ) -> None:
        """Salvage (or drop) every managed container that isn't a live cached handle.

        Retain rule first: a deleted/dormant session's corpse is removed
        WITHOUT paying a commit; a live-within-TTL corpse is salvaged (commit)
        then removed. Best-effort — a snapshot failure leaves the corpse for
        the next tick (the GC never raises). Each corpse is handled under the
        per-session lock with the cached-handle re-check.
        """
        for ref in containers:
            sid = ref.session_id
            if sid is None:
                # Unlabeled container — can't salvage without a session; drop it.
                await self._backend.force_remove(ref.sandbox_id)
                continue
            async with self._lock_for(sid):
                cached = self._handles.get(sid)
                if cached is not None and cached.sandbox_id == ref.sandbox_id:
                    continue  # the live, in-use container — never touch it
                ttl = settings.sandbox_snapshot_ttl_seconds
                state = states.get(sid)
                keep_fs = state is not None and not _is_session_dormant(state, now, ttl)
                if not keep_fs:
                    # Drop candidate (deleted/dormant per the tick-start snapshot)
                    # — re-verify dormancy under the lock (§5.5). A session that
                    # woke since the load must be salvaged, not dropped without a
                    # commit. (Only the drop direction can be wrong: a session
                    # can't grow MORE dormant within one tick.)
                    fresh = await self._fresh_session_state(sid)
                    keep_fs = fresh is not None and not _is_session_dormant(fresh, now, ttl)
                if keep_fs:
                    removable = await self._snapshot_and_record(
                        sid, ref.sandbox_id, disk_limit_bytes=settings.sandbox_snapshot_budget_bytes
                    )
                    if removable:
                        await self._backend.force_remove(ref.sandbox_id)
                    # else: snapshot failed — leave the corpse for the next tick.
                else:
                    # Deleted/dormant: remove without paying a commit.
                    await self._backend.force_remove(ref.sandbox_id)

    async def _gc_image_pass(
        self,
        verdicts: list[GcImageVerdict],
        states: dict[str, SessionSnapshotState],
        now: datetime,
        ttl_seconds: int,
        instance_id: str,
    ) -> list[GcImageVerdict]:
        """Remove every non-retained image (under per-session locks); return the retained."""
        retained: list[GcImageVerdict] = []
        for v in verdicts:
            if v.verdict == "retain":
                retained.append(v)
                continue
            sid = v.session_id
            if sid is None:
                # Residue with no session label — remove directly.
                await self._backend.remove_image(v.removal_ref)
                continue
            async with self._lock_for(sid):
                # Re-check under the lock: a waking session may now hold a
                # cached handle (raced between corpse-salvage and docker run);
                # never rmi a just-salvaged snapshot out from under it.
                if self._handles.get(sid) is not None:
                    retained.append(v)
                    continue
                # Condition re-verify (§5.5): a TTL-expiry removal is decided
                # from the tick-start snapshot — re-read dormancy under the lock
                # so a session that woke since the load keeps its canonical
                # snapshot instead of being expired a tick early. (Deleted /
                # residue verdicts can't flip back within a tick.)
                if v.reason == "retention_ttl":
                    fresh = await self._fresh_session_state(sid)
                    if fresh is not None and not _is_session_dormant(fresh, now, ttl_seconds):
                        retained.append(v)
                        continue
                removed = await self._backend.remove_image(v.removal_ref)
                if not removed:
                    # Refused (a child still references it) — retain this tick.
                    retained.append(v)
                    continue
                if v.reason == "retention_ttl":
                    await self._append_fs_event(
                        sid, SANDBOX_FS_EXPIRED_EVENT, {"reason": "retention_ttl"}
                    )
                if v.is_canonical:
                    await self._clear_pointer_if_owned(sid, instance_id, states)
        return retained

    async def _gc_pool_budget_pass(
        self,
        retained: list[GcImageVerdict],
        states: dict[str, SessionSnapshotState],
        pool_bytes: int | None,
        instance_id: str,
    ) -> None:
        """Evict most-dormant sessions first while this host is over its pool budget."""
        if pool_bytes is None:
            return
        base_sizes: dict[str, int] = {}
        sized: list[tuple[GcImageVerdict, int]] = []
        total = 0
        for v in retained:
            if not v.is_canonical:
                continue
            ub = await self._unique_bytes_for_image(v.image, base_sizes)
            total += ub
            sized.append((v, ub))
        if total <= pool_bytes:
            return

        def _dormancy_key(item: tuple[GcImageVerdict, int]) -> datetime:
            st = item[0].session_id and states.get(item[0].session_id)
            if st and st.last_event_at is not None:
                return st.last_event_at
            return datetime.min.replace(tzinfo=UTC)  # unknown dormancy ⇒ evict first

        for v, ub in sorted(sized, key=_dormancy_key):
            if total <= pool_bytes:
                break
            sid = v.session_id
            if sid is None:
                continue
            async with self._lock_for(sid):
                if self._handles.get(sid) is not None:
                    continue  # waking — skip
                if await self._backend.remove_image(v.removal_ref):
                    total -= ub
                    await self._append_fs_event(
                        sid, SANDBOX_FS_EXPIRED_EVENT, {"reason": "disk_pressure"}
                    )
                    await self._clear_pointer_if_owned(sid, instance_id, states)

    async def _gc_reconcile_pointers(
        self,
        retained: list[GcImageVerdict],
        states: dict[str, SessionSnapshotState],
        instance_id: str,
    ) -> None:
        """Heal a NULL/stale pointer for a retained canonical tag (§5.5 pass 4).

        Ownership-gated: a tick touches ``sessions.snapshot_*`` only for
        sessions whose ``snapshot_host`` is this host (or NULL, for the crash
        heal). Multi-host compare-and-swap is deferred — the ``snapshot_host``
        column is the seam that makes it additive.
        """
        base_sizes: dict[str, int] = {}  # shared across the pass (sessions share a base)
        for v in retained:
            if not v.is_canonical:
                continue
            sid = v.session_id
            if sid is None:
                continue
            st = states.get(sid)
            if st is None:
                continue
            if st.snapshot_host not in (None, instance_id):
                continue  # owned by another host — never reach across
            if st.snapshot_ref == v.removal_ref:
                continue  # already correct
            async with self._lock_for(sid):
                if self._handles.get(sid) is not None:
                    continue  # active; its own commit owns the pointer
                ub = await self._unique_bytes_for_image(v.image, base_sizes)
                await self._write_snapshot_pointer(sid, v.removal_ref, ub)

    async def _unique_bytes_for_image(self, image: ManagedImage, base_sizes: dict[str, int]) -> int:
        """Unique bytes for accounting: full size for a flattened (standalone)
        image, else ``tag.Size - base.Size``. ``base_sizes`` caches base lookups."""
        if image.labels.get(FLATTENED_LABEL_KEY) == FLATTENED_LABEL_VALUE:
            return image.size_bytes
        base_ref = image.labels.get(BASE_IMAGE_LABEL_KEY)
        if not base_ref:
            return image.size_bytes
        if base_ref not in base_sizes:
            try:
                base_sizes[base_ref] = await self._backend.image_size(base_ref)
            except SandboxBackendError:
                base_sizes[base_ref] = 0  # over-count is safe; never under-report
        return max(0, image.size_bytes - base_sizes[base_ref])

    async def _clear_pointer_if_owned(
        self, session_id: str, instance_id: str, states: dict[str, SessionSnapshotState]
    ) -> None:
        """Clear a session's pointer when removing its canonical artifact.

        Ownership-gated: skip when the pointer is owned by another host (a
        local cache of a peer's artifact, never the canonical copy). A deleted
        session (absent from ``states``) is cleared unconditionally — the
        ``UPDATE`` is a harmless no-op against the vanished row.
        """
        st = states.get(session_id)
        if st is not None and st.snapshot_host not in (None, instance_id):
            return
        from aios.harness import runtime

        pool = runtime.require_pool()
        async with pool.acquire() as conn:
            await queries.unscoped_clear_session_snapshot(conn, session_id)
