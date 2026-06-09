"""Unit tests for ``SandboxRegistry`` worker-side cleanup paths.

``TestReleaseIfMountsChanged`` covers the mount-drift detector that
recycles a cached sandbox when the session's attached memory stores
have changed since provisioning (issue #198). ``TestEvictReclamation``
covers the ``evict`` fast path used by ``tool_dispatch`` when a
sandbox-side failure suggests the sandbox itself is unhealthy.
``TestStaleHandleDetection`` covers the liveness probe inserted into
``get_or_provision`` so a container that Docker auto-removed (``--rm``
after entrypoint exit / OOM / daemon-side cleanup) doesn't poison the
next exec with "No such container" (issue #691).
"""

from __future__ import annotations

import asyncio
import gc
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.harness import runtime
from aios.models.memory_stores import Access, MemoryStoreResourceEcho
from aios.sandbox.backends.base import (
    CommandResult,
    Mount,
    SandboxHandle,
    SandboxSpec,
    Unrestricted,
)
from aios.sandbox.registry import SandboxRegistry
from aios.sandbox.spec import ProvisioningPlan
from tests.helpers.sandbox import FakeBackend, make_handle


def _echo(
    memory_store_id: str, name: str, access: Access = "read_write"
) -> MemoryStoreResourceEcho:
    return MemoryStoreResourceEcho(
        memory_store_id=memory_store_id,
        access=access,
        instructions="",
        name=name,
        description="",
        mount_path=f"/mnt/memory/{name}",
    )


def _seed(
    registry: SandboxRegistry,
    session_id: str,
    snapshot: frozenset[tuple[str, ...]] = frozenset(),
) -> SandboxHandle:
    handle = make_handle(session_id=session_id, mount_snapshot=snapshot)
    registry._handles[session_id] = handle
    registry._last_used[session_id] = 0.0
    return handle


class TestReleaseIfMountsChanged:
    async def test_no_cached_handle_is_noop(self) -> None:
        backend = FakeBackend()
        registry = SandboxRegistry(backend=backend)
        await registry.release_if_mounts_changed("sess_NONE", [_echo("memstore_a", "a")], [])
        destroys = [c for c in backend.calls if c[0] == "destroy"]
        assert destroys == []

    async def test_identical_snapshot_does_not_release(self) -> None:
        backend = FakeBackend()
        registry = SandboxRegistry(backend=backend)
        snapshot = frozenset([("memstore", "memstore_a", "a", "read_write")])
        _seed(registry, "sess_X", snapshot)

        await registry.release_if_mounts_changed("sess_X", [_echo("memstore_a", "a")], [])

        destroys = [c for c in backend.calls if c[0] == "destroy"]
        assert destroys == []
        assert registry.peek("sess_X") is not None

    async def test_reorder_does_not_release(self) -> None:
        backend = FakeBackend()
        registry = SandboxRegistry(backend=backend)
        snapshot = frozenset(
            [
                ("memstore", "memstore_a", "a", "read_write"),
                ("memstore", "memstore_b", "b", "read_only"),
            ]
        )
        _seed(registry, "sess_X", snapshot)

        await registry.release_if_mounts_changed(
            "sess_X",
            [_echo("memstore_b", "b", "read_only"), _echo("memstore_a", "a")],
            [],
        )

        destroys = [c for c in backend.calls if c[0] == "destroy"]
        assert destroys == []
        assert registry.peek("sess_X") is not None

    @pytest.mark.parametrize(
        "current_echoes,description",
        [
            ([_echo("memstore_a", "a"), _echo("memstore_b", "b")], "added a store"),
            ([], "detached all stores"),
            ([_echo("memstore_a", "renamed")], "name_at_attach changed"),
            ([_echo("memstore_a", "a", access="read_only")], "access changed"),
            ([_echo("memstore_other", "a")], "store_id changed"),
        ],
    )
    async def test_diverging_snapshot_releases(
        self, current_echoes: list[MemoryStoreResourceEcho], description: str
    ) -> None:
        backend = FakeBackend()
        registry = SandboxRegistry(backend=backend)
        snapshot = frozenset([("memstore", "memstore_a", "a", "read_write")])
        handle = _seed(registry, "sess_X", snapshot)

        await registry.release_if_mounts_changed("sess_X", current_echoes, [])

        destroys = [c for c in backend.calls if c[0] == "destroy"]
        assert len(destroys) == 1
        assert destroys[0][1]["sandbox_id"] == handle.sandbox_id
        assert registry.peek("sess_X") is None
        assert "sess_X" not in registry._last_used

    async def test_release_serialized_against_get_or_provision(self) -> None:
        """The per-session lock must serialize release_if_mounts_changed against
        a concurrent get_or_provision so the registry never returns a handle
        that's about to be torn down."""
        backend = FakeBackend()
        registry = SandboxRegistry(backend=backend)
        old_snapshot = frozenset([("memstore", "memstore_a", "a", "read_write")])
        old_handle = _seed(registry, "sess_X", old_snapshot)
        new_snapshot = frozenset([("memstore", "memstore_b", "b", "read_write")])

        provision_started = asyncio.Event()
        let_provision_continue = asyncio.Event()

        async def slow_build_spec(_session_id: str) -> ProvisioningPlan:
            provision_started.set()
            await let_provision_continue.wait()
            spec = SandboxSpec(
                session_id="sess_X",
                instance_id="inst_T",
                workspace=Mount(host_path=old_handle.workspace_path, sandbox_path="/workspace"),
                extra_mounts=(),
                environment={},
                labels={},
                network_policy=Unrestricted(),
                host_gateway_alias=None,
                image="aios-sandbox:test",
                mount_snapshot=new_snapshot,
            )
            return ProvisioningPlan(
                spec=spec,
                env_config=None,
                memory_echoes=[],
                github_echoes=[],
                git_proxy=None,
            )

        # Force a cache miss so get_or_provision contends for the lock.
        registry._handles.pop("sess_X")

        with (
            patch("aios.sandbox.registry.build_spec_from_session", slow_build_spec),
            patch("aios.sandbox.registry.ensure_workspace_runtime_dirs", AsyncMock()),
            patch("aios.sandbox.registry.install_packages", AsyncMock()),
            patch("aios.sandbox.registry.apply_network_lockdown", AsyncMock()),
        ):
            provision_task = asyncio.create_task(registry.get_or_provision("sess_X"))
            await provision_started.wait()

            release_task = asyncio.create_task(
                registry.release_if_mounts_changed("sess_X", [_echo("memstore_b", "b")], [])
            )
            # release_task is blocked on the lock provision_task is holding.
            await asyncio.sleep(0)
            assert not release_task.done()

            let_provision_continue.set()
            await provision_task
            await release_task

        # New handle's snapshot matches current echoes → no release fired.
        destroys = [c for c in backend.calls if c[0] == "destroy"]
        assert destroys == []


class TestLocksDictCleanup:
    """``SandboxRegistry._locks`` is reclaimed at registry teardown
    (``release_all``) only.  Both ``release()`` and ``evict()``
    deliberately keep the per-session entry so a concurrent
    ``get_or_provision`` (or one already in-flight) serializes via
    the same lock instance — popping while another task is mid-
    provision would let it run unprotected against the cleanup's
    broker-secret unregistration, wedging the new sandbox.  One
    ``asyncio.Lock`` per ever-touched session_id is a bounded leak
    that the worker's eventual restart clears; the race is
    unacceptable."""

    async def test_release_all_clears_locks(self) -> None:
        backend = FakeBackend()
        registry = SandboxRegistry(backend=backend)
        _seed(registry, "sess_X")
        _seed(registry, "sess_Y")
        _ = registry._lock_for("sess_X")
        _ = registry._lock_for("sess_Y")
        assert set(registry._locks) == {"sess_X", "sess_Y"}

        await registry.release_all()

        assert registry._locks == {}

    async def test_evict_does_not_pop_lock_entry(self) -> None:
        """``evict()`` must also keep ``_locks[sid]`` so an in-flight
        ``get_or_provision`` that already acquired the lock isn't left
        holding an instance that's no longer findable — a third caller
        arriving via ``_lock_for(sid)`` would then create a new lock
        and race with the in-flight provision."""
        backend = FakeBackend()
        registry = SandboxRegistry(backend=backend)
        _seed(registry, "sess_X")
        original_lock = registry._lock_for("sess_X")

        registry.evict("sess_X")

        assert registry._lock_for("sess_X") is original_lock

    async def test_release_does_not_pop_lock_entry_mid_release(self) -> None:
        """``release()`` must NOT pop ``_locks[sid]`` mid-release.

        ``release_if_mounts_changed`` and the idle reaper both wrap
        ``release()`` in ``async with self._lock_for(sid)`` so that a
        concurrent ``get_or_provision`` serializes behind them.  If
        ``release()`` pops ``_locks[sid]`` while its awaits (proxy
        stop, ``backend.destroy``) are still in flight, the lock the
        caller holds is no longer findable in the dict — a concurrent
        ``get_or_provision`` would call ``_lock_for()``, see no entry,
        and create a NEW lock.  Both tasks then run unprotected, and
        ``release()``'s subsequent ``_release_tool_broker_secret`` call
        unregisters the new provision's broker secret, wedging the
        new sandbox with no MCP authentication until the next eviction
        cycle.
        """
        backend = FakeBackend()
        registry = SandboxRegistry(backend=backend)
        _seed(registry, "sess_X")

        destroy_started = asyncio.Event()
        destroy_unblock = asyncio.Event()

        async def blocking_destroy(handle: SandboxHandle) -> None:
            destroy_started.set()
            await destroy_unblock.wait()

        backend.destroy = blocking_destroy  # type: ignore[method-assign]

        original_lock = registry._lock_for("sess_X")

        async def release_task() -> None:
            async with registry._lock_for("sess_X"):
                await registry.release("sess_X")

        task = asyncio.create_task(release_task())
        try:
            await destroy_started.wait()

            # While release() is in progress, ``_lock_for`` MUST return
            # the same lock instance the caller is holding.  Otherwise a
            # concurrent provision creates a new lock and races.
            lock_during_release = registry._lock_for("sess_X")
            assert lock_during_release is original_lock, (
                "release() popped _locks[sid] mid-release; a concurrent "
                "get_or_provision would create a new lock and race with "
                "the still-running release."
            )
        finally:
            destroy_unblock.set()
            await task


class TestEvictReclamation:
    """``evict`` reclaims the per-session ``GitProxy`` through ``stop()``."""

    async def test_evict_records_stop_task_in_strong_ref_set(self) -> None:
        """evict() must strongly reference its fire-and-forget stop task
        while it runs and discard it once the task completes."""
        backend = FakeBackend()
        registry = SandboxRegistry(backend=backend)

        class _StubProxy:
            async def stop(self) -> None: ...

        registry._git_proxies["sess_X"] = cast(Any, _StubProxy())
        registry._handles["sess_X"] = make_handle(session_id="sess_X")
        registry._last_used["sess_X"] = 0.0

        registry.evict("sess_X")
        # set.add runs synchronously inside evict — the loop hasn't yielded
        # yet, so the new task is still in flight.
        assert len(registry._evict_proxy_stop_tasks) == 1

        for _ in range(3):
            await asyncio.sleep(0)
        # add_done_callback(discard) reclaims the slot — otherwise the
        # set grows unboundedly across the worker's lifetime.
        assert len(registry._evict_proxy_stop_tasks) == 0

    async def test_evict_proxy_actually_stops_under_gc(self) -> None:
        """GC between evict() and the loop's first yield must not lose the
        stop task — asyncio only weak-refs tasks, so without the strong-ref
        set the freshly created task is eligible for collection here."""
        backend = FakeBackend()
        registry = SandboxRegistry(backend=backend)

        proxy_stopped = asyncio.Event()

        class _StubProxy:
            async def stop(self) -> None:
                proxy_stopped.set()

        registry._git_proxies["sess_X"] = cast(Any, _StubProxy())
        registry._handles["sess_X"] = make_handle(session_id="sess_X")
        registry._last_used["sess_X"] = 0.0

        registry.evict("sess_X")
        gc.collect()

        await asyncio.wait_for(proxy_stopped.wait(), timeout=1.0)


def _provisioning_plan(session_id: str) -> ProvisioningPlan:
    """Build a minimal :class:`ProvisioningPlan` for stale-handle tests.

    The plan is returned by a patched ``build_spec_from_session`` so the
    registry's cold path runs without touching the DB. The fresh handle's
    sandbox_id is controlled by the caller via ``FakeBackend.next_handle_id``;
    the spec itself doesn't carry one.
    """
    spec = SandboxSpec(
        session_id=session_id,
        instance_id="inst_TEST",
        workspace=Mount(host_path=Path("/tmp/w"), sandbox_path="/workspace"),
        extra_mounts=(),
        environment={},
        labels={},
        network_policy=Unrestricted(),
        host_gateway_alias=None,
        image="aios-sandbox:test",
    )
    return ProvisioningPlan(
        spec=spec,
        env_config=None,
        memory_echoes=[],
        github_echoes=[],
        git_proxy=None,
    )


def _provisioning_plan_limited(session_id: str) -> ProvisioningPlan:
    """Like :func:`_provisioning_plan` but with a :class:`Limited` network
    policy, so the registry's cold path actually runs
    ``apply_network_lockdown`` (the security gate under test)."""
    from aios.models.environments import EnvironmentConfig, LimitedNetworking
    from aios.sandbox.backends.base import Limited

    spec = SandboxSpec(
        session_id=session_id,
        instance_id="inst_TEST",
        workspace=Mount(host_path=Path("/tmp/w"), sandbox_path="/workspace"),
        extra_mounts=(),
        environment={},
        labels={},
        network_policy=Limited(allowed_hosts=frozenset({"api.example.com"})),
        host_gateway_alias=None,
        image="aios-sandbox:test",
    )
    return ProvisioningPlan(
        spec=spec,
        env_config=EnvironmentConfig(
            networking=LimitedNetworking(type="limited", allowed_hosts=["api.example.com"])
        ),
        memory_echoes=[],
        github_echoes=[],
        git_proxy=None,
    )


class TestLockdownFailsClosed:
    """A :class:`Limited` provision whose iptables lockdown fails must tear
    the sandbox down and abort, never hand back a box with unrestricted
    networking (silent #724 image-override bypass otherwise)."""

    async def test_lockdown_failure_destroys_sandbox_and_raises(self) -> None:
        backend = FakeBackend()
        # The post-create setup execs all route through backend.exec; make
        # the lockdown script (the only exec the cold path runs here, since
        # the other setup steps are patched out) return nonzero.
        backend.next_result = CommandResult(
            exit_code=2,
            stdout="",
            stderr="iptables: not found",
            timed_out=False,
            truncated=False,
        )
        registry = SandboxRegistry(backend=backend)

        broker = MagicMock()
        broker.port = 8765

        with (
            patch("aios.harness.runtime.require_tool_broker", lambda: broker),
            patch(
                "aios.sandbox.registry.build_spec_from_session",
                AsyncMock(return_value=_provisioning_plan_limited("sess_X")),
            ),
            patch("aios.sandbox.registry.ensure_workspace_runtime_dirs", AsyncMock()),
            patch("aios.sandbox.registry.install_packages", AsyncMock()),
            # NOTE: apply_network_lockdown is NOT patched — we want the real
            # fail-closed behavior to fire against the failing backend.exec.
        ):
            from aios.sandbox.backends.base import SandboxBackendError

            with pytest.raises(SandboxBackendError, match="network lockdown failed"):
                await registry.get_or_provision("sess_X")

        # Sandbox was created then torn down — no live handle left behind.
        assert any(c[0] == "create" for c in backend.calls)
        destroys = [c for c in backend.calls if c[0] == "destroy"]
        assert len(destroys) == 1, "failed lockdown must destroy the just-created sandbox"
        assert registry.peek("sess_X") is None, (
            "a Limited sandbox whose lockdown failed must not remain cached"
        )


class TestStaleHandleDetection:
    """``get_or_provision`` validates handle liveness on every warm hit.

    Issue #691: containers spawned with ``--rm`` auto-remove on any exit
    (entrypoint exit, OOM, daemon-side cleanup). The registry's cached
    handle outlives the dead container, so the next ``exec`` against the
    cached handle fails with Docker's "No such container". The fix is to
    probe the backend (``is_alive``) on the warm path and treat a
    not-alive answer the same as a cache miss — evict the dead cache
    entry and re-provision under the per-session lock.

    All tests patch the registry's provisioning helpers so the cold
    path runs without DB or Docker; they assert behavior via the
    :class:`FakeBackend` call log.
    """

    async def test_live_handle_returned_without_reprovision(self) -> None:
        """Default warm path: ``is_alive`` returns True ⇒ cached handle returned."""
        backend = FakeBackend()
        registry = SandboxRegistry(backend=backend)
        cached = _seed(registry, "sess_X")

        result = await registry.get_or_provision("sess_X")

        assert result is cached
        assert any(c[0] == "is_alive" for c in backend.calls)
        assert not any(c[0] == "create" for c in backend.calls), (
            "live handle must not trigger re-provision"
        )

    async def test_dead_handle_triggers_reprovision(self) -> None:
        """Stale cached handle ⇒ evict + provision; caller gets the new handle."""
        backend = FakeBackend(next_handle_id="fresh_sandbox_id")
        registry = SandboxRegistry(backend=backend)
        stale = _seed(registry, "sess_X")
        backend.dead_sandbox_ids.add(stale.sandbox_id)

        with (
            patch(
                "aios.sandbox.registry.build_spec_from_session",
                AsyncMock(return_value=_provisioning_plan("sess_X")),
            ),
            patch("aios.sandbox.registry.ensure_workspace_runtime_dirs", AsyncMock()),
            patch("aios.sandbox.registry.install_packages", AsyncMock()),
            patch("aios.sandbox.registry.apply_network_lockdown", AsyncMock()),
        ):
            result = await registry.get_or_provision("sess_X")

        assert result is not stale
        assert result.sandbox_id == "fresh_sandbox_id"
        assert registry.peek("sess_X") is result
        create_calls = [c for c in backend.calls if c[0] == "create"]
        assert len(create_calls) == 1, (
            "stale-handle path must re-provision exactly once via backend.create"
        )

    async def test_cold_start_skips_liveness_check(self) -> None:
        """Empty cache ⇒ no ``is_alive`` (nothing to check)."""
        backend = FakeBackend()
        registry = SandboxRegistry(backend=backend)

        with (
            patch(
                "aios.sandbox.registry.build_spec_from_session",
                AsyncMock(return_value=_provisioning_plan("sess_X")),
            ),
            patch("aios.sandbox.registry.ensure_workspace_runtime_dirs", AsyncMock()),
            patch("aios.sandbox.registry.install_packages", AsyncMock()),
            patch("aios.sandbox.registry.apply_network_lockdown", AsyncMock()),
        ):
            await registry.get_or_provision("sess_X")

        assert not any(c[0] == "is_alive" for c in backend.calls), (
            "cold start has no cached handle to probe — is_alive must not fire"
        )

    async def test_dead_handle_eviction_clears_proxy_and_broker_secret(self) -> None:
        """The evict-before-reprovision must release the proxy and broker secret.

        Without this, the new sandbox would inherit the dead handle's proxy
        (its uvicorn server and TCP port) and the broker would still serve
        the dead session's secret to the new container. ``evict()`` is the
        method that bundles these; verify the stale-handle path delegates
        through it.
        """
        backend = FakeBackend(next_handle_id="fresh_sandbox_id")
        registry = SandboxRegistry(backend=backend)
        stale = _seed(registry, "sess_X")
        backend.dead_sandbox_ids.add(stale.sandbox_id)

        proxy_stopped = asyncio.Event()

        class _StubProxy:
            async def stop(self) -> None:
                proxy_stopped.set()

        registry._git_proxies["sess_X"] = cast(Any, _StubProxy())

        broker = MagicMock()
        broker.port = 0

        with (
            patch(
                "aios.harness.runtime.require_tool_broker",
                lambda: broker,
            ),
            patch(
                "aios.sandbox.registry.build_spec_from_session",
                AsyncMock(return_value=_provisioning_plan("sess_X")),
            ),
            patch("aios.sandbox.registry.ensure_workspace_runtime_dirs", AsyncMock()),
            patch("aios.sandbox.registry.install_packages", AsyncMock()),
            patch("aios.sandbox.registry.apply_network_lockdown", AsyncMock()),
        ):
            await registry.get_or_provision("sess_X")

        # Proxy stop is fire-and-forget — give the loop a chance to run it.
        await asyncio.wait_for(proxy_stopped.wait(), timeout=1.0)
        # Broker secret is unregistered immediately (synchronous part of evict).
        broker.unregister_session.assert_called_with("sess_X")

    async def test_recycle_preserves_session_runtime_caches(self) -> None:
        """The mid-step recycle must NOT clear ``_session_memory_mounts``/
        ``_session_read_shas`` (issue #691 follow-up — data-loss guard).

        Ordering that makes this load-bearing: ``bash_handler`` calls
        ``get_or_provision`` and only THEN snapshots the memory mounts for
        its post-exec reconcile. The mounts cache is populated once at step
        start (``loop.py``) and is NOT repopulated by provisioning. If the
        stale-handle recycle cleared it, the bash reconcile's before/after
        diff would see no mounts and silently drop every memory-store write
        the command made. So a recycle inside get_or_provision must leave
        the session caches intact (the fresh sandbox re-mounts the same
        host dirs).
        """
        runtime._session_memory_mounts.clear()
        runtime._session_read_shas.clear()
        try:
            backend = FakeBackend(next_handle_id="fresh_sandbox_id")
            registry = SandboxRegistry(backend=backend)
            stale = _seed(registry, "sess_X")
            backend.dead_sandbox_ids.add(stale.sandbox_id)

            echo = _echo("memstore_a", "a")
            runtime.set_session_memory_mounts("sess_X", [echo])
            runtime.set_read_sha("sess_X", "memstore_a", "/notes.md", "sha_before")

            broker = MagicMock()
            broker.port = 0
            with (
                patch("aios.harness.runtime.require_tool_broker", lambda: broker),
                patch(
                    "aios.sandbox.registry.build_spec_from_session",
                    AsyncMock(return_value=_provisioning_plan("sess_X")),
                ),
                patch("aios.sandbox.registry.ensure_workspace_runtime_dirs", AsyncMock()),
                patch("aios.sandbox.registry.install_packages", AsyncMock()),
                patch("aios.sandbox.registry.apply_network_lockdown", AsyncMock()),
            ):
                await registry.get_or_provision("sess_X")

            assert runtime.get_session_memory_mounts("sess_X") == [echo], (
                "recycle cleared the memory-mounts cache mid-step → bash reconcile "
                "would see no mounts and lose the step's memory writes"
            )
            assert runtime.get_read_sha("sess_X", "memstore_a", "/notes.md") == "sha_before", (
                "recycle cleared the read-sha cache mid-step → spurious precondition misses"
            )
        finally:
            runtime._session_memory_mounts.clear()
            runtime._session_read_shas.clear()

    async def test_concurrent_stale_handle_reprovisions_once(self) -> None:
        """Two concurrent ``get_or_provision`` calls both seeing the same
        stale handle must re-provision exactly once.

        This exercises both arms of the under-lock identity check: the
        winner (task_a) finds its captured stale handle still cached and
        recycles it; the loser (task_b) acquires the lock after task_a
        installed a *different* fresh handle and trusts it via identity
        rather than recycling+reprovisioning a third container. Without
        the identity guard the loser would churn another container.
        """
        backend = FakeBackend(next_handle_id="fresh_sandbox_id")
        registry = SandboxRegistry(backend=backend)
        stale = _seed(registry, "sess_X")
        backend.dead_sandbox_ids.add(stale.sandbox_id)

        # Gate the cold-path provision so the second task piles up on the lock.
        let_provision_continue = asyncio.Event()
        first_provision_started = asyncio.Event()

        async def slow_build_spec(_session_id: str) -> ProvisioningPlan:
            first_provision_started.set()
            await let_provision_continue.wait()
            return _provisioning_plan("sess_X")

        with (
            patch("aios.sandbox.registry.build_spec_from_session", slow_build_spec),
            patch("aios.sandbox.registry.ensure_workspace_runtime_dirs", AsyncMock()),
            patch("aios.sandbox.registry.install_packages", AsyncMock()),
            patch("aios.sandbox.registry.apply_network_lockdown", AsyncMock()),
        ):
            task_a = asyncio.create_task(registry.get_or_provision("sess_X"))
            # Wait until task_a is inside _provision (lock held).
            await first_provision_started.wait()
            task_b = asyncio.create_task(registry.get_or_provision("sess_X"))
            # task_b is now contending for the lock. Let task_a finish.
            await asyncio.sleep(0)
            let_provision_continue.set()
            result_a, result_b = await asyncio.gather(task_a, task_b)

        # Both callers see the fresh handle.
        assert result_a.sandbox_id == "fresh_sandbox_id"
        assert result_b.sandbox_id == "fresh_sandbox_id"
        # Exactly one create call across the two callers.
        create_calls = [c for c in backend.calls if c[0] == "create"]
        assert len(create_calls) == 1, (
            f"expected exactly one re-provision; got {len(create_calls)} (calls={backend.calls})"
        )


def _seed_versioned(
    registry: SandboxRegistry, session_id: str, *, spec_version: int
) -> SandboxHandle:
    """Seed a live cached handle stamped with ``spec_version`` (#713)."""
    handle = make_handle(session_id=session_id, spec_version=spec_version)
    registry._handles[session_id] = handle
    registry._last_used[session_id] = 0.0
    return handle


class TestSpecVersionDrift:
    """``get_or_provision`` recycles a live cached sandbox when the
    session's ``spec_version`` has drifted past the snapshot stamped on
    the handle (#713).

    The version probe is the API-process / direct-SQL safety net behind
    the worker-only write-path eviction: a memory store or github repo
    attached/detached between steps bumps ``sessions.spec_version`` via a
    Postgres trigger, and the next warm hit notices the mismatch and
    re-provisions even though the cached container is still alive.

    The probe only runs when ``get_or_provision`` is passed a ``pool``
    (the cold-start span path already passes one); without a pool the
    version check is skipped so the existing lock/liveness tests keep
    their no-DB behavior. All tests patch the registry's provisioning
    helpers so the cold path runs without DB or Docker.
    """

    async def test_warm_hit_recycles_when_spec_version_changed(self) -> None:
        """Live handle, current spec_version > snapshot ⇒ recycle."""
        backend = FakeBackend(next_handle_id="fresh_sandbox_id")
        registry = SandboxRegistry(backend=backend)
        stale = _seed_versioned(registry, "sess_X", spec_version=1)

        with (
            patch(
                "aios.sandbox.registry.queries.unscoped_get_session_spec_version",
                AsyncMock(return_value=2),
            ),
            # The recycle path re-provisions via the span wrapper, which
            # looks up the account and appends a sandbox_provision span
            # pair; stub both so the MagicMock pool isn't driven through
            # real SQL.
            patch(
                "aios.services.sessions.load_session_account_id",
                AsyncMock(return_value="acct_x"),
            ),
            patch("aios.services.sessions.append_event", AsyncMock()),
            patch(
                "aios.sandbox.registry.build_spec_from_session",
                AsyncMock(return_value=_provisioning_plan("sess_X")),
            ),
            patch("aios.sandbox.registry.ensure_workspace_runtime_dirs", AsyncMock()),
            patch("aios.sandbox.registry.install_packages", AsyncMock()),
            patch("aios.sandbox.registry.apply_network_lockdown", AsyncMock()),
        ):
            result = await registry.get_or_provision("sess_X", pool=cast(Any, MagicMock()))

        assert result is not stale
        assert result.sandbox_id == "fresh_sandbox_id"
        create_calls = [c for c in backend.calls if c[0] == "create"]
        assert len(create_calls) == 1, "spec-version drift must re-provision exactly once"
        # The alive-but-drifted container must be destroyed (not just evicted):
        # evict() skips backend.destroy (designed for dead containers), but a
        # spec-version drift recycles a LIVE container — not destroying it would
        # leave it running until the next worker restart (#713 fix).
        destroys = [c for c in backend.calls if c[0] == "destroy"]
        assert len(destroys) == 1, "spec-version drift must backend.destroy the live old container"

    async def test_warm_hit_returns_handle_when_spec_version_matches(self) -> None:
        """Live handle, current spec_version == snapshot ⇒ cached returned."""
        backend = FakeBackend()
        registry = SandboxRegistry(backend=backend)
        cached = _seed_versioned(registry, "sess_X", spec_version=3)

        with patch(
            "aios.sandbox.registry.queries.unscoped_get_session_spec_version",
            AsyncMock(return_value=3),
        ):
            result = await registry.get_or_provision("sess_X", pool=cast(Any, MagicMock()))

        assert result is cached
        assert not any(c[0] == "create" for c in backend.calls), (
            "matching spec_version must not re-provision"
        )

    async def test_warm_hit_without_pool_skips_version_check(self) -> None:
        """No pool ⇒ the version query is never called; cached handle returned."""
        backend = FakeBackend()
        registry = SandboxRegistry(backend=backend)
        cached = _seed_versioned(registry, "sess_X", spec_version=1)

        version_query = AsyncMock(return_value=99)
        with patch(
            "aios.sandbox.registry.queries.unscoped_get_session_spec_version", version_query
        ):
            result = await registry.get_or_provision("sess_X")

        assert result is cached
        version_query.assert_not_awaited()
        assert not any(c[0] == "create" for c in backend.calls)

    async def test_spec_version_probe_failure_does_not_recycle(self) -> None:
        """A transient version-read error must not churn a healthy sandbox."""
        backend = FakeBackend()
        registry = SandboxRegistry(backend=backend)
        cached = _seed_versioned(registry, "sess_X", spec_version=1)

        with patch(
            "aios.sandbox.registry.queries.unscoped_get_session_spec_version",
            AsyncMock(side_effect=RuntimeError("db hiccup")),
        ):
            result = await registry.get_or_provision("sess_X", pool=cast(Any, MagicMock()))

        assert result is cached, "a failed probe must return the live cached handle"
        assert not any(c[0] == "create" for c in backend.calls)

    async def test_handle_carries_spec_version_from_spec(self) -> None:
        """Cold provision stamps the plan's spec_version onto the cached handle."""
        backend = FakeBackend()
        registry = SandboxRegistry(backend=backend)
        plan = _provisioning_plan("sess_X")
        # Rebuild the spec with a non-default spec_version to prove the
        # backend copies it through to the handle.
        from dataclasses import replace

        plan = ProvisioningPlan(
            spec=replace(plan.spec, spec_version=7),
            env_config=plan.env_config,
            memory_echoes=plan.memory_echoes,
            github_echoes=plan.github_echoes,
            git_proxy=plan.git_proxy,
        )

        with (
            patch(
                "aios.sandbox.registry.build_spec_from_session",
                AsyncMock(return_value=plan),
            ),
            patch("aios.sandbox.registry.ensure_workspace_runtime_dirs", AsyncMock()),
            patch("aios.sandbox.registry.install_packages", AsyncMock()),
            patch("aios.sandbox.registry.apply_network_lockdown", AsyncMock()),
        ):
            await registry.get_or_provision("sess_X")

        handle = registry.peek("sess_X")
        assert handle is not None
        assert handle.spec_version == 7
