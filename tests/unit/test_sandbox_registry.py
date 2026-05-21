"""Unit tests for ``SandboxRegistry`` worker-side cleanup paths.

``TestReleaseIfMountsChanged`` covers the mount-drift detector that
recycles a cached sandbox when the session's attached memory stores
have changed since provisioning (issue #198). ``TestEvictReclamation``
covers the ``evict`` fast path used by ``tool_dispatch`` when a
sandbox-side failure suggests the sandbox itself is unhealthy.
"""

from __future__ import annotations

import asyncio
import gc
from typing import Any, cast

import pytest

from aios.models.memory_stores import Access, MemoryStoreResourceEcho
from aios.sandbox.backends.base import SandboxHandle
from aios.sandbox.registry import SandboxRegistry
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
        from unittest.mock import AsyncMock

        from aios.sandbox.spec import ProvisioningPlan

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
            from aios.sandbox.backends.base import Mount, SandboxSpec, Unrestricted

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

        from unittest.mock import patch

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
        ``release()``'s subsequent ``_release_mcp_broker_secret`` call
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
