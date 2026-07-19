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
import dataclasses
import gc
from collections.abc import Iterator
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.config import get_settings
from aios.db.queries import EnvVarCredentialEcho
from aios.harness import runtime
from aios.ids import VAULT_CREDENTIAL
from aios.models.environments import UnrestrictedNetworking
from aios.models.memory_stores import Access, MemoryStoreResourceEcho
from aios.sandbox.backends.base import (
    CommandResult,
    ManagedSandboxRef,
    Mount,
    SandboxBackendError,
    SandboxHandle,
    SandboxSpec,
)
from aios.sandbox.registry import SandboxRegistry
from aios.sandbox.spec import ProvisioningPlan
from aios.services.vaults import ResolvedEnvVarCredential
from tests.helpers.sandbox import FakeBackend, FakePool, make_handle


@pytest.fixture(autouse=True)
def _fake_runtime_pool() -> Iterator[None]:
    """Provide a fake ``runtime.pool`` so ``release()``'s snapshot-pointer write
    succeeds — otherwise ``require_pool()`` raises, the pointer write is treated
    as a snapshot failure, and ``backend.destroy`` is never reached (which would
    hang the blocking-destroy test and fail the mount-drift destroy assertions)."""
    prev = runtime.pool
    runtime.pool = cast(Any, FakePool())
    try:
        yield
    finally:
        runtime.pool = prev


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


def _env_echo(credential_id: str, updated_at: datetime) -> EnvVarCredentialEcho:
    return EnvVarCredentialEcho(credential_id=credential_id, updated_at=updated_at)


class FakeSecretProxy:
    """Stand-in for ``SecretEgressProxy`` — records ``stop()`` calls and
    exposes a fake ``port`` so registry lifecycle paths can be exercised
    without booting a real TLS server (#877)."""

    def __init__(self, port: int = 49152) -> None:
        self.port = port
        self.stop = AsyncMock()


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
        await registry.release_if_mounts_changed("sess_NONE", [_echo("memstore_a", "a")], [], [])
        destroys = [c for c in backend.calls if c[0] == "destroy"]
        assert destroys == []

    async def test_identical_snapshot_does_not_release(self) -> None:
        backend = FakeBackend()
        registry = SandboxRegistry(backend=backend)
        snapshot = frozenset([("memstore", "memstore_a", "a", "read_write")])
        _seed(registry, "sess_X", snapshot)

        await registry.release_if_mounts_changed("sess_X", [_echo("memstore_a", "a")], [], [])

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

        await registry.release_if_mounts_changed("sess_X", current_echoes, [], [])

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
                network_policy=UnrestrictedNetworking(),
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
                env_var_credentials=(),
            )

        # Force a cache miss so get_or_provision contends for the lock.
        registry._handles.pop("sess_X")

        with (
            patch("aios.sandbox.registry.build_spec_from_session", slow_build_spec),
            patch("aios.sandbox.registry.install_egress_ca", AsyncMock()),
            patch("aios.sandbox.registry.install_packages", AsyncMock()),
            patch("aios.sandbox.registry.apply_network_lockdown", AsyncMock()),
        ):
            provision_task = asyncio.create_task(registry.get_or_provision("sess_X"))
            await provision_started.wait()

            release_task = asyncio.create_task(
                registry.release_if_mounts_changed("sess_X", [_echo("memstore_b", "b")], [], [])
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
    (``stop_all``) only.  Both ``release()`` and ``evict()``
    deliberately keep the per-session entry so a concurrent
    ``get_or_provision`` (or one already in-flight) serializes via
    the same lock instance — popping while another task is mid-
    provision would let it run unprotected against the cleanup's
    broker-secret unregistration, wedging the new sandbox.  One
    ``asyncio.Lock`` per ever-touched session_id is a bounded leak
    that the worker's eventual restart clears; the race is
    unacceptable."""

    async def test_stop_all_clears_locks(self) -> None:
        backend = FakeBackend()
        registry = SandboxRegistry(backend=backend)
        _seed(registry, "sess_X")
        _seed(registry, "sess_Y")
        _ = registry._lock_for("sess_X")
        _ = registry._lock_for("sess_Y")
        assert set(registry._locks) == {"sess_X", "sess_Y"}

        await registry.stop_all()

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
        network_policy=UnrestrictedNetworking(),
        host_gateway_alias=None,
        image="aios-sandbox:test",
    )
    return ProvisioningPlan(
        spec=spec,
        env_config=None,
        memory_echoes=[],
        github_echoes=[],
        git_proxy=None,
        env_var_credentials=(),
    )


def _provisioning_plan_limited(session_id: str) -> ProvisioningPlan:
    """Like :func:`_provisioning_plan` but with a :class:`Limited` network
    policy, so the registry's cold path actually runs
    ``apply_network_lockdown`` (the security gate under test)."""
    from aios.models.environments import EnvironmentConfig, LimitedNetworking

    spec = SandboxSpec(
        session_id=session_id,
        instance_id="inst_TEST",
        workspace=Mount(host_path=Path("/tmp/w"), sandbox_path="/workspace"),
        extra_mounts=(),
        environment={},
        labels={},
        network_policy=LimitedNetworking(type="limited", allowed_hosts=["api.example.com"]),
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
        env_var_credentials=(),
    )


class TestLockdownFailsClosed:
    """A :class:`Limited` provision whose iptables lockdown fails must tear
    the sandbox down and abort, never hand back a box with unrestricted
    networking (silent #724 image-override bypass otherwise)."""

    async def test_lockdown_failure_destroys_sandbox_and_raises(self) -> None:
        backend = FakeBackend()
        # Durable session sandboxes: the lockdown is applied from the netns
        # sidecar, not backend.exec. Make the apply sidecar return nonzero so
        # the fail-closed gate fires (a poisoned/missing iptables in the
        # sidecar's operator image, or a partial flush).
        backend.sidecar_results = [
            CommandResult(
                exit_code=2,
                stdout="",
                stderr="iptables: not found",
                timed_out=False,
                truncated=False,
            )
        ]
        registry = SandboxRegistry(backend=backend)

        broker = MagicMock()
        broker.port = 8765

        with (
            patch("aios.harness.runtime.require_tool_broker", lambda: broker),
            patch(
                "aios.sandbox.registry.build_spec_from_session",
                AsyncMock(return_value=_provisioning_plan_limited("sess_X")),
            ),
            patch("aios.sandbox.registry.install_egress_ca", AsyncMock()),
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


class TestSecretProxyDnatThreading:
    """The registry threads the secret-egress-proxy endpoint + the credential
    hosts into ``apply_network_lockdown`` so the lockdown can DNAT credential
    :443 egress through the proxy (#878).

    ``apply_network_lockdown`` is patched with an ``AsyncMock`` spy:
    ``_apply_egress_rules``'s extraction logic still runs in full, so the
    captured kwargs reflect the real wiring without booting a sidecar.
    """

    @staticmethod
    def _provision_patches(plan: ProvisioningPlan, lockdown: AsyncMock) -> ExitStack:
        """Enter the cold-provision dep patches + a spy ``apply_network_lockdown``."""
        broker = MagicMock()
        broker.port = 8765
        stack = ExitStack()
        for cm in (
            patch("aios.harness.runtime.require_tool_broker", lambda: broker),
            patch(
                "aios.sandbox.registry.build_spec_from_session",
                AsyncMock(return_value=plan),
            ),
            patch("aios.sandbox.registry.install_egress_ca", AsyncMock()),
            patch("aios.sandbox.registry.install_packages", AsyncMock()),
            patch("aios.sandbox.registry.apply_network_lockdown", lockdown),
        ):
            stack.enter_context(cm)
        return stack

    async def test_secret_proxy_dnat_threaded_when_creds_present(self) -> None:
        backend = FakeBackend()
        registry = SandboxRegistry(backend=backend)
        cred = ResolvedEnvVarCredential(
            credential_id="cred_1",
            secret_name="API_TOKEN",
            secret_value="real-secret",
            allowed_hosts=("api.secret.com", "data.secret.com/v1"),
            updated_at=datetime(2025, 1, 1, tzinfo=UTC),
            placeholder="AIOS_SECRET_PLACEHOLDER_deadbeef",
        )
        base = _provisioning_plan_limited("sess_X")
        plan = ProvisioningPlan(
            spec=base.spec,
            env_config=base.env_config,
            memory_echoes=base.memory_echoes,
            github_echoes=base.github_echoes,
            git_proxy=base.git_proxy,
            env_var_credentials=(cred,),
            secret_proxy=cast(Any, FakeSecretProxy(port=49152)),
        )
        lockdown = AsyncMock()

        with self._provision_patches(plan, lockdown):
            await registry.get_or_provision("sess_X")

        kwargs = lockdown.call_args.kwargs
        assert ("aios-worker", 49152) in kwargs["extra_host_ports"]
        assert kwargs["dnat_target"] == ("aios-worker", 49152)
        # The ``/v1`` path prefix is stripped and the bare hosts de-dup'd.
        assert set(kwargs["dnat_hosts"]) == {"api.secret.com", "data.secret.com"}

    async def test_no_dnat_when_no_secret_proxy(self) -> None:
        backend = FakeBackend()
        registry = SandboxRegistry(backend=backend)
        plan = _provisioning_plan_limited("sess_X")  # secret_proxy None, no creds
        lockdown = AsyncMock()

        with self._provision_patches(plan, lockdown):
            await registry.get_or_provision("sess_X")

        kwargs = lockdown.call_args.kwargs
        assert kwargs["dnat_target"] is None
        assert not kwargs["dnat_hosts"]
        assert ("aios-worker", 49152) not in kwargs["extra_host_ports"]

    async def test_run_secret_proxy_recorded_and_dnat_threaded(self) -> None:
        backend = FakeBackend()
        registry = SandboxRegistry(backend=backend)
        cred = ResolvedEnvVarCredential(
            credential_id="cred_1",
            secret_name="API_TOKEN",
            secret_value="real-secret",
            allowed_hosts=("api.secret.com",),
            updated_at=datetime(2025, 1, 1, tzinfo=UTC),
            placeholder="AIOS_SECRET_PLACEHOLDER_deadbeef",
        )
        base = _provisioning_plan_limited("run_X")
        proxy = cast(Any, FakeSecretProxy(port=49152))
        plan = ProvisioningPlan(
            spec=base.spec,
            env_config=base.env_config,
            memory_echoes=base.memory_echoes,
            github_echoes=base.github_echoes,
            git_proxy=base.git_proxy,
            env_var_credentials=(cred,),
            secret_proxy=proxy,
        )
        lockdown = AsyncMock()
        broker = MagicMock()
        broker.port = 8765

        with ExitStack() as stack:
            for cm in (
                patch("aios.harness.runtime.require_tool_broker", lambda: broker),
                patch("aios.sandbox.registry.build_spec_from_run", AsyncMock(return_value=plan)),
                patch("aios.sandbox.registry.install_egress_ca", AsyncMock()),
                patch("aios.sandbox.registry.install_packages", AsyncMock()),
                patch("aios.sandbox.registry.apply_network_lockdown", lockdown),
            ):
                stack.enter_context(cm)
            await registry.get_or_provision_run("run_X")

        assert registry._secret_proxies["run_X"] is proxy
        kwargs = lockdown.call_args.kwargs
        assert kwargs["dnat_target"] == ("aios-worker", 49152)
        assert kwargs["dnat_hosts"] == ["api.secret.com"]

    async def test_run_create_failure_stops_secret_proxy(self) -> None:
        backend = FakeBackend()
        backend.create = AsyncMock(side_effect=RuntimeError("create failed"))  # type: ignore[method-assign]
        registry = SandboxRegistry(backend=backend)
        base = _provisioning_plan_limited("run_X")
        proxy = cast(Any, FakeSecretProxy())
        plan = ProvisioningPlan(
            spec=base.spec,
            env_config=base.env_config,
            memory_echoes=base.memory_echoes,
            github_echoes=base.github_echoes,
            git_proxy=None,
            env_var_credentials=(),
            secret_proxy=proxy,
        )

        with (
            patch("aios.sandbox.registry.build_spec_from_run", AsyncMock(return_value=plan)),
            pytest.raises(RuntimeError, match="create failed"),
        ):
            await registry.get_or_provision_run("run_X")

        proxy.stop.assert_awaited_once()
        assert "run_X" not in registry._secret_proxies

    async def test_release_run_stops_secret_proxy_without_handle(self) -> None:
        backend = FakeBackend()
        registry = SandboxRegistry(backend=backend)
        proxy = FakeSecretProxy()
        registry._secret_proxies["run_X"] = cast(Any, proxy)

        await registry.release_run("run_X")

        proxy.stop.assert_awaited_once()
        assert "run_X" not in registry._secret_proxies


class TestUnrestrictedSecretEgressDnat:
    """Under an Unrestricted (or no-config) environment that carries env-var
    credentials, the registry routes egress wiring to ``apply_secret_egress_dnat``
    (DNAT-only, general egress open) instead of ``apply_network_lockdown`` (#1153).
    Both are patched with spies so the branch selection + credential-host
    extraction run in full without booting a sidecar.
    """

    @staticmethod
    def _unrestricted_plan_with_creds(
        session_id: str, *, proxy_port: int = 49152
    ) -> ProvisioningPlan:
        cred = ResolvedEnvVarCredential(
            credential_id="cred_1",
            secret_name="API_TOKEN",
            secret_value="real-secret",
            allowed_hosts=("api.secret.com", "data.secret.com/v1"),
            updated_at=datetime(2025, 1, 1, tzinfo=UTC),
            placeholder="AIOS_SECRET_PLACEHOLDER_deadbeef",
        )
        base = _provisioning_plan(session_id)  # Unrestricted, env_config=None
        return ProvisioningPlan(
            spec=base.spec,
            env_config=base.env_config,
            memory_echoes=base.memory_echoes,
            github_echoes=base.github_echoes,
            git_proxy=base.git_proxy,
            env_var_credentials=(cred,),
            secret_proxy=cast(Any, FakeSecretProxy(port=proxy_port)),
        )

    @staticmethod
    def _patches(plan: ProvisioningPlan, lockdown: AsyncMock, dnat_only: AsyncMock) -> ExitStack:
        broker = MagicMock()
        broker.port = 8765
        stack = ExitStack()
        for cm in (
            patch("aios.harness.runtime.require_tool_broker", lambda: broker),
            patch("aios.sandbox.registry.build_spec_from_session", AsyncMock(return_value=plan)),
            patch("aios.sandbox.registry.install_egress_ca", AsyncMock()),
            patch("aios.sandbox.registry.install_packages", AsyncMock()),
            patch("aios.sandbox.registry.apply_network_lockdown", lockdown),
            patch("aios.sandbox.registry.apply_secret_egress_dnat", dnat_only),
        ):
            stack.enter_context(cm)
        return stack

    async def test_dnat_only_called_not_lockdown_when_creds_under_unrestricted(self) -> None:
        backend = FakeBackend()
        registry = SandboxRegistry(backend=backend)
        plan = self._unrestricted_plan_with_creds("sess_X")
        lockdown = AsyncMock()
        dnat_only = AsyncMock()

        with self._patches(plan, lockdown, dnat_only):
            await registry.get_or_provision("sess_X")

        # The Unrestricted path uses the DNAT-only helper, never the lockdown.
        lockdown.assert_not_awaited()
        dnat_only.assert_awaited_once()
        kwargs = dnat_only.call_args.kwargs
        assert kwargs["dnat_target"] == ("aios-worker", 49152)
        # The ``/v1`` path prefix is stripped and bare hosts de-dup'd — same
        # extraction the Limited path uses (shared ``_secret_dnat`` helper).
        assert set(kwargs["dnat_hosts"]) == {"api.secret.com", "data.secret.com"}

    async def test_runtime_threaded_to_dnat_only(self) -> None:
        backend = FakeBackend()
        registry = SandboxRegistry(backend=backend)
        plan = self._unrestricted_plan_with_creds("sess_X")
        # The plan's spec carries the sandbox runtime; assert it reaches the DNAT call.
        lockdown = AsyncMock()
        dnat_only = AsyncMock()

        with self._patches(plan, lockdown, dnat_only):
            await registry.get_or_provision("sess_X")

        assert "runtime" in dnat_only.call_args.kwargs

    async def test_no_egress_wiring_when_unrestricted_without_creds(self) -> None:
        # Unrestricted, no secret proxy / no creds → neither helper fires (the
        # historical early-return path, preserved).
        backend = FakeBackend()
        registry = SandboxRegistry(backend=backend)
        plan = _provisioning_plan("sess_X")  # Unrestricted, no creds, no proxy
        lockdown = AsyncMock()
        dnat_only = AsyncMock()

        with self._patches(plan, lockdown, dnat_only):
            await registry.get_or_provision("sess_X")

        lockdown.assert_not_awaited()
        dnat_only.assert_not_awaited()


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
            patch("aios.sandbox.registry.install_egress_ca", AsyncMock()),
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
            patch("aios.sandbox.registry.install_egress_ca", AsyncMock()),
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
            patch("aios.sandbox.registry.install_egress_ca", AsyncMock()),
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
                patch("aios.sandbox.registry.install_egress_ca", AsyncMock()),
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
            patch("aios.sandbox.registry.install_egress_ca", AsyncMock()),
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
            patch("aios.sandbox.registry.install_egress_ca", AsyncMock()),
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
            env_var_credentials=(),
        )

        with (
            patch(
                "aios.sandbox.registry.build_spec_from_session",
                AsyncMock(return_value=plan),
            ),
            patch("aios.sandbox.registry.install_egress_ca", AsyncMock()),
            patch("aios.sandbox.registry.install_packages", AsyncMock()),
            patch("aios.sandbox.registry.apply_network_lockdown", AsyncMock()),
        ):
            await registry.get_or_provision("sess_X")

        handle = registry.peek("sess_X")
        assert handle is not None
        assert handle.spec_version == 7


class TestSecretProxyLifecycle:
    """The per-session ``SecretEgressProxy`` is owned by the registry, not the
    handle (mirroring ``GitProxy``): stopped on release, mount-drift release,
    evict, and stop_all, and stored at provision time (#877).

    LIFECYCLE-ONLY and inert until #878 wires egress routing — these tests
    never drive a secret swap, only the start/store/stop bookkeeping.
    """

    async def test_release_stops_secret_proxy(self) -> None:
        backend = FakeBackend()
        registry = SandboxRegistry(backend=backend)
        _seed(registry, "sess_X")
        proxy = FakeSecretProxy()
        registry._secret_proxies["sess_X"] = cast(Any, proxy)

        await registry.release("sess_X")

        proxy.stop.assert_awaited_once()
        assert "sess_X" not in registry._secret_proxies

    async def test_release_if_mounts_changed_stops_secret_proxy(self) -> None:
        """A diverged echo set fires release(), which stops the secret proxy."""
        backend = FakeBackend()
        registry = SandboxRegistry(backend=backend)
        snapshot = frozenset([("memstore", "memstore_a", "a", "read_write")])
        _seed(registry, "sess_X", snapshot)
        proxy = FakeSecretProxy()
        registry._secret_proxies["sess_X"] = cast(Any, proxy)

        # Detach all memory stores → snapshot diverges → release fires.
        await registry.release_if_mounts_changed("sess_X", [], [], [])

        proxy.stop.assert_awaited_once()
        assert "sess_X" not in registry._secret_proxies

    async def test_evict_stops_secret_proxy(self) -> None:
        """``evict`` stops the secret proxy in the background (reusing the
        evict task-keepalive set); await the tracked task to observe it."""
        backend = FakeBackend()
        registry = SandboxRegistry(backend=backend)
        registry._handles["sess_X"] = make_handle(session_id="sess_X")
        registry._last_used["sess_X"] = 0.0
        proxy = FakeSecretProxy()
        registry._secret_proxies["sess_X"] = cast(Any, proxy)

        registry.evict("sess_X")
        assert "sess_X" not in registry._secret_proxies
        # The stop is fire-and-forget — drain the tracked tasks.
        for _ in range(3):
            await asyncio.sleep(0)
        proxy.stop.assert_awaited_once()

    async def test_stop_all_stops_all_secret_proxies(self) -> None:
        backend = FakeBackend()
        registry = SandboxRegistry(backend=backend)
        _seed(registry, "sess_X")
        _seed(registry, "sess_Y")
        proxy_x = FakeSecretProxy()
        proxy_y = FakeSecretProxy()
        registry._secret_proxies["sess_X"] = cast(Any, proxy_x)
        registry._secret_proxies["sess_Y"] = cast(Any, proxy_y)

        await registry.stop_all()

        proxy_x.stop.assert_awaited_once()
        proxy_y.stop.assert_awaited_once()
        assert registry._secret_proxies == {}

    async def test_provision_stores_secret_proxy(self) -> None:
        """A cold provision whose plan carries a non-None ``secret_proxy``
        stores it on ``registry._secret_proxies`` (mirrors
        ``test_handle_carries_spec_version_from_spec``)."""
        backend = FakeBackend()
        registry = SandboxRegistry(backend=backend)
        proxy = FakeSecretProxy()
        plan = _provisioning_plan("sess_X")
        plan = ProvisioningPlan(
            spec=plan.spec,
            env_config=plan.env_config,
            memory_echoes=plan.memory_echoes,
            github_echoes=plan.github_echoes,
            git_proxy=plan.git_proxy,
            env_var_credentials=(),
            secret_proxy=cast(Any, proxy),
        )

        with (
            patch(
                "aios.sandbox.registry.build_spec_from_session",
                AsyncMock(return_value=plan),
            ),
            patch("aios.sandbox.registry.install_egress_ca", AsyncMock()),
            patch("aios.sandbox.registry.install_packages", AsyncMock()),
            patch("aios.sandbox.registry.apply_network_lockdown", AsyncMock()),
        ):
            await registry.get_or_provision("sess_X")

        assert registry._secret_proxies["sess_X"] is cast(Any, proxy)

    async def test_provision_failure_after_build_spec_stops_both_proxies(self) -> None:
        """A raise from ``_resolve_snapshot`` — after ``build_spec_from_session``
        has started the proxies and registered the broker secret, but before the
        sandbox exists — must stop BOTH proxies exactly once, unregister the
        broker, and leave neither proxy in the registry dicts. Without the
        cleanup span covering resolution, the proxies leak (their ports + the
        broker secret outlive the worker) (#877)."""
        backend = FakeBackend()
        registry = SandboxRegistry(backend=backend)
        git_proxy = FakeSecretProxy()
        secret_proxy = FakeSecretProxy()
        plan = _provisioning_plan("sess_X")
        plan = ProvisioningPlan(
            spec=plan.spec,
            env_config=plan.env_config,
            memory_echoes=plan.memory_echoes,
            github_echoes=plan.github_echoes,
            git_proxy=cast(Any, git_proxy),
            env_var_credentials=(),
            secret_proxy=cast(Any, secret_proxy),
        )
        broker = MagicMock()

        with (
            patch(
                "aios.sandbox.registry.build_spec_from_session",
                AsyncMock(return_value=plan),
            ),
            patch.object(
                registry,
                "_resolve_snapshot",
                AsyncMock(side_effect=RuntimeError("store probe indeterminate")),
            ),
            patch("aios.harness.runtime.require_tool_broker", return_value=broker),
            pytest.raises(RuntimeError, match="store probe indeterminate"),
        ):
            await registry.get_or_provision("sess_X")

        git_proxy.stop.assert_awaited_once()
        secret_proxy.stop.assert_awaited_once()
        broker.unregister_session.assert_called_once_with("sess_X")
        assert "sess_X" not in registry._git_proxies
        assert "sess_X" not in registry._secret_proxies
        # No sandbox was ever created — teardown must not destroy a phantom.
        assert [c for c in backend.calls if c[0] in ("create", "destroy")] == []


class TestEnvVarCredentialDrift:
    """The step-time half of constraint A: an env-var credential rotation
    (bumped ``updated_at``), add, or remove diverges the mount snapshot and
    recycles the cached sandbox; an identical echo set does not (#877)."""

    _TS_V1 = datetime(2026, 6, 10, tzinfo=UTC)
    _TS_V2 = datetime(2026, 6, 11, tzinfo=UTC)

    @pytest.mark.parametrize(
        "current_echoes,description,expect_release",
        [
            ([_env_echo("vcr_01", _TS_V2)], "rotated (updated_at bumped)", True),
            (
                [_env_echo("vcr_01", _TS_V1), _env_echo("vcr_02", _TS_V1)],
                "added a cred",
                True,
            ),
            ([], "removed the cred", True),
            ([_env_echo("vcr_01", _TS_V1)], "identical", False),
        ],
    )
    async def test_env_var_drift(
        self,
        current_echoes: list[EnvVarCredentialEcho],
        description: str,
        expect_release: bool,
    ) -> None:
        backend = FakeBackend()
        registry = SandboxRegistry(backend=backend)
        snapshot = frozenset([(VAULT_CREDENTIAL, "vcr_01", self._TS_V1.isoformat())])
        handle = _seed(registry, "sess_X", snapshot)

        await registry.release_if_mounts_changed("sess_X", [], [], current_echoes)

        destroys = [c for c in backend.calls if c[0] == "destroy"]
        if expect_release:
            assert len(destroys) == 1, f"{description!r} should recycle the sandbox"
            assert destroys[0][1]["sandbox_id"] == handle.sandbox_id
            assert registry.peek("sess_X") is None
        else:
            assert destroys == [], f"{description!r} must not recycle the sandbox"
            assert registry.peek("sess_X") is not None


class TestRunSandboxLifecycle:
    """Dedicated unit assertions for the workflow-run (#988) sandbox lifecycle:
    warm-hit container reuse, ``release_run``, and the ``_release_owner`` owner-kind
    routing — paths exercised by integration tests but not previously pinned by
    dedicated unit assertions (issue #995, "test coverage breadth").
    """

    @staticmethod
    def _run_provision_patches(plan: ProvisioningPlan) -> ExitStack:
        """Patch the run cold-path's external collaborators so ``_provision_run``
        runs against the in-memory backend only. The Unrestricted, no-credential
        plan makes ``_apply_egress_rules`` a no-op, so no lockdown sidecar fires."""
        stack = ExitStack()
        for cm in (
            patch("aios.sandbox.registry.build_spec_from_run", AsyncMock(return_value=plan)),
            patch("aios.sandbox.registry.install_egress_ca", AsyncMock()),
            patch("aios.sandbox.registry.install_packages", AsyncMock()),
        ):
            stack.enter_context(cm)
        return stack

    async def test_warm_hit_reuses_container_create_count_stays_one(self) -> None:
        """Two ``get_or_provision_run`` calls in one run provision the container
        ONCE: the second is a warm hit (liveness probe only, no re-create). This
        pins the #988 e2e warm-hit-reuse strategy item — ``create_count == 1``
        across two ``sandbox()`` calls in one run."""
        backend = FakeBackend()
        registry = SandboxRegistry(backend=backend)
        plan = _provisioning_plan("wfr_X")  # Unrestricted, no creds → no egress wiring

        with self._run_provision_patches(plan):
            h1 = await registry.get_or_provision_run("wfr_X")
            h2 = await registry.get_or_provision_run("wfr_X")

        assert h1 is h2  # same cached handle
        creates = [c for c in backend.calls if c[0] == "create"]
        assert len(creates) == 1, "warm hit must NOT re-create the container"
        # The warm call took the liveness-probe path, not a second provision.
        assert any(c[0] == "is_alive" for c in backend.calls)

    async def test_warm_hit_after_dead_container_reprovisions(self) -> None:
        """If the cached container died between calls, the warm probe fails and the
        run cold-reprovisions (a fresh ``create``) — the run sandbox is snapshot-free
        so the dead handle is dropped without salvage."""
        backend = FakeBackend(next_handle_id="run_sandbox_1")
        registry = SandboxRegistry(backend=backend)
        plan = _provisioning_plan("wfr_X")

        with self._run_provision_patches(plan):
            await registry.get_or_provision_run("wfr_X")
            # Mark the cached container dead so the next warm probe fails.
            backend.dead_sandbox_ids.add("run_sandbox_1")
            backend.next_handle_id = "run_sandbox_2"
            h2 = await registry.get_or_provision_run("wfr_X")

        assert h2.sandbox_id == "run_sandbox_2"
        creates = [c for c in backend.calls if c[0] == "create"]
        assert len(creates) == 2  # cold start + reprovision after death
        # The dead container was destroyed (no snapshot — run sandboxes are scratch).
        assert any(c[0] == "destroy" for c in backend.calls)
        assert not any(c[0] == "snapshot" for c in backend.calls)

    async def test_release_run_destroys_cached_container_no_snapshot(self) -> None:
        """``release_run`` is a bare destroy + cache eviction — no snapshot/pointer
        machinery (a run sandbox has no durable rootfs)."""
        backend = FakeBackend()
        registry = SandboxRegistry(backend=backend)
        plan = _provisioning_plan("wfr_X")

        with self._run_provision_patches(plan):
            handle = await registry.get_or_provision_run("wfr_X")

        await registry.release_run("wfr_X")

        destroys = [c for c in backend.calls if c[0] == "destroy"]
        assert len(destroys) == 1
        assert destroys[0][1]["sandbox_id"] == handle.sandbox_id
        assert not any(c[0] == "snapshot" for c in backend.calls)
        assert registry.peek("wfr_X") is None

    async def test_release_run_is_noop_when_not_cached(self) -> None:
        """``release_run`` for an unknown run never touches the backend."""
        backend = FakeBackend()
        registry = SandboxRegistry(backend=backend)

        await registry.release_run("wfr_unknown")

        assert not any(c[0] == "destroy" for c in backend.calls)

    async def test_release_owner_routes_run_id_to_release_run(self) -> None:
        """A ``wfr_`` owner routes to ``release_run`` (the ephemeral-destroy path),
        NOT the session ``release`` path."""
        backend = FakeBackend()
        registry = SandboxRegistry(backend=backend)
        registry.release_run = AsyncMock()  # type: ignore[method-assign]
        registry.release = AsyncMock()  # type: ignore[method-assign]

        await registry._release_owner("wfr_01TEST")

        registry.release_run.assert_awaited_once_with("wfr_01TEST")
        registry.release.assert_not_awaited()

    async def test_release_owner_routes_session_id_to_release(self) -> None:
        """A ``sess_`` owner falls through to the session ``release`` path."""
        backend = FakeBackend()
        registry = SandboxRegistry(backend=backend)
        registry.release_run = AsyncMock()  # type: ignore[method-assign]
        registry.release = AsyncMock()  # type: ignore[method-assign]

        await registry._release_owner("sess_01TEST")

        registry.release.assert_awaited_once_with("sess_01TEST")
        registry.release_run.assert_not_awaited()


class TestSalvageBreaker:
    """The salvage circuit breaker (flatten-brick fix): consecutive salvage
    failures of one corpse suppress further retries, and the transition to
    open fires a ONE-SHOT channel-less operator wake — persisted alarmed only
    after the wake write lands, so a lost wake is retried, not dropped.
    """

    def _corpse_ref(
        self, session_id: str, sandbox_id: str = "corpse0123456789"
    ) -> ManagedSandboxRef:
        return ManagedSandboxRef(sandbox_id=sandbox_id, session_id=session_id, running=False)

    def _snapshot_count(self, backend: FakeBackend) -> int:
        return sum(1 for verb, _ in backend.calls if verb == "snapshot")

    async def test_breaker_opens_after_threshold_and_alerts_once(self) -> None:
        backend = FakeBackend()
        session_id = "sess_brk"
        ref = self._corpse_ref(session_id)
        backend.managed = [ref]
        backend.snapshot_raises = True
        registry = SandboxRegistry(backend=backend)
        alert = AsyncMock()
        registry._alert_operator = alert  # type: ignore[method-assign]
        threshold = get_settings().sandbox_salvage_breaker_threshold

        # Sub-threshold failures keep retrying the salvage (a snapshot attempt
        # each) and never alert.
        for _ in range(threshold - 1):
            with pytest.raises(SandboxBackendError, match="failed"):
                await registry._salvage_session_corpses(session_id)
        assert alert.await_count == 0
        assert self._snapshot_count(backend) == threshold - 1

        # The threshold-th failure opens the breaker and fires the one-shot.
        with pytest.raises(SandboxBackendError, match="failed"):
            await registry._salvage_session_corpses(session_id)
        assert alert.await_count == 1
        assert self._snapshot_count(backend) == threshold

        # Once open, provisioning is suppressed: NO further snapshot attempt
        # and NO re-alert.
        with pytest.raises(SandboxBackendError, match="breaker open"):
            await registry._salvage_session_corpses(session_id)
        assert alert.await_count == 1
        assert self._snapshot_count(backend) == threshold  # unchanged

    async def test_breaker_half_open_recovers_after_cooldown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = FakeBackend()
        session_id = "sess_half_open"
        ref = self._corpse_ref(session_id)
        backend.managed = [ref]
        backend.snapshot_raises = True
        registry = SandboxRegistry(backend=backend)
        registry._alert_operator = AsyncMock()  # type: ignore[method-assign]
        threshold = get_settings().sandbox_salvage_breaker_threshold
        now = 1000.0
        monkeypatch.setattr("aios.sandbox.registry.time.monotonic", lambda: now)

        for _ in range(threshold):
            with pytest.raises(SandboxBackendError):
                await registry._salvage_session_corpses(session_id)
        with pytest.raises(SandboxBackendError, match="breaker open"):
            await registry._salvage_session_corpses(session_id)

        now += 300.0
        backend.snapshot_raises = False
        await registry._salvage_session_corpses(session_id)
        assert ref.sandbox_id not in registry._salvage_failures
        assert ("force_remove", {"sandbox_id": ref.sandbox_id}) in backend.calls

    async def test_breaker_alert_retries_when_wake_write_fails(self) -> None:
        """A failed wake write must leave the one-shot un-alarmed so the next
        provision retries it — the ``alarmed``-after-success ordering."""
        backend = FakeBackend()
        session_id = "sess_retry"
        backend.managed = [self._corpse_ref(session_id)]
        backend.snapshot_raises = True
        registry = SandboxRegistry(backend=backend)
        threshold = get_settings().sandbox_salvage_breaker_threshold

        attempts: list[str] = []

        async def _flaky_alert(session_id: str, content: str, *, cause: str) -> None:
            attempts.append(cause)
            if len(attempts) == 1:
                raise RuntimeError("wake write failed")

        registry._alert_operator = _flaky_alert  # type: ignore[method-assign]

        # Drive to the transition; the alert is attempted once and fails.
        for _ in range(threshold):
            with pytest.raises(SandboxBackendError):
                await registry._salvage_session_corpses(session_id)
        assert len(attempts) == 1

        # Breaker now open: the suppressed pass retries the lost wake, which
        # lands this time.
        with pytest.raises(SandboxBackendError, match="breaker open"):
            await registry._salvage_session_corpses(session_id)
        assert len(attempts) == 2

        # Alarmed now persisted → no further re-fire.
        with pytest.raises(SandboxBackendError, match="breaker open"):
            await registry._salvage_session_corpses(session_id)
        assert len(attempts) == 2

    async def test_successful_salvage_resets_failure_counter(self) -> None:
        backend = FakeBackend()
        session_id = "sess_reset"
        ref = self._corpse_ref(session_id)
        backend.managed = [ref]
        backend.snapshot_raises = True
        registry = SandboxRegistry(backend=backend)
        registry._alert_operator = AsyncMock()  # type: ignore[method-assign]

        with pytest.raises(SandboxBackendError):
            await registry._salvage_session_corpses(session_id)
        # value = (owning_session_id, failures, alarmed)
        assert registry._salvage_failures[ref.sandbox_id] == (session_id, 1, False)

        # A subsequent SUCCESSFUL salvage clears the counter and removes the corpse.
        backend.snapshot_raises = False
        await registry._salvage_session_corpses(session_id)
        assert ref.sandbox_id not in registry._salvage_failures
        assert ("force_remove", {"sandbox_id": ref.sandbox_id}) in backend.calls

    async def test_counter_cleared_when_corpse_disappears(self) -> None:
        backend = FakeBackend()
        session_id = "sess_gone"
        ref = self._corpse_ref(session_id)
        backend.managed = [ref]
        backend.snapshot_raises = True
        registry = SandboxRegistry(backend=backend)
        registry._alert_operator = AsyncMock()  # type: ignore[method-assign]

        with pytest.raises(SandboxBackendError):
            await registry._salvage_session_corpses(session_id)
        assert ref.sandbox_id in registry._salvage_failures

        # Corpse removed out of band → the stale counter is GC'd on the next pass.
        backend.managed = []
        await registry._salvage_session_corpses(session_id)
        assert ref.sandbox_id not in registry._salvage_failures

    async def test_disk_deferral_counts_toward_breaker(self) -> None:
        """A flatten deferred for insufficient disk surfaces as a snapshot
        ``SandboxBackendError`` → ``removable=False`` → increments the breaker
        exactly like any other salvage failure (no silent bypass)."""

        class _DiskDeferBackend(FakeBackend):
            async def snapshot(
                self,
                sandbox_id: str,
                tag: str,
                *,
                empty_floor_bytes: int,
                flatten_if_unique_bytes_over: int | None,
            ) -> Any:
                self.calls.append(("snapshot", {"sandbox_id": sandbox_id, "tag": tag}))
                raise SandboxBackendError("flatten deferred: 1000 free bytes, 999999999 required")

        backend = _DiskDeferBackend()
        session_id = "sess_disk"
        backend.managed = [self._corpse_ref(session_id)]
        registry = SandboxRegistry(backend=backend)
        alert = AsyncMock()
        registry._alert_operator = alert  # type: ignore[method-assign]
        threshold = get_settings().sandbox_salvage_breaker_threshold

        for _ in range(threshold):
            with pytest.raises(SandboxBackendError):
                await registry._salvage_session_corpses(session_id)
        assert alert.await_count == 1
        assert self._snapshot_count(backend) == threshold
        with pytest.raises(SandboxBackendError, match="breaker open"):
            await registry._salvage_session_corpses(session_id)

    async def test_alert_operator_uses_tell_existing_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The operator alert must go through the platform's channel-less
        ``Tell(ExistingSession)`` writer (user message + deferred wake), NOT
        a non-stimulus lifecycle event that only surfaces on some future wake."""
        backend = FakeBackend()
        registry = SandboxRegistry(backend=backend)
        tell = AsyncMock()
        monkeypatch.setattr("aios.services.sessions.tell_existing_session", tell)
        monkeypatch.setattr(
            "aios.services.sessions.load_session_account_id",
            AsyncMock(return_value="acct_x"),
        )

        await registry._alert_operator(
            "sess_1", "salvage breaker OPEN", cause="sandbox.salvage_breaker_open"
        )

        tell.assert_awaited_once()
        call = tell.await_args
        assert call is not None
        assert call.args[1] == "sess_1"
        assert call.kwargs["content"] == "salvage breaker OPEN"
        assert call.kwargs["cause"] == "sandbox.salvage_breaker_open"
        assert call.kwargs["account_id"] == "acct_x"

    async def test_pass_for_one_session_does_not_reset_another_sessions_breaker(self) -> None:
        """A salvage pass for session A must NOT clear session B's open-breaker
        state. ``list_managed`` is session-filtered, so the stale-counter GC must
        be scoped to the CURRENT session's entries; otherwise A's pass GCs B's
        entry (B's corpse isn't in A's present set), B retries the expensive
        flatten and re-emits its 'one-shot' alert — the breaker is defeated in
        normal multi-session operation."""
        backend = FakeBackend()
        corpse_a = self._corpse_ref("sess_A", sandbox_id="corpseAAAAAAAAAA")
        corpse_b = self._corpse_ref("sess_B", sandbox_id="corpseBBBBBBBBBB")
        backend.managed = [corpse_a, corpse_b]
        backend.snapshot_raises = True
        registry = SandboxRegistry(backend=backend)
        registry._alert_operator = AsyncMock()  # type: ignore[method-assign]
        threshold = get_settings().sandbox_salvage_breaker_threshold

        # Drive session B's corpse to an OPEN breaker.
        for _ in range(threshold):
            with pytest.raises(SandboxBackendError):
                await registry._salvage_session_corpses("sess_B")
        assert registry._salvage_failures[corpse_b.sandbox_id] == ("sess_B", threshold, True)

        # A salvage pass for session A (its own corpse also fails) must leave
        # B's entry fully intact — this is the fail-before/pass-after.
        with pytest.raises(SandboxBackendError, match="failed"):
            await registry._salvage_session_corpses("sess_A")
        assert corpse_b.sandbox_id in registry._salvage_failures
        assert registry._salvage_failures[corpse_b.sandbox_id] == ("sess_B", threshold, True)

        # B's breaker is still OPEN: a fresh B pass is suppressed (no new
        # snapshot attempt, no re-alert) rather than retrying the flatten.
        snaps_before = self._snapshot_count(backend)
        with pytest.raises(SandboxBackendError, match="breaker open"):
            await registry._salvage_session_corpses("sess_B")
        assert self._snapshot_count(backend) == snaps_before

async def test_vault_rotation_snapshots_recreates_spec_and_resumes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Archive/recreate notification recycles FS and rebuilds credential state."""
    from aios.sandbox.backends.base import BASE_IMAGE_LABEL_KEY
    from aios.sandbox.spec import _assemble_plan, snapshot_tag

    class CapturingBackend(FakeBackend):
        specs: list[SandboxSpec]

        def __init__(self) -> None:
            super().__init__()
            self.specs = []

        async def create(self, spec: SandboxSpec) -> SandboxHandle:
            self.specs.append(spec)
            return await super().create(spec)

    now = datetime.now(UTC)
    current = [
        ResolvedEnvVarCredential(
            credential_id="vc_archived",
            secret_name="GITHUB_TOKEN",
            secret_value="token-a",
            allowed_hosts=("github.com",),
            updated_at=now,
            placeholder="PLACEHOLDER_A",
        )
    ]
    resume_ref: list[str | None] = [None]
    backend = CapturingBackend()
    registry = SandboxRegistry(backend)

    def build_plan() -> ProvisioningPlan:
        return _assemble_plan(
            session_id="sess_X",
            instance_id="inst_TEST",
            image="aios-sandbox:test",
            workspace_path=Path("/tmp/w"),
            env_config=None,
            session_env={},
            memory_echoes=[],
            github_echoes=[],
            git_proxy=None,
            tool_broker_url="http://worker:1",
            tool_broker_secret="broker-secret",
            snapshot_ref=resume_ref[0],
            env_var_credentials=tuple(current),
            secret_proxy=cast(Any, FakeSecretProxy()),
        )

    async def build(_session_id: str) -> ProvisioningPlan:
        return build_plan()

    settings = MagicMock()
    settings.instance_id = "inst_TEST"
    settings.sandbox_snapshot_empty_floor_bytes = 0
    settings.sandbox_snapshot_budget_bytes = None
    settings.sandbox_cpu_quota = None
    settings.sandbox_memory_bytes = None
    settings.sandbox_pids_limit = None
    settings.sandbox_seccomp_profile = "/tmp/seccomp.json"
    settings.sandbox_runtime = None
    settings.tool_broker_socket_path = None
    tag = snapshot_tag("inst_TEST", "sess_X")

    with (
        patch("aios.sandbox.registry.build_spec_from_session", side_effect=build),
        patch("aios.sandbox.registry.get_settings", return_value=settings),
        patch("aios.sandbox.spec.get_settings", return_value=settings),
        patch("aios.sandbox.volumes.ensure_session_attachments_dir", return_value=Path("/tmp/a")),
        patch("aios.sandbox.volumes.ensure_session_uploads_dir", return_value=Path("/tmp/u")),
        patch("aios.sandbox.registry.install_egress_ca", AsyncMock()),
        patch("aios.sandbox.registry.install_packages", AsyncMock()),
        patch.object(registry, "_apply_egress_rules", AsyncMock()),
        patch("aios.sandbox.registry.queries.list_session_ids_for_vault", AsyncMock(return_value=["sess_X"])),
    ):
        await registry.get_or_provision("sess_X")
        assert backend.specs[-1].environment["GITHUB_TOKEN"] == "PLACEHOLDER_A"

        # Actual archive+recreate shape: the row/id (and therefore placeholder)
        # changes, then the vault notification drives the registry recycle.
        current[:] = [dataclasses.replace(current[0], credential_id="vc_recreated", secret_value="token-a2", placeholder="PLACEHOLDER_A2")]
        await registry.recycle_sessions_for_vault("vlt_X")
        assert any(call[0] == "snapshot" for call in backend.calls)

        resume_ref[0] = tag
        backend.image_labels_by_ref[tag] = {BASE_IMAGE_LABEL_KEY: "aios-sandbox:test"}
        await registry.get_or_provision("sess_X")

    resumed = backend.specs[-1]
    assert resumed.snapshot_image == tag
    assert resumed.environment["GITHUB_TOKEN"] == "PLACEHOLDER_A2"
    assert "PLACEHOLDER_A" not in resumed.environment.values()
