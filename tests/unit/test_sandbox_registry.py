"""Unit tests for ``SandboxRegistry.release_if_mounts_changed`` (issue #198).

Exercises the worker-side mount-drift detector that recycles a cached
container when the session's attached memory stores have changed since
the container was provisioned. The check fires once per step at the top
of ``loop._run_session_step_body``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from aios.models.memory_stores import Access, MemoryStoreResourceEcho
from aios.sandbox.container import ContainerHandle
from aios.sandbox.registry import SandboxRegistry


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


def _handle(
    session_id: str, snapshot: frozenset[tuple[str, str, str]] = frozenset()
) -> ContainerHandle:
    return ContainerHandle(
        session_id=session_id,
        container_id="abc123def456abc123def456",
        workspace_path=Path("/tmp/w"),
        mount_snapshot=snapshot,
    )


class TestReleaseIfMountsChanged:
    async def test_no_cached_handle_is_noop(self) -> None:
        registry = SandboxRegistry()
        provisioner_release = AsyncMock()
        with patch("aios.sandbox.registry.provisioner_release", provisioner_release):
            await registry.release_if_mounts_changed("sess_NONE", [_echo("memstore_a", "a")], [])
        provisioner_release.assert_not_awaited()

    async def test_identical_snapshot_does_not_release(self) -> None:
        registry = SandboxRegistry()
        snapshot = frozenset([("memstore", "memstore_a", "a", "read_write")])
        registry._handles["sess_X"] = _handle("sess_X", snapshot)
        registry._last_used["sess_X"] = 0.0

        provisioner_release = AsyncMock()
        with patch("aios.sandbox.registry.provisioner_release", provisioner_release):
            await registry.release_if_mounts_changed("sess_X", [_echo("memstore_a", "a")], [])

        provisioner_release.assert_not_awaited()
        assert registry.peek("sess_X") is not None

    async def test_reorder_does_not_release(self) -> None:
        registry = SandboxRegistry()
        snapshot = frozenset(
            [
                ("memstore", "memstore_a", "a", "read_write"),
                ("memstore", "memstore_b", "b", "read_only"),
            ]
        )
        registry._handles["sess_X"] = _handle("sess_X", snapshot)
        registry._last_used["sess_X"] = 0.0

        provisioner_release = AsyncMock()
        with patch("aios.sandbox.registry.provisioner_release", provisioner_release):
            await registry.release_if_mounts_changed(
                "sess_X",
                [_echo("memstore_b", "b", "read_only"), _echo("memstore_a", "a")],
                [],
            )

        provisioner_release.assert_not_awaited()
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
        registry = SandboxRegistry()
        snapshot = frozenset([("memstore", "memstore_a", "a", "read_write")])
        handle = _handle("sess_X", snapshot)
        registry._handles["sess_X"] = handle
        registry._last_used["sess_X"] = 0.0

        provisioner_release = AsyncMock()
        with patch("aios.sandbox.registry.provisioner_release", provisioner_release):
            await registry.release_if_mounts_changed("sess_X", current_echoes, [])

        provisioner_release.assert_awaited_once_with(handle)
        assert registry.peek("sess_X") is None
        assert "sess_X" not in registry._last_used

    async def test_release_serialized_against_get_or_provision(self) -> None:
        """The per-session lock must serialize release_if_mounts_changed against
        a concurrent get_or_provision so the registry never returns a handle
        that's about to be torn down."""
        import asyncio

        registry = SandboxRegistry()
        old_snapshot = frozenset([("memstore", "memstore_a", "a", "read_write")])
        old_handle = _handle("sess_X", old_snapshot)
        registry._handles["sess_X"] = old_handle
        registry._last_used["sess_X"] = 0.0

        provision_started = asyncio.Event()
        let_provision_continue = asyncio.Event()

        async def slow_provision(_session_id: str) -> ContainerHandle:
            provision_started.set()
            await let_provision_continue.wait()
            return _handle("sess_X", frozenset([("memstore", "memstore_b", "b", "read_write")]))

        # Force a cache miss so get_or_provision contends for the lock.
        registry._handles.pop("sess_X")

        provisioner_release = AsyncMock()
        with (
            patch("aios.sandbox.registry.provision_for_session", slow_provision),
            patch("aios.sandbox.registry.provisioner_release", provisioner_release),
        ):
            provision_task = asyncio.create_task(registry.get_or_provision("sess_X"))
            await provision_started.wait()

            # Re-seed the cache so release has something to release once it
            # gets the lock — this models a real cycle: provision finishes,
            # then drift is detected, then release fires.
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
        provisioner_release.assert_not_awaited()
