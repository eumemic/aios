"""Pointer-discipline coverage for the registry's snapshot lifecycle (§5.1-§5.4).

Drives ``release`` / salvage-reconcile / reset against a ``FakeBackend`` with
the snapshot-pointer queries patched, so we can assert the ordering invariant
(pointer written after snapshot success and BEFORE rm; never on failure), the
first-commit crash heal, and the reset clear-plus-event — all without Docker.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from aios.harness import runtime
from aios.sandbox.registry import SandboxRegistry
from tests.helpers.sandbox import FakeBackend, FakePool, make_handle


@pytest.fixture
def harness(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[SandboxRegistry, FakeBackend, list[Any]]]:
    """A registry over a FakeBackend with the pointer queries patched.

    ``timeline`` records pointer-set / pointer-clear / destroy / fs-event in the
    order they happen so the after-success-before-rm invariant is assertable.
    """
    backend = FakeBackend()
    registry = SandboxRegistry(backend=backend)
    timeline: list[Any] = []

    async def _set(
        _conn: Any, session_id: str, *, ref: str, host: str, snapshot_bytes: int
    ) -> None:
        timeline.append(("pointer_set", ref, snapshot_bytes))

    async def _clear(_conn: Any, session_id: str) -> None:
        timeline.append(("pointer_clear", session_id))

    monkeypatch.setattr("aios.sandbox.registry.queries.unscoped_set_session_snapshot", _set)
    monkeypatch.setattr("aios.sandbox.registry.queries.unscoped_clear_session_snapshot", _clear)
    monkeypatch.setattr(
        "aios.sandbox.registry.queries.unscoped_get_session_snapshot_bytes",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        "aios.services.sessions.load_session_account_id", AsyncMock(return_value="acct")
    )

    async def _append(
        pool: Any, session_id: str, kind: str, data: dict[str, Any], **k: Any
    ) -> None:
        timeline.append(("fs_event", data.get("event"), data.get("reason")))

    monkeypatch.setattr("aios.services.sessions.append_event", _append)

    # Record destroy in the same timeline so ordering vs the pointer is visible.
    orig_destroy = backend.destroy

    async def _destroy(handle: Any) -> None:
        timeline.append(("destroy", handle.sandbox_id))
        await orig_destroy(handle)

    backend.destroy = _destroy  # type: ignore[method-assign]

    prev_pool = runtime.pool
    runtime.pool = cast(Any, FakePool())
    try:
        yield registry, backend, timeline
    finally:
        runtime.pool = prev_pool


@pytest.mark.asyncio
async def test_release_writes_pointer_before_rm(
    harness: tuple[SandboxRegistry, FakeBackend, list[Any]],
) -> None:
    registry, _backend, timeline = harness
    registry._handles["sess_x"] = make_handle(session_id="sess_x")
    registry._last_used["sess_x"] = 0.0

    await registry.release("sess_x")

    kinds = [t[0] for t in timeline]
    assert "pointer_set" in kinds, "release must write the snapshot pointer"
    assert "destroy" in kinds, "release must remove the container after snapshotting"
    assert kinds.index("pointer_set") < kinds.index("destroy"), (
        "the pointer must be written AFTER snapshot success and BEFORE rm"
    )


@pytest.mark.asyncio
async def test_snapshot_failure_retains_corpse_no_pointer(
    harness: tuple[SandboxRegistry, FakeBackend, list[Any]],
) -> None:
    registry, backend, timeline = harness
    backend.snapshot_raises = True
    registry._handles["sess_x"] = make_handle(session_id="sess_x")
    registry._last_used["sess_x"] = 0.0

    await registry.release("sess_x")

    kinds = [t[0] for t in timeline]
    assert "pointer_set" not in kinds, "a failed snapshot must not write a pointer"
    assert "destroy" not in kinds, "a failed snapshot must retain the corpse (no rm)"


@pytest.mark.asyncio
async def test_pointer_write_failure_treated_as_snapshot_failure(
    harness: tuple[SandboxRegistry, FakeBackend, list[Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, _backend, timeline = harness

    async def _boom(*a: Any, **k: Any) -> None:
        raise RuntimeError("db down")

    monkeypatch.setattr("aios.sandbox.registry.queries.unscoped_set_session_snapshot", _boom)
    registry._handles["sess_x"] = make_handle(session_id="sess_x")
    registry._last_used["sess_x"] = 0.0

    await registry.release("sess_x")

    # A failed pointer write is treated identically to a failed snapshot verb:
    # corpse retained, no rm.
    assert ("destroy", "abc123def456abc123def456") not in timeline


@pytest.mark.asyncio
async def test_salvage_preamble_heals_null_pointer(
    harness: tuple[SandboxRegistry, FakeBackend, list[Any]],
) -> None:
    """First-commit crash heal (§5.3): the canonical tag exists locally but the
    pointer is NULL → the preamble reconcile sets it before resolution runs."""
    registry, backend, timeline = harness
    from aios.config import get_settings
    from aios.sandbox.spec import snapshot_tag

    tag = snapshot_tag(get_settings().instance_id, "sess_x")
    backend.image_labels_by_ref[tag] = {}  # tag exists locally
    backend.image_sizes_by_ref[tag] = 500_000

    await registry._reconcile_pointer_from_local("sess_x")

    pointer_sets = [t for t in timeline if t[0] == "pointer_set"]
    assert pointer_sets, "the heal must set the pointer when the canonical tag exists"
    assert pointer_sets[0][1] == tag


@pytest.mark.asyncio
async def test_reset_clears_pointer_and_events(
    harness: tuple[SandboxRegistry, FakeBackend, list[Any]],
) -> None:
    registry, _backend, timeline = harness

    await registry._reset_snapshot("sess_x", reason="snapshot_missing")

    assert ("pointer_clear", "sess_x") in timeline
    assert ("fs_event", "sandbox_fs_reset", "snapshot_missing") in timeline


@pytest.mark.asyncio
async def test_resolve_base_drift_removes_and_resets(
    harness: tuple[SandboxRegistry, FakeBackend, list[Any]],
) -> None:
    """Base-image drift (§5.3): the snapshot's recorded base != the current env
    image → discard (store.remove + pointer clear + event), cold start."""
    registry, backend, timeline = harness
    from aios.sandbox.backends.base import Mount, SandboxSpec, Unrestricted

    ref = "aios-sbx-default-sess_x:latest"
    backend.image_labels_by_ref[ref] = {"aios.base_image": "OLD-IMAGE"}
    spec = SandboxSpec(
        session_id="sess_x",
        instance_id="default",
        workspace=Mount(host_path=cast(Any, "/tmp/w"), sandbox_path="/workspace"),
        extra_mounts=(),
        environment={},
        labels={},
        network_policy=Unrestricted(),
        host_gateway_alias=None,
        image="NEW-IMAGE",  # differs from the snapshot's base_image
        snapshot_image=ref,
    )

    resolved = await registry._resolve_snapshot("sess_x", spec)

    assert resolved.snapshot_image is None, "drift must clear the resume source (cold start)"
    assert ref in backend.removed_image_refs, "the stale snapshot artifact must actually be removed"
    assert ("pointer_clear", "sess_x") in timeline
    assert ("fs_event", "sandbox_fs_reset", "environment_image_changed") in timeline


@pytest.mark.asyncio
async def test_resolve_snapshot_missing_resets(
    harness: tuple[SandboxRegistry, FakeBackend, list[Any]],
) -> None:
    """Pointer set + store verified-not-found → snapshot_missing reset + cold start."""
    registry, _backend, timeline = harness
    from aios.sandbox.backends.base import Mount, SandboxSpec, Unrestricted

    ref = "aios-sbx-default-sess_x:latest"  # not in image_labels_by_ref ⇒ absent
    spec = SandboxSpec(
        session_id="sess_x",
        instance_id="default",
        workspace=Mount(host_path=cast(Any, "/tmp/w"), sandbox_path="/workspace"),
        extra_mounts=(),
        environment={},
        labels={},
        network_policy=Unrestricted(),
        host_gateway_alias=None,
        image="NEW-IMAGE",
        snapshot_image=ref,
    )

    resolved = await registry._resolve_snapshot("sess_x", spec)

    assert resolved.snapshot_image is None
    assert ("fs_event", "sandbox_fs_reset", "snapshot_missing") in timeline
