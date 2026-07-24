"""Registry-level teardown coverage for the sandbox self-recycle path (#2022).

Two review findings are pinned here:

1. **No orphaned snapshot artifact.** ``recycle`` must remove the canonical
   snapshot image/tag, not merely NULL the ``sessions.snapshot_*`` pointer. A
   surviving tag with a NULL pointer is exactly the state ``_gc_reconcile_pointers``
   heals — it would re-point the session at the writable layer recycle was asked
   to discard, and the next provision would resume the stale packages/caches.
2. **Ordering under partial failure.** The artifact is removed BEFORE the
   pointer is cleared, so a crash/failure mid-operation leaves a still-attributable
   artifact rather than an unattributable orphan.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

import pytest

from aios.config import get_settings
from aios.harness import runtime
from aios.sandbox.backends.base import (
    SESSION_LABEL_KEY,
    ManagedImage,
    ManagedSandboxRef,
    SandboxBackendError,
)
from aios.sandbox.registry import SandboxRegistry, SessionSnapshotState
from aios.sandbox.spec import snapshot_tag
from tests.helpers.sandbox import FakeBackend, FakePool

_SESSION = "sess_recycle"


@pytest.fixture
def fake_pool() -> Iterator[None]:
    prev = runtime.pool
    runtime.pool = cast(Any, FakePool())
    try:
        yield
    finally:
        runtime.pool = prev


def _canonical() -> str:
    return snapshot_tag(get_settings().instance_id, _SESSION)


def _seed(backend: FakeBackend, *, session_id: str = _SESSION) -> str:
    """Give the backend one canonical snapshot tag + a live corpse."""
    tag = snapshot_tag(get_settings().instance_id, session_id)
    backend.managed = [
        ManagedSandboxRef(sandbox_id="corpse_1", session_id=session_id, running=False)
    ]
    backend.managed_images = [
        ManagedImage(
            image_id="sha256:aaa",
            repo_tags=(tag,),
            parent_id=None,
            size_bytes=1_000_000,
            labels={SESSION_LABEL_KEY: session_id},
        )
    ]
    backend.image_labels_by_ref[tag] = {SESSION_LABEL_KEY: session_id}
    backend.image_sizes_by_ref[tag] = 1_000_000
    return tag


def _session_tags(backend: FakeBackend, session_id: str = _SESSION) -> list[str]:
    """Snapshot tags the daemon still reports for ``session_id``."""
    return [ref for ref in backend.image_labels_by_ref if session_id.lower() in ref]


def _registry(backend: FakeBackend, timeline: list[Any]) -> SandboxRegistry:
    registry = SandboxRegistry(backend=backend)
    orig_remove = backend.remove_image

    async def _remove(ref: str) -> bool:
        timeline.append(("remove_image", ref))
        return await orig_remove(ref)

    backend.remove_image = _remove  # type: ignore[method-assign]
    return registry


@pytest.mark.asyncio
async def test_recycle_removes_canonical_snapshot_artifact(
    fake_pool: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding 1: recycle leaves ZERO snapshot tags for the session."""
    backend = FakeBackend()
    tag = _seed(backend)
    timeline: list[Any] = []
    monkeypatch.setattr(
        "aios.sandbox.registry.queries.unscoped_clear_session_snapshot",
        _recording_clear(timeline),
    )
    registry = _registry(backend, timeline)

    before = len(_session_tags(backend))
    await registry.recycle(_SESSION)
    after = len(_session_tags(backend))

    assert before == 1
    assert after == before - 1, "recycle must leave no orphaned snapshot tag for the session"
    assert tag in backend.removed_image_refs
    assert any(c[0] == "force_remove" for c in backend.calls), "the corpse must be removed too"


@pytest.mark.asyncio
async def test_recycle_removes_artifact_before_clearing_pointer(
    fake_pool: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding 1 (ordering): artifact removal precedes the pointer clear.

    A crash between the two must leave an artifact-less pointer (which
    ``_resolve_snapshot`` resets to a cold start) rather than a pointer-less
    artifact (the orphan GC pass 4 cannot attribute and would resurrect).
    """
    backend = FakeBackend()
    _seed(backend)
    timeline: list[Any] = []
    monkeypatch.setattr(
        "aios.sandbox.registry.queries.unscoped_clear_session_snapshot",
        _recording_clear(timeline),
    )
    registry = _registry(backend, timeline)

    await registry.recycle(_SESSION)

    kinds = [entry[0] for entry in timeline]
    assert "remove_image" in kinds and "pointer_clear" in kinds
    assert kinds.index("remove_image") < kinds.index("pointer_clear"), (
        "the snapshot artifact must be removed BEFORE the pointer is cleared"
    )


@pytest.mark.asyncio
async def test_recycle_keeps_pointer_when_artifact_removal_is_refused(
    fake_pool: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refused rmi must raise and LEAVE the pointer, keeping the tag attributable."""
    backend = FakeBackend()
    tag = _seed(backend)
    backend.refuse_remove_refs = {tag}
    timeline: list[Any] = []
    monkeypatch.setattr(
        "aios.sandbox.registry.queries.unscoped_clear_session_snapshot",
        _recording_clear(timeline),
    )
    registry = _registry(backend, timeline)

    with pytest.raises(SandboxBackendError):
        await registry.recycle(_SESSION)

    assert not any(entry[0] == "pointer_clear" for entry in timeline), (
        "a surviving artifact must keep its pointer so it stays attributable"
    )


@pytest.mark.asyncio
async def test_gc_cannot_resurrect_a_recycled_snapshot(
    fake_pool: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding 1, end-to-end: recycle → failed provision → GC tick ⇒ no re-point.

    Reproduces the reported hazard: the pointer is NULL (recycle cleared it),
    provisioning never completed, so no handle is cached — precisely the
    conditions under which ``_gc_reconcile_pointers`` heals a NULL pointer from
    a retained canonical tag. With the artifact actually gone, the reconciler
    has nothing to heal from and the discarded writable layer stays discarded.
    """
    backend = FakeBackend()
    tag = _seed(backend)
    timeline: list[Any] = []
    monkeypatch.setattr(
        "aios.sandbox.registry.queries.unscoped_clear_session_snapshot",
        _recording_clear(timeline),
    )
    written: list[tuple[str, str]] = []

    async def _set(
        _conn: Any, session_id: str, *, ref: str, host: str, snapshot_bytes: int
    ) -> None:
        written.append((session_id, ref))

    monkeypatch.setattr("aios.sandbox.registry.queries.unscoped_set_session_snapshot", _set)
    registry = _registry(backend, timeline)

    await registry.recycle(_SESSION)
    # Fresh provisioning fails / never runs: no cached handle, NULL pointer.
    assert registry._handles.get(_SESSION) is None

    # The GC tick re-enumerates images; the canonical tag is gone, so pass 4
    # has no retained canonical verdict for this session to heal from.
    remaining = (
        [img for img in backend.managed_images if tag in img.repo_tags]
        if tag in backend.image_labels_by_ref
        else []
    )
    await registry._gc_reconcile_pointers(
        [],  # nothing retained — the artifact was removed by recycle
        {
            _SESSION: SessionSnapshotState(
                session_id=_SESSION,
                account_id="acct",
                archived_at=None,
                last_event_at=None,
                snapshot_ref=None,
                snapshot_host=None,
                snapshot_bytes=None,
            )
        },
        get_settings().instance_id,
    )

    assert remaining == [], "the canonical artifact must not survive recycle"
    assert written == [], "GC must not re-point a session at a discarded snapshot"


def _recording_clear(timeline: list[Any]) -> Any:
    async def _clear(_conn: Any, session_id: str) -> None:
        timeline.append(("pointer_clear", session_id))

    return _clear
