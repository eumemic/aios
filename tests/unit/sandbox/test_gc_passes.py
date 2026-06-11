"""The GC passes' under-lock dormancy RE-VERIFY (durable session sandboxes, §5.5).

The retain rule is decided from a ``states`` snapshot loaded once per tick, but
the design requires the condition to be **re-verified under the per-session
lock**: a session that wakes (or crosses back under the TTL) between the load
and a drop decision must keep its filesystem. These tests drive the impure
corpse/image passes with a stale-dormant tick-start state and a fresh-read that
says active, and assert the woke session is salvaged / retained — not dropped.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from aios.config import get_settings
from aios.harness import runtime
from aios.sandbox.backends.base import ManagedImage, ManagedSandboxRef
from aios.sandbox.registry import GcImageVerdict, SandboxRegistry, SessionSnapshotState
from aios.sandbox.spec import snapshot_tag
from tests.helpers.sandbox import FakeBackend, FakePool

_NOW = datetime(2026, 6, 10, tzinfo=UTC)
_TTL = 30 * 24 * 3600


@pytest.fixture
def fake_pool() -> Iterator[None]:
    prev = runtime.pool
    runtime.pool = cast(Any, FakePool())
    try:
        yield
    finally:
        runtime.pool = prev


def _state(session_id: str, *, dormant: bool) -> SessionSnapshotState:
    return SessionSnapshotState(
        session_id=session_id,
        account_id="acct",
        archived=False,
        last_event_at=_NOW - timedelta(days=40 if dormant else 1),
        snapshot_ref=snapshot_tag(get_settings().instance_id, session_id),
        snapshot_host=get_settings().instance_id,
        snapshot_bytes=1_000_000,
    )


def _row(session_id: str, *, dormant: bool) -> dict[str, Any]:
    """A ``gc_snapshot_session_states`` row shape (the fresh re-read)."""
    return {
        "id": session_id,
        "account_id": "acct",
        "archived_at": None,
        "last_event_at": _NOW - timedelta(days=40 if dormant else 1),
        "snapshot_ref": snapshot_tag(get_settings().instance_id, session_id),
        "snapshot_host": get_settings().instance_id,
        "snapshot_bytes": 1_000_000,
    }


def _patch_fresh_read(
    monkeypatch: pytest.MonkeyPatch, *, dormant: bool, deleted: bool = False
) -> None:
    rows = [] if deleted else [_row("sess_x", dormant=dormant)]
    monkeypatch.setattr(
        "aios.sandbox.registry.queries.gc_snapshot_session_states",
        AsyncMock(return_value=rows),
    )


@pytest.mark.asyncio
async def test_corpse_pass_salvages_session_that_woke_since_tick_start(
    fake_pool: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tick-start state says dormant, but a fresh read under the lock says active
    (the session woke) → the corpse is SALVAGED (committed), not dropped."""
    backend = FakeBackend()
    registry = SandboxRegistry(backend=backend)
    _patch_fresh_read(monkeypatch, dormant=False)  # woke: fresh read is active
    container = ManagedSandboxRef(sandbox_id="cid", session_id="sess_x", running=False)

    await registry._gc_corpse_pass(
        [container],
        {"sess_x": _state("sess_x", dormant=True)},  # stale tick-start snapshot
        _NOW,
        get_settings(),
        get_settings().instance_id,
    )

    assert any(c[0] == "snapshot" for c in backend.calls), (
        "a session that woke since the tick-start load must have its corpse salvaged"
    )


@pytest.mark.asyncio
async def test_corpse_pass_drops_still_dormant_session_without_commit(
    fake_pool: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tick-start dormant AND fresh read still dormant → drop without a commit."""
    backend = FakeBackend()
    registry = SandboxRegistry(backend=backend)
    _patch_fresh_read(monkeypatch, dormant=True)
    container = ManagedSandboxRef(sandbox_id="cid", session_id="sess_x", running=False)

    await registry._gc_corpse_pass(
        [container],
        {"sess_x": _state("sess_x", dormant=True)},
        _NOW,
        get_settings(),
        get_settings().instance_id,
    )

    assert not any(c[0] == "snapshot" for c in backend.calls), (
        "a still-dormant corpse must be dropped without paying a commit"
    )
    assert any(c[0] == "force_remove" for c in backend.calls)


@pytest.mark.asyncio
async def test_corpse_pass_drops_deleted_session(
    fake_pool: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fresh read returns no row (session deleted) → drop without a commit."""
    backend = FakeBackend()
    registry = SandboxRegistry(backend=backend)
    _patch_fresh_read(monkeypatch, dormant=True, deleted=True)
    container = ManagedSandboxRef(sandbox_id="cid", session_id="sess_x", running=False)

    await registry._gc_corpse_pass(
        [container],
        {},  # absent at tick start too
        _NOW,
        get_settings(),
        get_settings().instance_id,
    )

    assert not any(c[0] == "snapshot" for c in backend.calls)
    assert any(c[0] == "force_remove" for c in backend.calls)


@pytest.mark.asyncio
async def test_image_pass_retains_ttl_removal_for_session_that_woke(
    fake_pool: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retention_ttl removal verdict is re-verified under the lock — a session
    that woke since the tick-start load keeps its canonical snapshot image."""
    backend = FakeBackend()
    registry = SandboxRegistry(backend=backend)
    _patch_fresh_read(monkeypatch, dormant=False)  # woke
    tag = snapshot_tag(get_settings().instance_id, "sess_x")
    verdict = GcImageVerdict(
        image=ManagedImage(
            image_id="img", repo_tags=(tag,), parent_id=None, size_bytes=1, labels={}
        ),
        session_id="sess_x",
        is_canonical=True,
        removal_ref=tag,
        verdict="remove",
        reason="retention_ttl",
    )

    retained = await registry._gc_image_pass(
        [verdict],
        {"sess_x": _state("sess_x", dormant=True)},
        _NOW,
        _TTL,
        get_settings().instance_id,
    )

    assert retained == [verdict], "a woke session's snapshot must not be expired a tick early"
    assert not any(c[0] == "remove_image" for c in backend.calls)
