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
from aios.ids import make_id
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


def _state(session_id: str, *, dormant: bool, archived: bool = False) -> SessionSnapshotState:
    return SessionSnapshotState(
        session_id=session_id,
        account_id="acct",
        archived=archived,
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
async def test_corpse_pass_drops_archived_session_without_commit(
    fake_pool: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An archived session's corpse is dropped even when its activity is recent."""
    backend = FakeBackend()
    registry = SandboxRegistry(backend=backend)
    container = ManagedSandboxRef(sandbox_id="cid", session_id="sess_x", running=False)
    fresh = AsyncMock()
    monkeypatch.setattr(registry, "_fresh_session_state", fresh)

    await registry._gc_corpse_pass(
        [container],
        {"sess_x": _state("sess_x", dormant=False, archived=True)},
        _NOW,
        get_settings(),
        get_settings().instance_id,
    )

    assert not any(c[0] == "snapshot" for c in backend.calls)
    assert any(c[0] == "force_remove" for c in backend.calls)
    fresh.assert_not_awaited()


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
async def test_corpse_pass_run_owner_dropped_without_db_lookup_or_snapshot(
    fake_pool: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A workflow-run (wfr_) corpse is bare-destroyed by owner kind — never
    routed through the session retain path (no gc_snapshot_session_states
    query, no snapshot)."""
    backend = FakeBackend()
    registry = SandboxRegistry(backend=backend)
    fresh = AsyncMock(return_value=[])
    monkeypatch.setattr("aios.sandbox.registry.queries.gc_snapshot_session_states", fresh)
    run_id = make_id("wfr")  # a valid wfr_ ULID-shaped owner id
    container = ManagedSandboxRef(sandbox_id="cid", session_id=run_id, running=False)

    await registry._gc_corpse_pass(
        [container], {}, _NOW, get_settings(), get_settings().instance_id
    )

    assert any(c[0] == "force_remove" for c in backend.calls)
    assert not any(c[0] == "snapshot" for c in backend.calls)
    # The run branch must NOT consult the sessions table at all — on master
    # the coincidence-path issues this guaranteed-empty query.
    fresh.assert_not_awaited()


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


# ─── account-cap eviction pass (per-account snapshot quota, §5.7) ──────────────


def _acct_state(session_id: str, *, account_id: str, days_dormant: float) -> SessionSnapshotState:
    return SessionSnapshotState(
        session_id=session_id,
        account_id=account_id,
        archived=False,
        last_event_at=_NOW - timedelta(days=days_dormant),
        snapshot_ref=snapshot_tag(get_settings().instance_id, session_id),
        snapshot_host=get_settings().instance_id,
        snapshot_bytes=1_000_000,
    )


def _canonical_verdict(session_id: str, *, size_bytes: int) -> GcImageVerdict:
    """A retained canonical verdict whose image has no base/flattened labels, so
    ``_unique_bytes_for_image`` charges the full ``size_bytes``."""
    tag = snapshot_tag(get_settings().instance_id, session_id)
    return GcImageVerdict(
        image=ManagedImage(
            image_id=f"img-{session_id}",
            repo_tags=(tag,),
            parent_id=None,
            size_bytes=size_bytes,
            labels={},
        ),
        session_id=session_id,
        is_canonical=True,
        removal_ref=tag,
        verdict="retain",
        reason="live",
    )


def _patch_caps(monkeypatch: pytest.MonkeyPatch, caps: dict[str, int | None]) -> None:
    async def _resolve(_conn: Any, account_id: str) -> int | None:
        return caps.get(account_id)

    monkeypatch.setattr(
        "aios.sandbox.registry.queries.resolve_effective_sandbox_snapshot_bytes",
        AsyncMock(side_effect=_resolve),
    )


def _stub_event_and_pointer(registry: SandboxRegistry) -> None:
    """Stub the DB-touching side effects of an eviction so the pass can run on
    the FakePool (which has no real ``transaction``). The event text itself is
    asserted in ``tests/unit/test_context_fs_lifecycle.py``."""
    registry._append_fs_event = AsyncMock()  # type: ignore[method-assign]
    registry._clear_pointer_if_owned = AsyncMock()  # type: ignore[method-assign]


@pytest.mark.asyncio
async def test_account_cap_pass_evicts_most_dormant_first(
    fake_pool: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An over-cap account's MOST-DORMANT snapshots are evicted first until the
    account is back under cap, each with ``sandbox_fs_expired {account_cap}``."""
    backend = FakeBackend()
    registry = SandboxRegistry(backend=backend)
    _stub_event_and_pointer(registry)
    _patch_caps(monkeypatch, {"acct_a": 2_500_000})

    # Three sessions of acct_a, 1MB each (3MB > 2.5MB cap). Evicting the single
    # most-dormant (sess_old, 6 days) brings the account to 2MB ≤ cap.
    fresh = _canonical_verdict("sess_new", size_bytes=1_000_000)
    mid = _canonical_verdict("sess_mid", size_bytes=1_000_000)
    old = _canonical_verdict("sess_old", size_bytes=1_000_000)
    states = {
        "sess_new": _acct_state("sess_new", account_id="acct_a", days_dormant=1),
        "sess_mid": _acct_state("sess_mid", account_id="acct_a", days_dormant=3),
        "sess_old": _acct_state("sess_old", account_id="acct_a", days_dormant=6),
    }

    await registry._gc_account_cap_pass([fresh, mid, old], states, get_settings().instance_id)

    removed = [c[1]["ref"] for c in backend.calls if c[0] == "remove_image"]
    assert removed == [old.removal_ref], (
        "only the single most-dormant snapshot should be evicted to drop under cap"
    )
    # The eviction emits a model-visible sandbox_fs_expired {account_cap} event.
    registry._append_fs_event.assert_awaited_once_with(  # type: ignore[attr-defined]
        "sess_old", "sandbox_fs_expired", {"reason": "account_cap"}
    )


@pytest.mark.asyncio
async def test_account_cap_pass_leaves_under_cap_account_untouched(
    fake_pool: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An account whose total is at/under its cap loses nothing."""
    backend = FakeBackend()
    registry = SandboxRegistry(backend=backend)
    _patch_caps(monkeypatch, {"acct_a": 5_000_000})
    verdicts = [
        _canonical_verdict("sess_1", size_bytes=1_000_000),
        _canonical_verdict("sess_2", size_bytes=1_000_000),
    ]
    states = {
        "sess_1": _acct_state("sess_1", account_id="acct_a", days_dormant=10),
        "sess_2": _acct_state("sess_2", account_id="acct_a", days_dormant=20),
    }

    await registry._gc_account_cap_pass(verdicts, states, get_settings().instance_id)

    assert not any(c[0] == "remove_image" for c in backend.calls)


@pytest.mark.asyncio
async def test_account_cap_pass_skips_accounts_with_no_cap(
    fake_pool: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An account with no configured cap is never enforced, however large."""
    backend = FakeBackend()
    registry = SandboxRegistry(backend=backend)
    _patch_caps(monkeypatch, {"acct_a": None})
    verdicts = [_canonical_verdict("sess_1", size_bytes=999_000_000)]
    states = {"sess_1": _acct_state("sess_1", account_id="acct_a", days_dormant=99)}

    await registry._gc_account_cap_pass(verdicts, states, get_settings().instance_id)

    assert not any(c[0] == "remove_image" for c in backend.calls)


@pytest.mark.asyncio
async def test_account_cap_pass_is_per_account(
    fake_pool: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each account is enforced against its own cap independently: an over-cap
    account is trimmed while an under-cap one is left alone."""
    backend = FakeBackend()
    registry = SandboxRegistry(backend=backend)
    _stub_event_and_pointer(registry)
    _patch_caps(monkeypatch, {"acct_over": 1_500_000, "acct_under": 5_000_000})
    over_old = _canonical_verdict("over_old", size_bytes=1_000_000)
    over_new = _canonical_verdict("over_new", size_bytes=1_000_000)
    under = _canonical_verdict("under_1", size_bytes=1_000_000)
    states = {
        "over_old": _acct_state("over_old", account_id="acct_over", days_dormant=9),
        "over_new": _acct_state("over_new", account_id="acct_over", days_dormant=1),
        "under_1": _acct_state("under_1", account_id="acct_under", days_dormant=99),
    }

    await registry._gc_account_cap_pass(
        [over_old, over_new, under], states, get_settings().instance_id
    )

    removed = [c[1]["ref"] for c in backend.calls if c[0] == "remove_image"]
    assert removed == [over_old.removal_ref]


@pytest.mark.asyncio
async def test_account_cap_pass_skips_waking_session(
    fake_pool: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A session that holds a live cached handle (waking) is never evicted out
    from under itself, even when it is the most-dormant over-cap candidate."""
    backend = FakeBackend()
    registry = SandboxRegistry(backend=backend)
    _stub_event_and_pointer(registry)
    _patch_caps(monkeypatch, {"acct_a": 1_500_000})
    old = _canonical_verdict("sess_old", size_bytes=1_000_000)
    new = _canonical_verdict("sess_new", size_bytes=1_000_000)
    states = {
        "sess_old": _acct_state("sess_old", account_id="acct_a", days_dormant=9),
        "sess_new": _acct_state("sess_new", account_id="acct_a", days_dormant=1),
    }
    # The most-dormant candidate is waking — mark it as holding a cached handle.
    registry._handles["sess_old"] = cast(Any, object())

    await registry._gc_account_cap_pass([old, new], states, get_settings().instance_id)

    removed = [c[1]["ref"] for c in backend.calls if c[0] == "remove_image"]
    # The waking session is skipped; the next-most-dormant is evicted instead.
    assert old.removal_ref not in removed
    assert removed == [new.removal_ref]


@pytest.mark.asyncio
async def test_reconcile_skips_snapshot_evicted_this_tick(
    fake_pool: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pass 4 must not re-point a session whose canonical image an earlier pass
    removed this tick.

    The eviction passes leave the verdict in ``retained`` and don't mutate the
    tick-start ``states``, so an evicted session still presents its pre-eviction
    pointer (here NULL — the first-commit-crash-heal window: a committed tag
    whose pointer write was lost). Without the ``already_evicted`` skip, pass 4
    heals that NULL pointer by writing a pointer to the image pass 3 just
    removed — a dangling pointer that makes the session unresumable.
    """
    instance_id = get_settings().instance_id
    backend = FakeBackend()
    registry = SandboxRegistry(backend=backend)
    _stub_event_and_pointer(registry)  # stub eviction's event + pointer-clear side effects
    verdict = _canonical_verdict("sess_x", size_bytes=2_000_000)
    states = {
        "sess_x": SessionSnapshotState(
            session_id="sess_x",
            account_id="acct",
            archived=False,
            last_event_at=_NOW - timedelta(days=1),
            snapshot_ref=None,  # live canonical image on disk, but NULL DB pointer
            snapshot_host=instance_id,
            snapshot_bytes=None,
        )
    }
    set_pointer = AsyncMock()
    monkeypatch.setattr("aios.sandbox.registry.queries.unscoped_set_session_snapshot", set_pointer)

    # Baseline: with nothing evicted, pass 4 legitimately heals the NULL pointer.
    # This proves the resurrection path is live, so the skip below is not vacuous.
    await registry._gc_reconcile_pointers([verdict], states, instance_id, already_evicted=set())
    set_pointer.assert_awaited_once()
    set_pointer.reset_mock()

    # Pass 3 evicts sess_x under disk pressure (2 MB snapshot vs 1 MB pool budget).
    evicted = await registry._gc_pool_budget_pass([verdict], states, 1_000_000, instance_id)
    assert evicted == {"sess_x"}
    assert verdict.removal_ref in backend.removed_image_refs

    # Pass 4, told what was evicted this tick, must NOT re-point the removed image.
    await registry._gc_reconcile_pointers([verdict], states, instance_id, already_evicted=evicted)
    set_pointer.assert_not_awaited()
