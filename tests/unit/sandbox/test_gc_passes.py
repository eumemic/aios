"""Impure snapshot-GC pass tests for under-lock lifecycle re-verification.

Destructive archive cleanup must re-read archive state, grace, pointer, and host
ownership while holding the session lock. Lifecycle races fail closed; deleted
sessions and ephemeral run corpses may be destroyed without snapshotting.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from aios.config import get_settings
from aios.harness import runtime
from aios.ids import make_id
from aios.sandbox.backends.base import ManagedImage, ManagedSandboxRef
from aios.sandbox.registry import GcImageVerdict, SandboxRegistry, SessionSnapshotState
from aios.sandbox.snapshot_store import TarballStore
from aios.sandbox.spec import snapshot_tag
from tests.helpers.sandbox import FakeBackend, FakePool

_NOW = datetime(2026, 6, 10, tzinfo=UTC)
_ARCHIVE_GRACE = 30 * 24 * 3600


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
        archived_at=None,
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

    assert any(c[0] == "snapshot" for c in backend.calls)
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
async def test_image_pass_retains_archived_removal_after_unarchive(
    fake_pool: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit unarchive in the fresh read fails closed."""
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
        reason="archived",
    )

    retained = await registry._gc_image_pass(
        [verdict],
        {"sess_x": _state("sess_x", dormant=True)},
        _NOW,
        _ARCHIVE_GRACE,
        get_settings().instance_id,
    )

    assert retained == [verdict], "an explicitly unarchived session must retain its snapshot"
    assert not any(c[0] == "remove_image" for c in backend.calls)


# ─── observational account-cap pass (per-account snapshot quota, §5.7) ────────


def _acct_state(session_id: str, *, account_id: str, days_dormant: float) -> SessionSnapshotState:
    return SessionSnapshotState(
        session_id=session_id,
        account_id=account_id,
        archived_at=None,
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
        reason="protected_live",
    )


def _patch_caps(monkeypatch: pytest.MonkeyPatch, caps: dict[str, int | None]) -> None:
    async def _resolve(_conn: Any, account_id: str) -> int | None:
        return caps.get(account_id)

    monkeypatch.setattr(
        "aios.sandbox.registry.queries.resolve_effective_sandbox_snapshot_bytes",
        AsyncMock(side_effect=_resolve),
    )


def _stub_event_and_pointer(registry: SandboxRegistry) -> None:
    """Stub DB-touching methods so observational-pass tests can assert that
    neither lifecycle events nor pointer changes occur on the transaction-less FakePool."""
    registry._append_fs_event = AsyncMock()  # type: ignore[method-assign]
    registry._clear_pointer_if_owned = AsyncMock()  # type: ignore[method-assign]


@pytest.mark.asyncio
async def test_account_cap_pass_evicts_most_dormant_first(
    fake_pool: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An over-cap account is reported without removing snapshots or emitting
    filesystem lifecycle events."""
    backend = FakeBackend()
    registry = SandboxRegistry(backend=backend)
    _stub_event_and_pointer(registry)
    _patch_caps(monkeypatch, {"acct_a": 2_500_000})

    # Three sessions of acct_a, 1MB each (3MB > 2.5MB cap). The pass observes
    # the overage while retaining every lifecycle-protected snapshot.
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
    assert removed == []
    # Observing account pressure emits no filesystem lifecycle event.
    registry._append_fs_event.assert_not_awaited()  # type: ignore[attr-defined]


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
    """An account with no configured cap remains untouched, however large."""
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
    """Account-cap observation is independent per account and retains snapshots
    for both over-cap and under-cap accounts."""
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
    assert removed == []


@pytest.mark.asyncio
async def test_account_cap_pass_skips_waking_session(
    fake_pool: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Account pressure remains observational when a session has a live cached
    handle; no snapshot is removed."""
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
    # Pressure accounting does not remove either the waking or dormant snapshot.
    assert old.removal_ref not in removed
    assert removed == []


@pytest.mark.asyncio
async def test_reconcile_skips_snapshot_evicted_this_tick(
    fake_pool: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An observational pressure pass retains the canonical image and allows
    pointer reconciliation to heal a missing pointer in the same tick."""
    instance_id = get_settings().instance_id
    backend = FakeBackend()
    registry = SandboxRegistry(backend=backend)
    _stub_event_and_pointer(registry)  # verify pressure causes no lifecycle side effects
    verdict = _canonical_verdict("sess_x", size_bytes=2_000_000)
    states = {
        "sess_x": SessionSnapshotState(
            session_id="sess_x",
            account_id="acct",
            archived_at=None,
            last_event_at=_NOW - timedelta(days=1),
            snapshot_ref=None,  # live canonical image on disk, but NULL DB pointer
            snapshot_host=instance_id,
            snapshot_bytes=None,
        )
    }
    set_pointer = AsyncMock()
    monkeypatch.setattr("aios.sandbox.registry.queries.unscoped_set_session_snapshot", set_pointer)

    # Pointer reconciliation legitimately heals the NULL pointer.
    await registry._gc_reconcile_pointers([verdict], states, instance_id, already_evicted=set())
    set_pointer.assert_awaited_once()
    set_pointer.reset_mock()

    # Pass 3 reports disk pressure (2 MB snapshot vs 1 MB pool budget) without removal.
    pressure = await registry._gc_pool_budget_pass([verdict], states, 1_000_000, instance_id)
    assert pressure.pressured
    assert pressure.pool_used_bytes == 2_000_000
    assert verdict.removal_ref not in backend.removed_image_refs

    # Pressure reports capacity state without deleting or suppressing reconciliation.
    await registry._gc_reconcile_pointers([verdict], states, instance_id)
    assert set_pointer.await_count == 1


@pytest.mark.asyncio
async def test_archived_current_positive_ownership_is_removed() -> None:
    backend = FakeBackend()
    registry = SandboxRegistry(backend=backend)
    tag = snapshot_tag(get_settings().instance_id, "sess_x")
    archived_at = _NOW - timedelta(days=2)
    state = SessionSnapshotState(
        "sess_x", "acct", archived_at, _NOW, tag, get_settings().instance_id, 1
    )
    verdict = GcImageVerdict(
        ManagedImage("img", (tag,), None, 1, {}), "sess_x", True, tag, "remove", "archived"
    )
    registry._fresh_session_state = AsyncMock(return_value=state)  # type: ignore[method-assign]
    registry._append_fs_event = AsyncMock()  # type: ignore[method-assign]
    registry._clear_pointer_if_owned = AsyncMock()  # type: ignore[method-assign]

    retained = await registry._gc_image_pass(
        [verdict], {"sess_x": state}, _NOW, 86400, get_settings().instance_id
    )

    assert retained == []
    assert tag in backend.removed_image_refs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["null_host", "stale_grace", "pointer_moved", "unarchived", "rearchived"]
)
async def test_archived_current_destructive_path_fails_closed(failure: str) -> None:
    backend = FakeBackend()
    registry = SandboxRegistry(backend=backend)
    host = get_settings().instance_id
    tag = snapshot_tag(host, "sess_x")
    archived_at = _NOW - timedelta(days=2)
    candidate = SessionSnapshotState("sess_x", "acct", archived_at, _NOW, tag, host, 1)
    fresh = candidate
    if failure == "null_host":
        fresh = replace(candidate, snapshot_host=None)
    elif failure == "stale_grace":
        fresh = replace(candidate, archived_at=_NOW - timedelta(seconds=86399))
    elif failure == "pointer_moved":
        fresh = replace(candidate, snapshot_ref="other")
    elif failure == "unarchived":
        fresh = replace(candidate, archived_at=None)
    else:
        fresh = replace(candidate, archived_at=archived_at + timedelta(seconds=1))
    verdict = GcImageVerdict(
        ManagedImage("img", (tag,), None, 1, {}), "sess_x", True, tag, "remove", "archived"
    )
    registry._fresh_session_state = AsyncMock(return_value=fresh)  # type: ignore[method-assign]

    retained = await registry._gc_image_pass([verdict], {"sess_x": candidate}, _NOW, 86400, host)

    assert retained == [verdict]
    assert tag not in backend.removed_image_refs


@pytest.mark.asyncio
async def test_archived_within_grace_corpse_is_salvaged() -> None:
    backend = FakeBackend()
    registry = SandboxRegistry(backend=backend)
    state = replace(_state("sess_x", dormant=False), archived_at=_NOW - timedelta(seconds=16))
    registry._fresh_session_state = AsyncMock(return_value=state)  # type: ignore[method-assign]
    registry._snapshot_and_record = AsyncMock(return_value=True)  # type: ignore[method-assign]
    container = ManagedSandboxRef(sandbox_id="cid", session_id="sess_x", running=False)
    settings = get_settings().model_copy(update={"sandbox_archive_gc_grace_seconds": 17})

    await registry._gc_corpse_pass(
        [container], {"sess_x": state}, _NOW, settings, get_settings().instance_id
    )

    registry._snapshot_and_record.assert_awaited_once()
    assert any(call[0] == "force_remove" for call in backend.calls)


@pytest.mark.asyncio
async def test_snapshot_failure_retains_archived_grace_corpse() -> None:
    backend = FakeBackend()
    registry = SandboxRegistry(backend=backend)
    state = replace(_state("sess_x", dormant=False), archived_at=_NOW)
    registry._fresh_session_state = AsyncMock(return_value=state)  # type: ignore[method-assign]
    registry._snapshot_and_record = AsyncMock(return_value=False)  # type: ignore[method-assign]
    container = ManagedSandboxRef(sandbox_id="cid", session_id="sess_x", running=False)
    settings = get_settings().model_copy(update={"sandbox_archive_gc_grace_seconds": 17})

    await registry._gc_corpse_pass(
        [container], {"sess_x": state}, _NOW, settings, get_settings().instance_id
    )

    registry._snapshot_and_record.assert_awaited_once()
    assert not any(call[0] == "force_remove" for call in backend.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("grace_seconds, age_seconds", [(17, 17), (0, 0)])
async def test_archived_corpse_at_boundary_is_bare_destroyed(
    grace_seconds: int, age_seconds: int
) -> None:
    backend = FakeBackend()
    registry = SandboxRegistry(backend=backend)
    archived_at = _NOW - timedelta(seconds=age_seconds)
    state = replace(_state("sess_x", dormant=False), archived_at=archived_at)
    registry._fresh_session_state = AsyncMock(return_value=state)  # type: ignore[method-assign]
    registry._snapshot_and_record = AsyncMock()  # type: ignore[method-assign]
    corpse = ManagedSandboxRef(sandbox_id="cid", session_id="sess_x", running=False)
    settings = get_settings().model_copy(update={"sandbox_archive_gc_grace_seconds": grace_seconds})

    await registry._gc_corpse_pass(
        [corpse], {"sess_x": state}, _NOW, settings, get_settings().instance_id
    )

    registry._snapshot_and_record.assert_not_awaited()
    assert ("force_remove", {"sandbox_id": "cid"}) in backend.calls


@pytest.mark.asyncio
async def test_archived_past_grace_corpse_is_bare_destroyed() -> None:
    backend = FakeBackend()
    registry = SandboxRegistry(backend=backend)
    state = replace(_state("sess_x", dormant=False), archived_at=_NOW - timedelta(days=2))
    registry._fresh_session_state = AsyncMock(return_value=state)  # type: ignore[method-assign]
    registry._snapshot_and_record = AsyncMock()  # type: ignore[method-assign]
    corpse = ManagedSandboxRef(sandbox_id="cid", session_id="sess_x", running=False)
    settings = get_settings().model_copy(update={"sandbox_archive_gc_grace_seconds": 86400})

    await registry._gc_corpse_pass(
        [corpse], {"sess_x": state}, _NOW, settings, get_settings().instance_id
    )

    registry._snapshot_and_record.assert_not_awaited()
    assert any(call[0] == "force_remove" for call in backend.calls)


@pytest.mark.asyncio
async def test_corpse_rearchive_race_fails_closed_and_salvages() -> None:
    backend = FakeBackend()
    registry = SandboxRegistry(backend=backend)
    scanned = replace(_state("sess_x", dormant=False), archived_at=_NOW - timedelta(days=2))
    fresh = replace(scanned, archived_at=_NOW - timedelta(days=2) + timedelta(seconds=1))
    registry._fresh_session_state = AsyncMock(return_value=fresh)  # type: ignore[method-assign]
    registry._snapshot_and_record = AsyncMock(return_value=True)  # type: ignore[method-assign]
    corpse = ManagedSandboxRef(sandbox_id="cid", session_id="sess_x", running=False)
    settings = get_settings().model_copy(update={"sandbox_archive_gc_grace_seconds": 86400})

    await registry._gc_corpse_pass(
        [corpse], {"sess_x": scanned}, _NOW, settings, get_settings().instance_id
    )

    registry._snapshot_and_record.assert_awaited_once()
    assert any(call[0] == "force_remove" for call in backend.calls)


@pytest.mark.asyncio
async def test_tarball_pointer_survives_docker_gc_reconciliation_and_prune(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Docker cache enumeration must never rewrite canonical durable refs."""
    backend = FakeBackend()
    registry = SandboxRegistry(backend=backend)
    registry._store = TarballStore(backend, tmp_path)
    sid = "sess_durable"
    durable_ref = f"{sid}/generation.tar"
    state = SessionSnapshotState(
        sid, "acct", None, _NOW, durable_ref, get_settings().instance_id, 7
    )
    set_pointer = AsyncMock()
    monkeypatch.setattr("aios.sandbox.registry.queries.unscoped_set_session_snapshot", set_pointer)
    verdict = _canonical_verdict(sid, size_bytes=7)

    await registry._gc_reconcile_pointers([verdict], {sid: state}, get_settings().instance_id)
    set_pointer.assert_not_awaited()
    # Simulate label-blind `docker image prune -af`: only the local cache vanishes.
    await backend.remove_image(verdict.removal_ref)
    assert state.snapshot_ref == durable_ref
