"""Completeness + liveness guards on the GC's reconcile-by-absence path.

Two findings from the 2026-08-15 re-verification of PR #2124, both about the
SAME inversion (aios#2138: "I could not read it" read downstream as "it does
not exist"), reached by inputs the earlier per-id fix does not cover:

1. **The listing itself has no completeness check.** The per-id machinery in
   ``list_managed_images`` proves that every id which APPEARED in ``docker
   images`` is accounted for. An id that was never listed is invisible to that
   proof, so a transiently-empty listing returns ``[]`` at face value and
   ``_gc_once`` then reconciles against ``present_refs == the empty set`` —
   one statement nulls ``snapshot_ref``/``snapshot_host``/``snapshot_bytes``
   for EVERY pointer on the host.

2. **The clear predicate has no liveness or age protection.** A snapshot
   belonging to a currently live session whose tag merely failed to appear is
   eligible; the ``snapshot_updated_at`` CAS only protects pointers written
   AFTER enumeration began, not unchanged pointers of live sessions.

Each guard is paired with a NEGATIVE CONTROL: a guard only ever seen to refuse
is indistinguishable from one that refuses everything, and a GC that declines
every tick trades "destroys live state" for "never collects".
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from aios.config import get_settings
from aios.sandbox.backends import docker as docker_backend
from aios.sandbox.backends.base import SandboxBackendError
from aios.sandbox.backends.docker import DockerBackend
from aios.sandbox.registry import GcPressureResult, SandboxRegistry

_NOW = datetime(2026, 6, 10, tzinfo=UTC)


def _managed_line(iid: str) -> str:
    return f'{iid}\t\t1\t["tag-{iid}"]\t{json.dumps({"Labels": {}})}'


# ─── Finding 1: the listing layer itself must prove completeness ────────────


@pytest.mark.asyncio
async def test_transiently_empty_listing_is_not_reported_as_an_empty_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty read that does not reproduce is UNKNOWN, not "no images".

    This is the input the per-id completeness proof cannot see: nothing was
    listed, so there is no id to re-probe. Returning ``[]`` here hands the GC
    a view in which every pointer on the host is absent.
    """
    ids = ["sha256:0000", "sha256:0001"]
    listings = 0

    async def fake_run(argv: list[str], **kwargs: object) -> tuple[int, bytes, bytes]:
        nonlocal listings
        if argv[1] == "images":
            listings += 1
            # First read comes back empty; the host in fact holds two images.
            if listings == 1:
                return 0, b"", b""
            return 0, ("\n".join(ids) + "\n").encode(), b""
        lines = [_managed_line(iid) for iid in argv[5:]]
        return 0, ("\n".join(lines) + "\n").encode(), b""

    monkeypatch.setattr(docker_backend, "run_docker_cli", fake_run)

    with pytest.raises(SandboxBackendError, match="incomplete managed image enumeration"):
        await DockerBackend().list_managed_images(instance_id="test")


@pytest.mark.asyncio
async def test_genuinely_empty_host_still_enumerates_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NEGATIVE CONTROL: a real empty host must not fail the tick forever.

    Failing closed on every empty listing would stall the GC permanently on a
    freshly provisioned host — the "never collects" failure the guard must not
    trade into.
    """
    listings = 0

    async def fake_run(argv: list[str], **kwargs: object) -> tuple[int, bytes, bytes]:
        nonlocal listings
        if argv[1] == "images":
            listings += 1
            return 0, b"", b""
        raise AssertionError("no inspect should run for an empty host")

    monkeypatch.setattr(docker_backend, "run_docker_cli", fake_run)

    assert await DockerBackend().list_managed_images(instance_id="test") == []
    # The emptiness was CONFIRMED, not assumed.
    assert listings == 2


@pytest.mark.asyncio
async def test_nonempty_listing_does_not_pay_for_a_second_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NEGATIVE CONTROL: the healthy path is untouched by the guard."""
    ids = ["sha256:0000", "sha256:0001"]
    listings = 0

    async def fake_run(argv: list[str], **kwargs: object) -> tuple[int, bytes, bytes]:
        nonlocal listings
        if argv[1] == "images":
            listings += 1
            return 0, ("\n".join(ids) + "\n").encode(), b""
        lines = [_managed_line(iid) for iid in argv[5:]]
        return 0, ("\n".join(lines) + "\n").encode(), b""

    monkeypatch.setattr(docker_backend, "run_docker_cli", fake_run)

    images = await DockerBackend().list_managed_images(instance_id="test")

    assert {image.image_id for image in images} == set(ids)
    assert listings == 1


def _stub_gc_passes(registry: SandboxRegistry) -> AsyncMock:
    """Neutralise every pass except the absence reconcile under test."""
    reconcile = AsyncMock()
    registry._gc_reconcile_absent_pointers = reconcile  # type: ignore[method-assign]
    registry._load_gc_states = AsyncMock(return_value={})  # type: ignore[method-assign]
    registry._gc_corpse_pass = AsyncMock()  # type: ignore[method-assign]
    registry._gc_image_pass = AsyncMock(return_value=[])  # type: ignore[method-assign]
    registry._gc_canonical_store_pass = AsyncMock()  # type: ignore[method-assign]
    registry._gc_pool_budget_pass = AsyncMock(  # type: ignore[method-assign]
        return_value=GcPressureResult()
    )
    registry._gc_account_cap_pass = AsyncMock(  # type: ignore[method-assign]
        return_value=GcPressureResult()
    )
    registry._gc_reconcile_pointers = AsyncMock()  # type: ignore[method-assign]
    return reconcile


@pytest.mark.asyncio
async def test_gc_does_not_clear_every_pointer_from_a_transiently_empty_listing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """END-TO-END: the wrong outcome is reached through ``_gc_once`` or not at all.

    Entering at the backend proves the guard refuses; entering at ``_gc_once``
    proves the refusal actually stops the DB write, which is the consequence
    the finding named.
    """
    listings = 0

    async def fake_run(argv: list[str], **kwargs: object) -> tuple[int, bytes, bytes]:
        nonlocal listings
        if argv[1] == "ps":
            return 0, b"", b""
        if argv[1] == "images":
            listings += 1
            if listings == 1:
                return 0, b"", b""
            return 0, b"sha256:0000\n", b""
        lines = [_managed_line(iid) for iid in argv[5:]]
        return 0, ("\n".join(lines) + "\n").encode(), b""

    monkeypatch.setattr(docker_backend, "run_docker_cli", fake_run)
    registry = SandboxRegistry(DockerBackend())
    reconcile = _stub_gc_passes(registry)

    with pytest.raises(SandboxBackendError, match="incomplete managed image enumeration"):
        await registry._gc_once(AsyncMock())

    reconcile.assert_not_awaited()


# ─── Finding 2: the clear predicate must protect live sessions ──────────────


@pytest.mark.asyncio
async def test_absence_reconcile_excludes_live_and_recent_sessions() -> None:
    """The clear must be scoped away from live/recent state, not just CAS'd.

    ``snapshot_updated_at <= observed_before`` protects a pointer written
    AFTER enumeration began. It does nothing for an UNCHANGED pointer whose
    session is live right now and whose tag merely failed to appear.
    """
    instance_id = get_settings().instance_id
    registry = SandboxRegistry(backend=AsyncMock())
    # A session this worker is actively holding a sandbox for.
    registry._handles["sess_live"] = object()  # type: ignore[assignment]
    reconcile = AsyncMock(return_value=0)

    from unittest.mock import patch

    with (
        patch("aios.sandbox.registry.queries.unscoped_reconcile_absent_host_snapshots", reconcile),
        patch("aios.harness.runtime.require_pool", return_value=_FakePool()),
    ):
        await registry._gc_reconcile_absent_pointers(set(), instance_id, observed_before=_NOW)

    reconcile.assert_awaited_once()
    kwargs = reconcile.await_args.kwargs  # type: ignore[union-attr]
    # The live session is handed to the query as protected, and an age floor
    # keeps just-written pointers out of a negative reconciliation.
    assert "sess_live" in set(kwargs["protected_session_ids"])
    assert kwargs["min_age"] > timedelta(0)


class _FakeConn:
    async def execute(self, *args: object, **kwargs: object) -> str:
        return "UPDATE 0"

    async def fetch(self, *args: object, **kwargs: object) -> list[object]:
        return []


class _FakeAcquire:
    async def __aenter__(self) -> _FakeConn:
        return _FakeConn()

    async def __aexit__(self, *args: object) -> bool:
        return False


class _FakePool:
    def acquire(self) -> _FakeAcquire:
        return _FakeAcquire()
