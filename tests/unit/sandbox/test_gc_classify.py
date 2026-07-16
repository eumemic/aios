"""Table-driven coverage for lifecycle-based snapshot GC classification.

Canonical snapshots of existing, non-archived sessions are protected regardless
of activity. Archived snapshots become collectible only when archive grace has
elapsed; deleted sessions and non-canonical residue are collectible immediately.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

from aios.sandbox.backends.base import SESSION_LABEL_KEY, ManagedImage
from aios.sandbox.registry import SessionSnapshotState, _classify_images
from aios.sandbox.spec import snapshot_tag

_HOST = "default"
_NOW = datetime(2026, 6, 10, tzinfo=UTC)


def _image(
    *,
    image_id: str,
    session_id: str | None,
    canonical: bool,
    parent_id: str | None = None,
    flattened: bool = False,
) -> ManagedImage:
    labels: dict[str, str] = {"aios.managed": "true"}
    if session_id is not None:
        labels[SESSION_LABEL_KEY] = session_id
    if flattened:
        labels["aios.flattened"] = "true"
    repo_tags = (snapshot_tag(_HOST, session_id),) if (canonical and session_id) else ()
    return ManagedImage(
        image_id=image_id,
        repo_tags=repo_tags,
        parent_id=parent_id,
        size_bytes=1_000_000,
        labels=labels,
    )


def _state(session_id: str, *, dormant: bool) -> SessionSnapshotState:
    last = _NOW - timedelta(days=40 if dormant else 1)
    return SessionSnapshotState(
        session_id=session_id,
        account_id="acct",
        archived_at=None,
        last_event_at=last,
        snapshot_ref=snapshot_tag(_HOST, session_id),
        snapshot_host=_HOST,
        snapshot_bytes=1_000_000,
    )


def _classify(
    images: list[ManagedImage], states: dict[str, SessionSnapshotState]
) -> dict[str, tuple[str, str]]:
    verdicts = _classify_images(
        images, states, now=_NOW, archive_grace_seconds=86400, this_host=_HOST
    )
    return {v.image.image_id: (v.verdict, v.reason) for v in verdicts}


def test_live_canonical_is_retained() -> None:
    img = _image(image_id="img_a", session_id="sess_a", canonical=True)
    out = _classify([img], {"sess_a": _state("sess_a", dormant=False)})
    assert out["img_a"] == ("retain", "protected_live")


def test_dormant_canonical_is_protected() -> None:
    img = _image(image_id="img_a", session_id="sess_a", canonical=True)
    out = _classify([img], {"sess_a": _state("sess_a", dormant=True)})
    assert out["img_a"] == ("retain", "protected_live")


def test_deleted_session_canonical_is_removed_no_event() -> None:
    """A session absent from the state map is deleted → remove, reason 'deleted'
    (no model-visible event; the session is gone)."""
    img = _image(image_id="img_a", session_id="sess_gone", canonical=True)
    out = _classify([img], {})  # session not present
    assert out["img_a"] == ("remove", "deleted")


def test_untagged_residue_is_removed() -> None:
    """An untagged (non-canonical) leaf — flatten leftover / crash residue —
    is removed even for a live session."""
    img = _image(image_id="img_resid", session_id="sess_a", canonical=False)
    out = _classify([img], {"sess_a": _state("sess_a", dormant=False)})
    assert out["img_resid"] == ("remove", "residue")


def test_structural_parent_skip_excludes_live_chain_interior() -> None:
    """An image that is the parent of another listed image (a live chain
    interior) is excluded — its leaf's removal cascade-deletes it."""
    leaf = _image(
        image_id="img_leaf", session_id="sess_a", canonical=True, parent_id="img_interior"
    )
    interior = _image(image_id="img_interior", session_id="sess_a", canonical=False)
    out = _classify([leaf, interior], {"sess_a": _state("sess_a", dormant=False)})
    assert "img_interior" not in out, "live-chain interior must be skipped, not classified"
    assert out["img_leaf"] == ("retain", "protected_live")


def test_archived_session_within_grace_is_retained() -> None:
    """An archived session remains protected until archive grace elapses."""
    img = _image(image_id="img_a", session_id="sess_a", canonical=True)
    state = SessionSnapshotState(
        session_id="sess_a",
        account_id="acct",
        archived_at=_NOW - timedelta(hours=12),
        last_event_at=_NOW - timedelta(days=1),
        snapshot_ref=snapshot_tag(_HOST, "sess_a"),
        snapshot_host=_HOST,
        snapshot_bytes=1,
    )
    out = _classify([img], {"sess_a": state})
    assert out["img_a"] == ("retain", "archive_grace")


def test_missing_dormancy_probe_is_not_dormant() -> None:
    """A session with no last_event_at (the should-not-happen no-events edge)
    reads as NOT dormant — conservative, never wipe on a missing probe."""
    img = _image(image_id="img_a", session_id="sess_a", canonical=True)
    state = SessionSnapshotState(
        session_id="sess_a",
        account_id="acct",
        archived_at=None,
        last_event_at=None,
        snapshot_ref=snapshot_tag(_HOST, "sess_a"),
        snapshot_host=_HOST,
        snapshot_bytes=1,
    )
    out = _classify([img], {"sess_a": state})
    assert out["img_a"] == ("retain", "protected_live")


def test_arbitrarily_dormant_non_archived_canonical_is_protected() -> None:
    img = _image(image_id="img_a", session_id="sess_a", canonical=True)
    state = _state("sess_a", dormant=True)
    out = _classify([img], {"sess_a": state})
    assert out["img_a"] == ("retain", "protected_live")


def test_archived_current_observes_grace_boundary() -> None:
    img = _image(image_id="img_a", session_id="sess_a", canonical=True)
    state = _state("sess_a", dormant=True)
    grace = 86400
    before = replace(state, archived_at=_NOW - timedelta(seconds=grace - 1))
    at = replace(state, archived_at=_NOW - timedelta(seconds=grace))
    assert (
        _classify_images(
            [img], {"sess_a": before}, now=_NOW, archive_grace_seconds=grace, this_host=_HOST
        )[0].verdict
        == "retain"
    )
    verdict = _classify_images(
        [img], {"sess_a": at}, now=_NOW, archive_grace_seconds=grace, this_host=_HOST
    )[0]
    assert (verdict.verdict, verdict.reason) == ("remove", "archived")


def test_zero_archive_grace_is_immediately_eligible() -> None:
    img = _image(image_id="img_a", session_id="sess_a", canonical=True)
    state = replace(_state("sess_a", dormant=False), archived_at=_NOW)
    verdict = _classify_images(
        [img], {"sess_a": state}, now=_NOW, archive_grace_seconds=0, this_host=_HOST
    )[0]
    assert (verdict.verdict, verdict.reason) == ("remove", "archived")


def test_nondefault_archive_grace_boundary() -> None:
    img = _image(image_id="img_a", session_id="sess_a", canonical=True)
    state = replace(_state("sess_a", dormant=False), archived_at=_NOW - timedelta(seconds=17))
    before = _classify_images(
        [img], {"sess_a": state}, now=_NOW, archive_grace_seconds=18, this_host=_HOST
    )[0]
    at = _classify_images(
        [img], {"sess_a": state}, now=_NOW, archive_grace_seconds=17, this_host=_HOST
    )[0]
    assert before.verdict == "retain"
    assert at.verdict == "remove"
