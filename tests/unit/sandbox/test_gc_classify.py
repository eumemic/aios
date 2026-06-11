"""Table-driven coverage for the GC retain-rule classifier (durable session
sandboxes, §5.5). ``_classify_images`` is pure, so it needs no Docker or DB.

The single rule: an image is RETAINED iff it is the canonical tag of an
existing session whose last activity is within the TTL. Everything else
managed-and-mine is removed — crash/flatten residue, deleted sessions (the
delete hook), and dormant sessions (flagged ``retention_ttl``). Untagged
interiors of live chains are skipped structurally.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from aios.sandbox.backends.base import SESSION_LABEL_KEY, ManagedImage
from aios.sandbox.registry import SessionSnapshotState, _classify_images
from aios.sandbox.spec import snapshot_tag

_HOST = "default"
_TTL = 30 * 24 * 3600
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
        archived=False,
        last_event_at=last,
        snapshot_ref=snapshot_tag(_HOST, session_id),
        snapshot_host=_HOST,
        snapshot_bytes=1_000_000,
    )


def _classify(
    images: list[ManagedImage], states: dict[str, SessionSnapshotState]
) -> dict[str, tuple[str, str]]:
    verdicts = _classify_images(images, states, now=_NOW, ttl_seconds=_TTL, this_host=_HOST)
    return {v.image.image_id: (v.verdict, v.reason) for v in verdicts}


def test_live_canonical_is_retained() -> None:
    img = _image(image_id="img_a", session_id="sess_a", canonical=True)
    out = _classify([img], {"sess_a": _state("sess_a", dormant=False)})
    assert out["img_a"] == ("retain", "live")


def test_dormant_canonical_is_removed_with_ttl_reason() -> None:
    img = _image(image_id="img_a", session_id="sess_a", canonical=True)
    out = _classify([img], {"sess_a": _state("sess_a", dormant=True)})
    assert out["img_a"] == ("remove", "retention_ttl")


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
    assert out["img_leaf"] == ("retain", "live")


def test_archived_session_within_ttl_is_retained() -> None:
    """Archived sessions follow the same dormancy rule (unarchive exists)."""
    img = _image(image_id="img_a", session_id="sess_a", canonical=True)
    state = SessionSnapshotState(
        session_id="sess_a",
        account_id="acct",
        archived=True,
        last_event_at=_NOW - timedelta(days=1),
        snapshot_ref=snapshot_tag(_HOST, "sess_a"),
        snapshot_host=_HOST,
        snapshot_bytes=1,
    )
    out = _classify([img], {"sess_a": state})
    assert out["img_a"] == ("retain", "live")


def test_missing_dormancy_probe_is_not_dormant() -> None:
    """A session with no last_event_at (the should-not-happen no-events edge)
    reads as NOT dormant — conservative, never wipe on a missing probe."""
    img = _image(image_id="img_a", session_id="sess_a", canonical=True)
    state = SessionSnapshotState(
        session_id="sess_a",
        account_id="acct",
        archived=False,
        last_event_at=None,
        snapshot_ref=snapshot_tag(_HOST, "sess_a"),
        snapshot_host=_HOST,
        snapshot_bytes=1,
    )
    out = _classify([img], {"sess_a": state})
    assert out["img_a"] == ("retain", "live")
