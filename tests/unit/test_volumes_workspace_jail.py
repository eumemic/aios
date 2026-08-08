"""Unit coverage for ``validate_workspace_path``.

The two-axis property the function guards:

- **Host-FS escape rejection.** Paths that resolve outside
  ``workspace_root`` (``/etc``, ``..``-traversal up and out) must be
  rejected at every call site.
- **Cross-tenant rejection.** Paths under
  ``workspace_root/{other_account_id}/...`` must be rejected when the
  caller is ``{account_id}``.

Plus the backward-compat carve-out for the pre-#409 default
(``<workspace_root>/<session_id>``): when callers at the bind-mount
boundary supply the session_id, legacy session rows must still resolve
so the worker can cold-start them after a restart. Callers at
session-create time leave ``session_id`` unset and the strict
account-jail check applies to user-supplied paths.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog

from aios.config import get_settings
from aios.errors import ForbiddenError
from aios.sandbox.volumes import validate_workspace_path


@pytest.fixture
def workspace_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    settings = get_settings()
    monkeypatch.setattr(settings, "workspace_root", tmp_path)
    return tmp_path


class TestStrictJail:
    """The create-time check (no session_id) must keep rejecting every
    pre-#590 escape vector."""

    def test_rejects_host_path(self, workspace_root: Path) -> None:
        with pytest.raises(ForbiddenError):
            validate_workspace_path("/etc", "acc_a")

    def test_rejects_cross_tenant_path(self, workspace_root: Path) -> None:
        cross = str(workspace_root / "acc_b" / "any_session")
        with pytest.raises(ForbiddenError):
            validate_workspace_path(cross, "acc_a")

    def test_rejects_dotdot_traversal(self, workspace_root: Path) -> None:
        traversed = str(workspace_root / "acc_a" / ".." / "acc_b" / "x")
        with pytest.raises(ForbiddenError):
            validate_workspace_path(traversed, "acc_a")

    def test_accepts_within_account_subdir(self, workspace_root: Path) -> None:
        own = str(workspace_root / "acc_a" / "shared")
        validate_workspace_path(own, "acc_a")


class TestLegacyDefaultCompat:
    """Pre-#409 sessions have ``workspace_volume_path =
    <workspace_root>/<session_id>`` — no per-tenant subdir. The
    bind-mount-boundary re-check must let these existing rows through;
    without that the worker can never cold-start any session that
    pre-dates the per-tenant default.

    See #626: the model surfaced a ``ForbiddenError`` on every tool
    call after the worker recycled the legacy session's sandbox.
    """

    def test_legacy_default_with_session_id_accepted(self, workspace_root: Path) -> None:
        """``<workspace_root>/<session_id>`` is the literal legacy
        default. When the caller (sandbox provisioner) supplies
        ``session_id``, this must resolve."""
        legacy_path = str(workspace_root / "sess_01abc")
        validate_workspace_path(legacy_path, "acc_a", session_id="sess_01abc")

    def test_legacy_default_without_session_id_rejected(self, workspace_root: Path) -> None:
        """At session-create time the caller doesn't know a
        session_id yet. User-supplied legacy-shaped paths must still be
        rejected so an attacker can't reach into ``<workspace_root>/
        <victim_session_id>`` by inventing a ULID."""
        legacy_path = str(workspace_root / "sess_01abc")
        with pytest.raises(ForbiddenError):
            validate_workspace_path(legacy_path, "acc_a")

    def test_legacy_form_descendant_not_accepted_as_legacy(self, workspace_root: Path) -> None:
        """Only the exact legacy path itself counts — descendants of
        ``<workspace_root>/<session_id>`` never appeared in any default
        and must not be treated as legacy. (The new-convention check
        still accepts paths under ``<workspace_root>/<account_id>/``
        unchanged.)"""
        deeper = str(workspace_root / "sess_01abc" / "evil")
        with pytest.raises(ForbiddenError):
            validate_workspace_path(deeper, "acc_a", session_id="sess_01abc")

    def test_legacy_form_for_a_different_session_rejected(self, workspace_root: Path) -> None:
        """Cross-tenant defense: the legacy carve-out is keyed on the
        session_id the provisioner is currently cold-starting. A path
        that matches the legacy shape but names a DIFFERENT session_id
        must be rejected — otherwise the carve-out would let any
        session bind-mount any other session's legacy workspace."""
        other_session_legacy = str(workspace_root / "sess_other")
        with pytest.raises(ForbiddenError):
            validate_workspace_path(other_session_legacy, "acc_a", session_id="sess_01abc")

    def test_legacy_path_symlinked_outside_workspace_root_rejected(
        self, workspace_root: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        """If ``<workspace_root>/<session_id>`` is a symlink whose target
        escapes ``workspace_root``, the carve-out must reject.  Without
        this check ``Path.resolve()`` dereferences the symlink on both
        sides of the equality comparison, the two resolved paths match,
        and the bind-mount would target the symlink's destination —
        re-opening the host-FS-escape vector that PR #590 closed."""
        outside_target = tmp_path_factory.mktemp("outside")
        symlink = workspace_root / "sess_01abc"
        symlink.symlink_to(outside_target)
        with pytest.raises(ForbiddenError):
            validate_workspace_path(str(symlink), "acc_a", session_id="sess_01abc")


class TestRelativePathRejection:
    """Relative ``workspace_path`` strings must be rejected with a
    clear error before ``Path.resolve()`` gets a chance to interpret
    them against the current process's CWD.

    See #626: legacy session rows persisted ``workspaces/<account>/<session>``
    when ``AIOS_WORKSPACE_ROOT`` was historically configured as a
    relative path.  ``Path.resolve()`` resolves these against the
    worker's CWD, the result lands outside the workspace jail, and
    every tool call surfaces ``ForbiddenError`` blamed on whatever
    path the model was just trying to read or write.  Failing fast
    on the relative-input case produces an unambiguous error that
    correctly identifies the stored ``workspace_volume_path`` as
    the culprit instead.
    """

    def test_relative_path_rejected_with_clear_message(self, workspace_root: Path) -> None:
        """A relative ``workspace_path`` must raise ``ForbiddenError``
        with the ``must be absolute`` message naming the actual non-
        absolute value, and ``detail`` echoes back both the raw input
        and the session_id (or ``None``) so log aggregation can point
        at the offending row without a separate DB query."""
        with pytest.raises(ForbiddenError) as exc_info:
            validate_workspace_path("workspaces/sess_01abc", "acc_a")
        assert "must be absolute" in str(exc_info.value)
        assert "got non-absolute value 'workspaces/sess_01abc'" in str(exc_info.value)
        assert exc_info.value.detail == {
            "workspace_path": "workspaces/sess_01abc",
            "session_id": None,
        }

    def test_relative_path_with_session_id_still_rejected(self, workspace_root: Path) -> None:
        """The legacy carve-out (``session_id`` provided) must not save
        a relative path: relative inputs are rejected before any
        equality-with-legacy-shape comparison runs, so an attacker
        cannot bypass the new guard by also matching the legacy shape.
        The session_id is also surfaced in ``detail`` for diagnostics."""
        with pytest.raises(ForbiddenError) as exc_info:
            validate_workspace_path("workspaces/sess_01abc", "acc_a", session_id="sess_01abc")
        assert "must be absolute" in str(exc_info.value)
        assert exc_info.value.detail == {
            "workspace_path": "workspaces/sess_01abc",
            "session_id": "sess_01abc",
        }

    def test_empty_string_treated_as_non_absolute(self, workspace_root: Path) -> None:
        """An empty ``raw_path`` is non-absolute and rejected with the
        same diagnostic — catches a vanishingly-improbable upstream bug
        (NULL coalesced to '' somewhere) without ambiguity."""
        with pytest.raises(ForbiddenError) as exc_info:
            validate_workspace_path("", "acc_a")
        assert "must be absolute" in str(exc_info.value)


@pytest.fixture
def capture_logs() -> Iterator[structlog.testing.LogCapture]:
    """Reconfigure structlog to capture emitted events in-process.

    Mirrors ``tests/unit/test_worker_exit_diagnostics.py`` so assertions
    read the structured key/value fields directly off
    ``capture_logs.entries`` rather than parsing rendered strings.
    """
    cap = structlog.testing.LogCapture()
    structlog.configure(processors=[cap])
    try:
        yield cap
    finally:
        structlog.reset_defaults()


class TestRejectionDiagnostic:
    """aios#2064: every rejection must leave a legible, non-secret trace.

    The account/worker ``AIOS_WORKSPACE_ROOT`` drift that disabled the
    standing session's filesystem tools surfaced only as an opaque
    ``ForbiddenError`` on each tool call — the two divergent resolved
    roots were invisible in the logs. ``validate_workspace_path`` now
    emits a ``workspace.path_rejected`` event carrying the resolved
    geometry (``workspace_root``, ``account_root``, ``resolved_path``)
    beside the raw input so the drift is diagnosable from logs alone.

    These tests pin the event name and the field set. They do NOT relax
    any jail assertion above — the ``ForbiddenError`` still fires; the
    log is strictly additional observability.
    """

    def test_outside_account_root_logs_resolved_geometry(
        self, workspace_root: Path, capture_logs: structlog.testing.LogCapture
    ) -> None:
        """A path that resolves outside the account root logs the full
        non-secret geometry that drove the fail-closed rejection."""
        outside = str(workspace_root / "acc_b" / "sess_x")
        with pytest.raises(ForbiddenError):
            validate_workspace_path(outside, "acc_a", session_id="sess_x")

        events = [
            e for e in capture_logs.entries if e.get("event") == "workspace.path_rejected"
        ]
        assert len(events) == 1
        entry = events[0]
        assert entry["log_level"] == "warning"
        assert entry["reason"] == "outside_account_root"
        assert entry["raw_path"] == outside
        assert entry["resolved_path"] == str(Path(outside).resolve())
        assert entry["workspace_root"] == str(workspace_root.resolve())
        assert entry["account_root"] == str((workspace_root / "acc_a").resolve())
        assert entry["account_id"] == "acc_a"
        assert entry["session_id"] == "sess_x"

    def test_non_absolute_path_logs_rejection(
        self, workspace_root: Path, capture_logs: structlog.testing.LogCapture
    ) -> None:
        """The relative-input branch also emits ``workspace.path_rejected``
        so a stale relative ``workspace_volume_path`` row is diagnosable
        from logs. ``resolved_path`` is ``None`` — a relative input is
        deliberately never ``resolve()``d here (it would bind to CWD)."""
        with pytest.raises(ForbiddenError):
            validate_workspace_path("workspaces/sess_x", "acc_a", session_id="sess_x")

        events = [
            e for e in capture_logs.entries if e.get("event") == "workspace.path_rejected"
        ]
        assert len(events) == 1
        entry = events[0]
        assert entry["log_level"] == "warning"
        assert entry["reason"] == "not_absolute"
        assert entry["raw_path"] == "workspaces/sess_x"
        assert entry["resolved_path"] is None
        assert entry["account_id"] == "acc_a"
        assert entry["session_id"] == "sess_x"

    def test_accepted_path_emits_no_rejection_log(
        self, workspace_root: Path, capture_logs: structlog.testing.LogCapture
    ) -> None:
        """A valid in-account path must NOT emit a rejection event — the
        log is a rejection signal, not a per-call trace, so it stays
        quiet on the happy path (no log spam, no false positives in
        aggregation/alerting)."""
        validate_workspace_path(str(workspace_root / "acc_a" / "shared"), "acc_a")
        assert not [
            e for e in capture_logs.entries if e.get("event") == "workspace.path_rejected"
        ]

    def test_rejection_log_carries_no_credentialish_fields(
        self, workspace_root: Path, capture_logs: structlog.testing.LogCapture
    ) -> None:
        """Guard the fail-closed-isolation invariant: the diagnostic must
        expose only path geometry + identifiers, never a secret. Pin the
        exact key set so a future edit can't silently widen it to include
        a token, header, or credential."""
        with pytest.raises(ForbiddenError):
            validate_workspace_path("/etc", "acc_a")

        events = [
            e for e in capture_logs.entries if e.get("event") == "workspace.path_rejected"
        ]
        assert len(events) == 1
        # structlog adds ``event`` and ``log_level``; the payload keys we
        # own are exactly these — no more.
        owned = set(events[0]) - {"event", "log_level"}
        assert owned == {
            "reason",
            "raw_path",
            "resolved_path",
            "workspace_root",
            "account_root",
            "account_id",
            "session_id",
        }
