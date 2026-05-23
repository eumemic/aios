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

from pathlib import Path

import pytest

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


class TestNonAbsoluteGuard:
    """Fail-fast guard: any non-absolute ``raw_path`` is rejected with a
    diagnostic message before ``Path.resolve()`` runs.

    Without this guard a stale pre-#409 session row whose
    ``workspace_volume_path`` was persisted as a relative path (e.g.
    ``workspaces/sess_X``) would be silently resolved against the
    worker's CWD by ``Path.resolve()``, yielding a confusing
    "must resolve to within the account's workspace subdirectory" error
    that hides the real cause — a row that needs the absolute-legacy
    backfill migration. See aios#668 (and #626 for the broader
    legacy-row story)."""

    def test_relative_path_raises_with_diagnostic_message(self, workspace_root: Path) -> None:
        with pytest.raises(ForbiddenError) as excinfo:
            validate_workspace_path("workspaces/sess_X", "acc_Y")
        assert "must be absolute" in excinfo.value.message
        assert "got non-absolute value 'workspaces/sess_X'" in excinfo.value.message

    def test_relative_path_detail_includes_session_id_and_raw_path(
        self, workspace_root: Path
    ) -> None:
        with pytest.raises(ForbiddenError) as excinfo:
            validate_workspace_path("relative/path", "acc_Y", session_id="sess_ABC")
        assert excinfo.value.detail == {
            "workspace_path": "relative/path",
            "session_id": "sess_ABC",
        }

    def test_relative_path_detail_with_no_session_id_is_none(self, workspace_root: Path) -> None:
        with pytest.raises(ForbiddenError) as excinfo:
            validate_workspace_path("relative/path", "acc_Y")
        assert excinfo.value.detail == {
            "workspace_path": "relative/path",
            "session_id": None,
        }

    def test_empty_string_treated_as_non_absolute(self, workspace_root: Path) -> None:
        with pytest.raises(ForbiddenError) as excinfo:
            validate_workspace_path("", "acc_Y")
        assert "must be absolute" in excinfo.value.message

    def test_guard_fires_before_other_checks(self, workspace_root: Path) -> None:
        """The non-absolute guard short-circuits — the caller sees the
        "must be absolute" diagnostic, not the generic
        "must resolve to within..." message produced by the downstream
        jail check. This documents the ordering invariant: a relative
        ``raw_path`` is never silently resolved against CWD."""
        with pytest.raises(ForbiddenError) as excinfo:
            validate_workspace_path("some/relative/path", "acc_Y")
        assert "must be absolute" in excinfo.value.message
        assert "must resolve to within" not in excinfo.value.message
