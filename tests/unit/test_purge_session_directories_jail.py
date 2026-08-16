"""Negative coverage for ``purge_session_directories``'s ownership guard.

The destructive property under test: session deletion ``rmtree``s host
directories, so the guard authorising that ``rmtree`` must REFUSE any
target that is not exclusively owned by the session being deleted -- it
must leave that directory INTACT.

The assertion is on the filesystem, not on an exception type.  Refusing
the ``rmtree`` is the safety property; aborting the delete is not.
``purge_session_directories`` runs after ``delete_session`` has already
committed the row removal, so raising would report a failed DELETE for a
session that is in fact gone.  An earlier revision of this guard did
raise, and it broke a legitimate reachable case -- see
``TestPermitsLegitimatePurge.test_workflow_shared_run_workspace_is_skipped_not_fatal``.

The pre-existing suite only asserted that a legitimate purge removes four
directories — nothing asserted anything was ever refused, so replacing the
guard body with ``pass`` left the suite green.  The tests here are written
so that neutering the guard turns them RED.

The reachable case that motivates them: ``workspace_volume_path`` is
user-supplied via ``POST /v1/sessions`` and ``validate_workspace_path``
checks ``is_relative_to(account_root)``, which is REFLEXIVE — so the
account root itself passes create-time validation and is stored verbatim.
Deleting that session must not ``rmtree`` the tenant's whole tree, taking
every sibling session's live workspace with it.

Positive controls are interleaved deliberately: a guard only ever seen to
refuse is indistinguishable from one that refuses everything, which would
make session deletion silently leak every directory it should reclaim.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aios.config import get_settings
from aios.sandbox.volumes import purge_session_directories

ACCOUNT = "acc_purge"
SESSION = "sess_purge"


@pytest.fixture
def workspace_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    settings = get_settings()
    monkeypatch.setattr(settings, "workspace_root", tmp_path)
    return tmp_path


def _populate(path: Path) -> Path:
    """Create ``path`` with a marker file whose survival is the assertion."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "marker.txt").write_text("payload")
    return path


def _session_owned_dirs(root: Path, session_id: str = SESSION) -> list[Path]:
    """The four directories a legitimate purge of ``session_id`` removes."""
    return [
        root / ACCOUNT / session_id,
        root / "_uploads" / session_id,
        root / "_attachments" / session_id,
        root / "_session_repos" / session_id,
    ]


class TestRefusesUnownedTargets:
    """Every one of these paths passes ``is_relative_to(workspace_root)``.
    Containment in the jail is NOT ownership; the guard must say no."""

    def test_refuses_account_root_and_deletes_nothing(self, workspace_root: Path) -> None:
        """The reachable data-loss case: workspace_path == the account root.

        Passes create-time validation (reflexive ``is_relative_to``), so it
        reaches the purge path verbatim. Purging it would ``rmtree`` every
        other live session's workspace for the tenant.
        """
        account_root = _populate(workspace_root / ACCOUNT)
        victim = _populate(account_root / "sess_other_live_session")
        mine = _populate(account_root / SESSION)

        purge_session_directories(SESSION, account_root, account_id=ACCOUNT)

        # "Refuses" means exactly one thing here: the directory is still there.
        assert victim.exists(), "another session's workspace was destroyed"
        assert (victim / "marker.txt").read_text() == "payload"
        assert account_root.exists()
        assert mine.exists(), "refusal must be atomic: nothing deleted"

    def test_refuses_workspace_root_itself(self, workspace_root: Path) -> None:
        _populate(workspace_root / ACCOUNT / "sess_other")
        purge_session_directories(SESSION, workspace_root, account_id=ACCOUNT)
        assert (workspace_root / ACCOUNT / "sess_other" / "marker.txt").exists()

    def test_refuses_sibling_sessions_directory(self, workspace_root: Path) -> None:
        """A stale/duplicated row pointing at ANOTHER session's dir."""
        sibling = _populate(workspace_root / ACCOUNT / "sess_sibling")
        purge_session_directories(SESSION, sibling, account_id=ACCOUNT)
        assert (sibling / "marker.txt").exists()

    def test_refuses_other_accounts_directory(self, workspace_root: Path) -> None:
        cross = _populate(workspace_root / "acc_other" / "sess_theirs")
        purge_session_directories(SESSION, cross, account_id=ACCOUNT)
        assert (cross / "marker.txt").exists()

    def test_refuses_shared_reserved_roots(self, workspace_root: Path) -> None:
        """``_uploads`` / ``_attachments`` hold EVERY session's subdir."""
        for reserved in ("_uploads", "_attachments", "_session_repos", "_memory_stores"):
            shared = _populate(workspace_root / reserved / "sess_someone_else")
            purge_session_directories(SESSION, workspace_root / reserved, account_id=ACCOUNT)
            assert (shared / "marker.txt").exists()

    def test_refuses_out_of_jail_path(self, workspace_root: Path, tmp_path: Path) -> None:
        outside = _populate(tmp_path.parent / f"outside_{tmp_path.name}")
        try:
            purge_session_directories(SESSION, outside, account_id=ACCOUNT)
            assert (outside / "marker.txt").exists()
        finally:
            (outside / "marker.txt").unlink()
            outside.rmdir()

    def test_refuses_symlink_that_escapes_the_jail(self, workspace_root: Path) -> None:
        """Resolution must happen on the REAL path, not the pattern.

        A symlink sitting at the session's own canonical location still
        points out of the jail; matching the string would authorise the
        ``rmtree`` of whatever it targets.
        """
        outside = _populate(workspace_root.parent / f"escape_{workspace_root.name}")
        link = workspace_root / ACCOUNT / SESSION
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(outside, target_is_directory=True)
        try:
            purge_session_directories(SESSION, link, account_id=ACCOUNT)
            assert (outside / "marker.txt").exists(), "symlink escaped the jail"
        finally:
            (outside / "marker.txt").unlink()
            outside.rmdir()

    def test_refusing_one_candidate_still_reclaims_the_session_s_own_dirs(
        self, workspace_root: Path
    ) -> None:
        """An unowned ``workspace_path`` must not suppress the OTHER purges.

        The uploads/attachments/repos dirs are derived from ``session_id``
        and are unambiguously this session's, whatever the workspace row
        says. Skipping is per-candidate: refusing the account root reclaims
        nothing less. An all-or-nothing refusal leaked these forever for
        precisely the sessions with an anomalous workspace row.
        """
        _, uploads, attachments, repos = _session_owned_dirs(workspace_root)
        for owned in (uploads, attachments, repos):
            _populate(owned)
        account_root = _populate(workspace_root / ACCOUNT)
        sibling = _populate(account_root / "sess_other_live_session")

        purge_session_directories(SESSION, account_root, account_id=ACCOUNT)

        assert sibling.exists(), "another session's workspace was destroyed"
        assert account_root.exists(), "the shared account root was rmtree'd"
        for owned in (uploads, attachments, repos):
            assert not owned.exists(), f"{owned} is this session's own and must be reclaimed"


class TestPermitsLegitimatePurge:
    """A guard only ever observed refusing could be refusing everything."""

    def test_workflow_shared_run_workspace_is_skipped_not_fatal(self, workspace_root: Path) -> None:
        """The regression that a raising guard caused (integration:
        ``test_operator_deleted_child_resolves_run_as_child_gone``).

        A workflow ``agent()`` child spawned with the default
        ``workspace='shared'`` stores the RUN's workspace
        (``<root>/<account_id>/_runs/<run_id>``) as its own
        ``workspace_volume_path``. That dir belongs to the run and is shared
        with the parent and every sibling child, so refusing to delete it is
        CORRECT -- but the refusal must not be fatal: ``delete_session`` has
        already committed the row removal by the time this runs, so raising
        reports a failed DELETE for a session that is in fact gone.
        """
        run_workspace = _populate(workspace_root / ACCOUNT / "_runs" / "run_abc")
        uploads = _populate(workspace_root / "_uploads" / SESSION)

        purge_session_directories(SESSION, run_workspace, account_id=ACCOUNT)

        assert (run_workspace / "marker.txt").exists(), "the run's shared workspace was destroyed"
        assert not uploads.exists(), "the child's own uploads dir must still be reclaimed"

    def test_purges_canonical_session_directories(self, workspace_root: Path) -> None:
        owned = _session_owned_dirs(workspace_root)
        for path in owned:
            _populate(path)
        # Bystanders that must survive a legitimate purge.
        sibling = _populate(workspace_root / ACCOUNT / "sess_sibling")
        other_uploads = _populate(workspace_root / "_uploads" / "sess_sibling")

        purge_session_directories(SESSION, owned[0], account_id=ACCOUNT)

        assert all(not path.exists() for path in owned), "legitimate purge did not delete"
        assert (sibling / "marker.txt").exists()
        assert (other_uploads / "marker.txt").exists()
        assert (workspace_root / ACCOUNT).exists(), "account root must survive"

    def test_purges_legacy_pre_409_workspace_layout(self, workspace_root: Path) -> None:
        """Pre-#409 rows hold ``<workspace_root>/<session_id>`` — still this
        session's alone, and still reclaimable."""
        legacy = _populate(workspace_root / SESSION)
        uploads = _populate(workspace_root / "_uploads" / SESSION)
        sibling = _populate(workspace_root / ACCOUNT / "sess_sibling")

        purge_session_directories(SESSION, legacy, account_id=ACCOUNT)

        assert not legacy.exists()
        assert not uploads.exists()
        assert (sibling / "marker.txt").exists()

    def test_purges_subdirectory_of_the_session_workspace(self, workspace_root: Path) -> None:
        """Strictly inside the session's own dir is owned by it."""
        nested = _populate(workspace_root / ACCOUNT / SESSION / "nested")
        purge_session_directories(SESSION, nested, account_id=ACCOUNT)
        assert not nested.exists()
        assert (workspace_root / ACCOUNT / SESSION).exists()

    def test_missing_directories_are_not_an_error(self, workspace_root: Path) -> None:
        """Deletion is idempotent — a re-delete must not raise."""
        owned = _session_owned_dirs(workspace_root)[0]
        _populate(owned)
        purge_session_directories(SESSION, owned, account_id=ACCOUNT)
        purge_session_directories(SESSION, owned, account_id=ACCOUNT)
        assert not owned.exists()
