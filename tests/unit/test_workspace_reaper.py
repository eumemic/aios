"""Unit tests for ``aios.harness.workspace_reaper`` (#40 — the 45G hole).

The reaper reclaims the per-session ``/workspace`` host dir
(``<workspace_root>/<account_id>/<session_id>``) of archived sessions — the
large un-reaped store the #1192 reaper does NOT cover.

The load-bearing safety property is the **canonical-path confinement gate**:
the reaper reaps ONLY the default ``<root>/<account>/<session>`` path derived
from the row's own ids, and ONLY when the session's stored
``workspace_volume_path`` resolves equal to it. This forecloses the
data-loss vectors the design review found — a user-overridden / clone-shared /
aliased / parent / out-of-tree / relative path never matches and is skipped
(a space leak, never a wrong delete).

These tests assert each of the six required guardrails directly:

* archived-only + DB-liveness keep-set: only sessions the DB query returns
  (archived ∧ aged ∧ not-active) are candidates; a running session's dir is
  never reaped (proven via the query result, which the SQL itself enforces);
* min-age floor: a dir under the mtime floor is skipped;
* fail-closed: a DB error reaps nothing; a confinement mismatch skips;
* kill-switch: ``workspace_reaper_enabled=False`` deletes nothing;
* dry-run: deletes nothing but reports would-reap counts/bytes.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import asyncpg
import pytest

from aios.harness import workspace_reaper
from aios.harness.workspace_reaper import sweep_archived_workspaces


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Point ``workspace_root`` at a tmpdir, drop floors to 0, reaper enabled."""
    root = tmp_path / "workspaces"
    root.mkdir()

    settings = MagicMock()
    settings.workspace_root = root
    settings.workspace_reaper_enabled = True
    settings.workspace_reaper_dry_run = False
    settings.workspace_reaper_min_archived_age_seconds = 86_400
    settings.workspace_reaper_min_mtime_age_seconds = 0
    monkeypatch.setattr(workspace_reaper, "get_settings", lambda: settings)

    return {"root": root, "settings": settings}


def _mk_workspace(root: Path, account_id: str, session_id: str, *, age_s: float = 10_000.0) -> Path:
    """Create the canonical ``<root>/<account>/<session>`` dir, aged past floors."""
    d = root / account_id / session_id
    d.mkdir(parents=True)
    (d / "scratch.txt").write_text("model wrote this")
    old = time.time() - age_s
    os.utime(d, (old, old))
    return d


def _fake_pool(
    candidates: list[dict[str, Any]] | Exception,
    *,
    live_paths: list[str] | None = None,
) -> MagicMock:
    """Pool whose acquired conn answers the reaper's two ``conn.fetch`` calls.

    ``candidates`` are the rows the candidate query would yield — i.e. already
    filtered to archived ∧ aged ∧ not-active by the SQL. Each is a dict with
    ``id`` / ``account_id`` / ``workspace_volume_path``. Pass an ``Exception`` to
    simulate a DB error on the candidate read.

    ``live_paths`` are the stored ``workspace_volume_path`` values the LIVE-clone
    keep-set query (``unscoped_live_workspace_volume_paths``) would return; the
    reaper realpath-normalizes them and skips any candidate that collides. The
    two fetches are routed by SQL text: the candidate query selects
    ``archived_at IS NOT NULL``, the live query ``archived_at IS NULL``.
    """
    live_paths = live_paths or []
    conn = MagicMock()
    conn.execute = AsyncMock()
    conn.fetchval = AsyncMock(return_value=False)

    class _Tx:
        async def __aenter__(self) -> None:
            return None

        async def __aexit__(self, *_a: Any) -> None:
            return None

    conn.transaction.return_value = _Tx()

    if isinstance(candidates, Exception):
        conn.fetch = AsyncMock(side_effect=candidates)
    else:

        async def _route_fetch(sql: str, *_a: Any, **_k: Any) -> list[Any]:
            if "archived_at IS NULL" in sql:
                # the live-clone keep-set query: rows with workspace_volume_path
                return [{"workspace_volume_path": p} for p in live_paths]
            # the candidate query: archived ∧ aged ∧ not-active rows
            return candidates

        conn.fetch = AsyncMock(side_effect=_route_fetch)

    class _Cm:
        async def __aenter__(self) -> Any:
            return conn

        async def __aexit__(self, *_a: Any) -> None:
            return None

    pool = MagicMock()
    pool.acquire.return_value = _Cm()
    pool._conn = conn  # expose for assertions
    return pool


def _row(account_id: str, session_id: str, workspace_volume_path: str) -> dict[str, Any]:
    return {
        "id": session_id,
        "account_id": account_id,
        "workspace_volume_path": workspace_volume_path,
    }


# ── guardrail 1+2: archived-only + DB-liveness keep-set ──────────────────────


async def test_archived_aged_inactive_session_dir_is_reaped(env: dict[str, Any]) -> None:
    """A canonical-path workspace of an archived/aged/not-active session is reaped.

    The DB query (whose result we inject) is what enforces archived ∧ aged ∧
    NOT active — so a candidate reaching the reaper is by construction past the
    keep-set; the reaper deletes its workspace files.
    """
    d = _mk_workspace(env["root"], "acct1", "sess_archived")
    canonical = str(d)
    pool = _fake_pool([_row("acct1", "sess_archived", canonical)])

    result = await sweep_archived_workspaces(pool)

    assert result.reaped == 1
    assert not d.exists(), "an archived/aged/not-active session's workspace must be reaped"
    assert result.bytes_freed > 0


async def test_running_session_never_in_candidate_set(env: dict[str, Any]) -> None:
    """The keep-set: a live/running session is never a candidate, so its dir survives.

    The query returns ONLY archived ∧ not-active rows; a running session is
    absent. We model that by returning an empty candidate set while a running
    session's dir exists on disk — it must be untouched.
    """
    live = _mk_workspace(env["root"], "acct1", "sess_running")
    pool = _fake_pool([])  # query excludes the running session

    result = await sweep_archived_workspaces(pool)

    assert result.reaped == 0
    assert live.exists(), "a running session's workspace dir must survive (keep-set)"


# ── the CRITICAL confinement finding: shared / overridden / aliased paths ────


async def test_shared_override_path_is_never_reaped(env: dict[str, Any]) -> None:
    """A clone-shared / user-override workspace_volume_path is SKIPPED, not reaped.

    The data-loss vector the review found: ``clone_session`` lets two sessions
    share one volume. If session A is archived but its stored path is the SHARED
    override (not its canonical default), reaping the stored path would delete a
    possibly-live session B's ``/workspace``. The canonical-path gate skips it:
    the stored override never equals ``<root>/<account>/<sessA>``.
    """
    # Session B's live workspace, which A's archived row points at via override.
    shared = _mk_workspace(env["root"], "acct1", "sess_B_shared")
    # A's canonical default does NOT exist on disk (A used the override instead).
    pool = _fake_pool([_row("acct1", "sess_A_archived", str(shared))])

    result = await sweep_archived_workspaces(pool)

    assert result.reaped == 0
    assert result.skipped_confinement == 1
    assert shared.exists(), "a shared/override workspace must never be reaped"


async def test_parent_dir_path_is_never_reaped(env: dict[str, Any]) -> None:
    """A stored path that is the account root (parent of all sessions) is skipped."""
    account_root = env["root"] / "acct1"
    account_root.mkdir()
    other = _mk_workspace(env["root"], "acct1", "sess_other")
    pool = _fake_pool([_row("acct1", "sess_archived", str(account_root))])

    result = await sweep_archived_workspaces(pool)

    assert result.reaped == 0
    assert result.skipped_confinement == 1
    assert account_root.exists()
    assert other.exists(), "reaping a parent path would have destroyed a sibling session"


async def test_workspace_root_itself_is_never_reaped(env: dict[str, Any]) -> None:
    """A stored path resolving to workspace_root (even via trailing slash) is skipped.

    Defeats the catastrophic 'rmtree the whole root' case: the canonical path is
    always two levels deep, so it can never equal the root, and a stored value
    of the root (or root + trailing slash) never resolve-equals the canonical.
    """
    sibling = _mk_workspace(env["root"], "acct1", "sess_sibling")
    pool = _fake_pool([_row("acct1", "sess_archived", str(env["root"]) + "/")])

    result = await sweep_archived_workspaces(pool)

    assert result.reaped == 0
    assert result.skipped_confinement == 1
    assert env["root"].exists()
    assert sibling.exists()


async def test_out_of_tree_path_is_never_reaped(env: dict[str, Any], tmp_path: Path) -> None:
    """A stored path outside workspace_root never resolve-equals the canonical → skipped."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "precious").write_text("do not touch")
    pool = _fake_pool([_row("acct1", "sess_archived", str(outside))])

    result = await sweep_archived_workspaces(pool)

    assert result.reaped == 0
    assert result.skipped_confinement == 1
    assert (outside / "precious").exists()


async def test_relative_or_empty_path_is_never_reaped(env: dict[str, Any]) -> None:
    """Relative / empty stored paths never resolve-equal the canonical → skipped."""
    _mk_workspace(env["root"], "acct1", "sess_archived")  # canonical exists
    pool = _fake_pool(
        [
            _row("acct1", "sess_archived", ""),
            _row("acct1", "sess_archived", "workspaces/acct1/sess_archived"),
        ]
    )

    result = await sweep_archived_workspaces(pool)

    assert result.reaped == 0
    assert result.skipped_confinement == 2
    assert (env["root"] / "acct1" / "sess_archived").exists()


async def test_symlink_canonical_to_outside_is_never_followed(
    env: dict[str, Any], tmp_path: Path
) -> None:
    """A symlink AT the canonical location pointing OUTSIDE the root is skipped."""
    outside = tmp_path / "outside_target"
    outside.mkdir()
    (outside / "precious").write_text("do not touch")
    account_dir = env["root"] / "acct1"
    account_dir.mkdir()
    link = account_dir / "sess_archived"
    link.symlink_to(outside)
    # Stored path is the canonical leaf itself (the symlink) — it realpaths to
    # the target, but the is_symlink leaf-check skips it before any compare.
    pool = _fake_pool([_row("acct1", "sess_archived", str(link))])

    result = await sweep_archived_workspaces(pool)

    assert result.reaped == 0
    assert result.skipped_confinement == 1
    assert outside.exists()
    assert (outside / "precious").exists()


async def test_symlink_canonical_to_sibling_session_is_never_followed(
    env: dict[str, Any],
) -> None:
    """A symlink at the canonical leaf pointing at a LIVE SIBLING session's dir
    under the SAME root is skipped — the data-loss case the resolve-on-leaf bug
    would have caused (the symlink check must run on the UN-resolved leaf).
    """
    victim = _mk_workspace(env["root"], "acct1", "sess_victim")  # a live sibling
    account_dir = env["root"] / "acct1"
    link = account_dir / "sess_archived"  # canonical leaf of the archived session
    link.symlink_to(victim)  # same-root target — defeats an out-of-tree-only belt
    pool = _fake_pool([_row("acct1", "sess_archived", str(link))])

    result = await sweep_archived_workspaces(pool)

    assert result.reaped == 0
    assert result.skipped_confinement == 1
    assert victim.exists(), "a same-root symlink target (live sibling) must survive"
    assert (victim / "scratch.txt").exists()


async def test_symlinked_account_component_to_live_sibling_is_never_followed(
    env: dict[str, Any],
) -> None:
    """CRITICAL (#1395 never-delete breach): a symlink at the ACCOUNT component
    ``<root>/<account_id>`` pointing at a real sibling account's dir must NOT be
    followed — reaping the archived session's canonical leaf would ``rmtree`` the
    LIVE victim's session dir under the symlinked-through account.

    The leaf-only ``is_symlink()`` check is blind to a PARENT symlink, and
    ``realpath(stored) == realpath(canonical)`` collapses the parent symlink
    identically on both sides (so it matches and waves the candidate through).
    Fix 1 confines on the FULLY-RESOLVED canonical path (no symlink anywhere in
    the chain, realpath still two levels under the resolved root), so a
    parent-component symlink breaks the realpath equality ⇒ skipped, never reaped.

    Non-vacuous: against the unfixed leaf-only confinement this test FAILS —
    ``reaped == 1`` and the victim's data is deleted; with Fix 1 it PASSES.
    """
    root = env["root"]
    # The live victim: account ``acct_victim`` with a real, populated session dir.
    victim_account = root / "acct_victim"
    victim_account.mkdir()
    victim_session = victim_account / "sess_victim"
    victim_session.mkdir()
    (victim_session / "live-data.txt").write_text("a LIVE cross-account session's files")

    # The attacker account component is a SYMLINK to the victim account dir, so
    # ``<root>/acct_attacker/sess_victim`` traverses the parent symlink to the
    # victim's real session dir under the SAME root.
    attacker_account = root / "acct_attacker"
    attacker_account.symlink_to(victim_account)

    # The archived session's stored path is its canonical default through the
    # symlinked account — it realpath-collapses to the victim's real dir, exactly
    # as the canonical recomputed from ids does (the bug the two old guards miss).
    canonical_via_symlink = str(attacker_account / "sess_victim")
    pool = _fake_pool([_row("acct_attacker", "sess_victim", canonical_via_symlink)])

    result = await sweep_archived_workspaces(pool)

    assert result.reaped == 0, "a parent-component symlink must never be followed into a delete"
    assert result.skipped_confinement == 1
    assert victim_session.exists(), "the LIVE cross-account victim's session dir must remain"
    assert (victim_session / "live-data.txt").exists(), "the victim's live data must remain"


async def test_live_clone_sharing_archived_parents_canonical_dir_is_skipped(
    env: dict[str, Any],
) -> None:
    """Fix 2 (residual, same never-delete class): a LIVE session whose
    ``workspace_volume_path`` realpath-equals an archived candidate's canonical
    dir must cause the archived candidate to be SKIPPED.

    ``clone_session`` lets a live clone share another session's volume — and a
    live clone can point at an archived parent's OWN canonical default path. If
    the reaper reaped the archived parent's row it would ``rmtree`` the directory
    the live clone is actively using ⇒ cross-session live data loss. The
    live-clone keep-set cross-check skips the parent when a live session collides.

    Non-vacuous: without the keep-set cross-check the parent's canonical dir
    passes every confinement check (its OWN stored path equals its OWN canonical),
    so it is reaped (``reaped == 1``) and the live clone's volume is destroyed;
    with Fix 2 the live collision skips it (``reaped == 0``).
    """
    # The archived parent's canonical default dir (its stored path == canonical).
    parent_dir = _mk_workspace(env["root"], "acct1", "sess_parent")
    parent_canonical = str(parent_dir)

    # A LIVE clone shares the parent's directory (its stored workspace path is the
    # parent's canonical dir). It is non-archived, so it is in the live keep-set.
    pool = _fake_pool(
        [_row("acct1", "sess_parent", parent_canonical)],
        live_paths=[parent_canonical],
    )

    result = await sweep_archived_workspaces(pool)

    assert result.reaped == 0, "reaping a dir a LIVE clone shares would destroy the clone's volume"
    assert result.skipped_confinement == 1
    assert parent_dir.exists(), "the shared directory the live clone uses must remain"
    assert (parent_dir / "scratch.txt").exists()


async def test_nonterminal_shared_run_keeps_archived_launcher_workspace(
    env: dict[str, Any],
) -> None:
    """A pending/running/suspended shared run protects its archived launcher's path."""
    launcher_dir = _mk_workspace(env["root"], "acct1", "sess_launcher")
    pool = _fake_pool(
        [_row("acct1", "sess_launcher", str(launcher_dir))],
        live_paths=[str(launcher_dir)],
    )

    result = await sweep_archived_workspaces(pool)

    assert result.reaped == 0
    assert launcher_dir.exists(), "a non-terminal shared run still uses this workspace"
    sql = " ".join(call.args[0] for call in pool._conn.fetch.await_args_list)
    assert "workspace_mode = 'shared'" in sql
    assert "status IN ('pending', 'running', 'suspended')" in sql


async def test_terminal_shared_run_allows_archive_grace_reclamation(
    env: dict[str, Any],
) -> None:
    """Terminal shared runs are absent from the keep-set, so normal reaping proceeds."""
    launcher_dir = _mk_workspace(env["root"], "acct1", "sess_launcher_terminal")
    pool = _fake_pool([_row("acct1", "sess_launcher_terminal", str(launcher_dir))])

    result = await sweep_archived_workspaces(pool)

    assert result.reaped == 1
    assert not launcher_dir.exists()


async def test_nonterminal_fresh_run_does_not_keep_archived_workspace(
    env: dict[str, Any],
) -> None:
    """Fresh-workspace runs remain outside this shared-workspace keep-set."""
    launcher_dir = _mk_workspace(env["root"], "acct1", "sess_launcher_fresh")
    pool = _fake_pool([_row("acct1", "sess_launcher_fresh", str(launcher_dir))])

    result = await sweep_archived_workspaces(pool)

    assert result.reaped == 1
    assert not launcher_dir.exists()


async def test_shared_run_created_after_scan_is_caught_by_pre_delete_revalidation(
    env: dict[str, Any],
) -> None:
    """A shared run appearing after the initial scan is caught before ``rmtree``."""
    launcher_dir = _mk_workspace(env["root"], "acct1", "sess_launcher_race")
    conn = MagicMock()
    keep_reads = 0

    async def _route_fetch(sql: str, *_a: Any, **_k: Any) -> list[Any]:
        nonlocal keep_reads
        if "archived_at IS NOT NULL" in sql:
            return [_row("acct1", "sess_launcher_race", str(launcher_dir))]
        keep_reads += 1
        return [] if keep_reads == 1 else [{"workspace_volume_path": str(launcher_dir)}]

    conn.fetch = AsyncMock(side_effect=_route_fetch)
    conn.execute = AsyncMock()
    conn.fetchval = AsyncMock(return_value=True)

    class _Tx:
        async def __aenter__(self) -> None:
            return None

        async def __aexit__(self, *_a: Any) -> None:
            return None

    conn.transaction.return_value = _Tx()

    class _Cm:
        async def __aenter__(self) -> Any:
            return conn

        async def __aexit__(self, *_a: Any) -> None:
            return None

    pool = MagicMock()
    pool.acquire.return_value = _Cm()

    result = await sweep_archived_workspaces(pool)

    assert result.reaped == 0
    assert launcher_dir.exists()
    assert keep_reads == 1
    conn.fetchval.assert_awaited_once()


async def test_off_shape_ids_are_never_reaped(env: dict[str, Any]) -> None:
    """An account/session id carrying a path separator or ``..`` is skipped.

    Defense-in-depth: a corrupt/crafted id can't reshape the delete-target path.
    """
    _mk_workspace(env["root"], "acct1", "sess_archived")  # a real dir exists
    pool = _fake_pool(
        [
            _row("acct1", "../sess_archived", str(env["root"] / "acct1" / "sess_archived")),
            _row("..", "sess_archived", str(env["root"])),
            _row("acct1", "a/b", str(env["root"] / "acct1" / "a" / "b")),
        ]
    )

    result = await sweep_archived_workspaces(pool)

    assert result.reaped == 0
    assert result.skipped_confinement == 3
    assert (env["root"] / "acct1" / "sess_archived").exists()


# ── guardrail 3: min-age floor (mtime belt) ──────────────────────────────────


async def test_fresh_dir_below_mtime_floor_is_kept(env: dict[str, Any]) -> None:
    """A workspace touched more recently than the mtime floor is skipped."""
    env["settings"].workspace_reaper_min_mtime_age_seconds = 3600
    d = env["root"] / "acct1" / "sess_archived"
    d.mkdir(parents=True)  # mtime ~= now, under the 1h floor
    pool = _fake_pool([_row("acct1", "sess_archived", str(d.resolve()))])

    result = await sweep_archived_workspaces(pool)

    assert result.reaped == 0
    assert result.skipped_too_fresh == 1
    assert d.exists()


# ── guardrail 4: fail-closed ─────────────────────────────────────────────────


async def test_db_error_reaps_nothing(env: dict[str, Any]) -> None:
    """A candidate-read failure reaps NOTHING (fail-closed)."""
    d = _mk_workspace(env["root"], "acct1", "sess_archived")
    pool = _fake_pool(asyncpg.PostgresError("boom"))

    result = await sweep_archived_workspaces(pool)

    assert result.reaped == 0
    assert result.bytes_freed == 0
    assert d.exists(), "fail-closed: a DB error must never delete anything"


async def test_missing_canonical_dir_is_a_noop_skip(env: dict[str, Any]) -> None:
    """An archived session whose canonical dir is already gone is a benign skip."""
    pool = _fake_pool([_row("acct1", "sess_gone", str(env["root"] / "acct1" / "sess_gone"))])

    result = await sweep_archived_workspaces(pool)

    assert result.reaped == 0
    assert result.skipped_missing == 1


# ── guardrail 5: kill-switch ─────────────────────────────────────────────────


async def test_kill_switch_disables_everything(env: dict[str, Any]) -> None:
    """``workspace_reaper_enabled=False`` deletes nothing and never queries the DB."""
    env["settings"].workspace_reaper_enabled = False
    d = _mk_workspace(env["root"], "acct1", "sess_archived")
    pool = _fake_pool([_row("acct1", "sess_archived", str(d))])

    result = await sweep_archived_workspaces(pool)

    assert result.reaped == 0
    assert d.exists()
    pool._conn.fetch.assert_not_awaited()  # disabled ⇒ no DB query at all


async def test_default_settings_ship_dark() -> None:
    """The shipped default is OFF (dark) — this reaper deletes real working files."""
    from aios.config import Settings

    field = Settings.model_fields["workspace_reaper_enabled"]
    assert field.default is False, "workspace reaper must ship disabled (review-gated)"


# ── guardrail 6: dry-run + observability ─────────────────────────────────────


async def test_dry_run_deletes_nothing_but_reports(env: dict[str, Any]) -> None:
    """Dry-run logs would-reap counts + reclaimable bytes but deletes nothing."""
    env["settings"].workspace_reaper_dry_run = True
    d = _mk_workspace(env["root"], "acct1", "sess_archived")
    pool = _fake_pool([_row("acct1", "sess_archived", str(d))])

    result = await sweep_archived_workspaces(pool)

    assert result.dry_run is True
    assert result.reaped == 1, "dry-run still COUNTS what it would reap"
    assert result.bytes_freed > 0, "dry-run still reports reclaimable bytes"
    assert d.exists(), "dry-run must delete nothing"


async def test_structured_counters_partition_candidates(
    env: dict[str, Any], tmp_path: Path
) -> None:
    """Every candidate lands in exactly one bucket: reaped / confinement / missing / fresh."""
    reaped = _mk_workspace(env["root"], "acct1", "sess_reap")
    outside = tmp_path / "outside"
    outside.mkdir()
    rows = [
        _row("acct1", "sess_reap", str(reaped)),  # → reaped
        _row("acct1", "sess_override", str(outside)),  # → confinement
        _row("acct1", "sess_gone", str(env["root"] / "acct1" / "sess_gone")),  # → missing
    ]
    pool = _fake_pool(rows)

    result = await sweep_archived_workspaces(pool)

    total = (
        result.reaped
        + result.skipped_confinement
        + result.skipped_missing
        + result.skipped_too_fresh
        + result.skipped_error
    )
    assert total == len(rows)
    assert result.reaped == 1
    assert result.skipped_confinement == 1
    assert result.skipped_missing == 1


async def test_rmtree_failure_counts_as_error_not_missing(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ``rmtree`` failure is recorded as ``skipped_error`` (distinct from missing)
    and does not abort the sweep — a second valid candidate is still reaped.
    """
    import shutil as _shutil

    bad = _mk_workspace(env["root"], "acct1", "sess_bad")
    good = _mk_workspace(env["root"], "acct1", "sess_good")

    real_rmtree = _shutil.rmtree

    def _flaky_rmtree(path: Any, *a: Any, **k: Any) -> None:
        if str(path) == str(bad):
            raise OSError("read-only filesystem")
        real_rmtree(path, *a, **k)

    monkeypatch.setattr(_shutil, "rmtree", _flaky_rmtree)
    pool = _fake_pool([_row("acct1", "sess_bad", str(bad)), _row("acct1", "sess_good", str(good))])

    result = await sweep_archived_workspaces(pool)

    assert result.skipped_error == 1
    assert result.skipped_missing == 0
    assert result.reaped == 1, "a failure on one dir must not abort the sweep"
    assert bad.exists(), "the un-removable dir is left for the next sweep"
    assert not good.exists()
