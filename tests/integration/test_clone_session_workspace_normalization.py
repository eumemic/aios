"""Integration: clone_session explicit-path normalization vs. the reaper recheck.

Verifies the fix for the workspace-reaper recheck stored-form mismatch. The
reaper's under-lock recheck (``unscoped_workspace_path_is_live``) compares an
``os.path.realpath`` probe against the stored ``workspace_volume_path`` column
in SQL; it can only match a live clone sharing another session's directory when
that stored value is itself realpath-normalized. The service-layer
``clone_session`` MUST therefore store the realpath-normalized explicit
``workspace_path``, matching the sibling shared-path writers.

Covers the realistic vector from the bug report: an ``AIOS_WORKSPACE_ROOT``
whose literal crosses a symlink ancestor (``/var/lib`` -> ``/mnt/data/lib``),
so the archived candidate's default-stored path and a verbatim explicit clone
arg differ from their realpath form. With the fix, the clone stores the realpath
form and the recheck catches it; without it (storing the literal), the recheck
misses the live clone and the reaper would ``rmtree`` a directory it references.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import asyncpg
import pytest

from aios.config import get_settings
from aios.db import queries
from aios.db.pool import create_pool
from aios.harness import workspace_reaper
from aios.harness.workspace_reaper import sweep_archived_workspaces
from aios.ids import SESSION, make_id
from aios.services import sessions as sessions_service

pytestmark = pytest.mark.integration

ACCOUNT = "acc_norm"
AGENT = "agent_norm"
ENV = "env_norm"


def _symlinked_root(tmp_path: Path) -> tuple[Path, str, str, str, Path]:
    """Build a workspace_root whose literal crosses a symlink ancestor.

    Returns ``(literal_root, candidate_id, literal_shared, real_shared,
    shared_dir)`` where ``literal_shared != real_shared`` (the realpath
    divergence) and ``str(shared_dir) == real_shared``. ``real_shared`` is the
    archived candidate's canonical default dir, which the live clone shares.
    Mirrors the bug report's ``/var/lib`` -> ``/mnt/data/lib`` vector.

    ``shared_dir`` (the Path) is returned so async tests can call its methods
    without tripping ASYNC240 — the Path is constructed in this sync helper, so
    ruff does not flag downstream ``shared_dir.exists()`` (mirrors the reaper
    unit tests' ``_mk_workspace`` convention).
    """
    real_root = tmp_path / "data" / "aios" / "workspaces"
    account_dir = real_root / ACCOUNT
    account_dir.mkdir(parents=True)
    candidate_id = "sess_archived"
    shared_dir = account_dir / candidate_id
    shared_dir.mkdir()
    (shared_dir / "old-file").write_text("pre-archive")
    old = time.time() - 2 * 24 * 3600
    os.utime(shared_dir, (old, old))
    os.utime(shared_dir / "old-file", (old, old))
    var_dir = tmp_path / "var"
    var_dir.mkdir()
    os.symlink(tmp_path / "data", var_dir / "lib")
    literal_root = var_dir / "lib" / "aios" / "workspaces"
    literal_shared = str(literal_root / ACCOUNT / candidate_id)
    real_shared = os.path.realpath(literal_shared)
    assert literal_shared != real_shared, "setup must produce a realpath divergence"
    assert str(shared_dir) == real_shared
    return literal_root, candidate_id, literal_shared, real_shared, shared_dir


async def _seed_base(conn: asyncpg.Connection[Any]) -> None:
    await conn.execute(
        "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
        "VALUES ($1, NULL, TRUE, 'norm')",
        ACCOUNT,
    )
    await conn.execute(
        "INSERT INTO agents (id, account_id, name, model, system, version) "
        "VALUES ($1, $2, 'norm', 'test/model', '', 1)",
        AGENT,
        ACCOUNT,
    )
    await conn.execute(
        "INSERT INTO environments (id, account_id, name) VALUES ($1, $2, 'norm')",
        ENV,
        ACCOUNT,
    )


async def _insert_live_session(
    conn: asyncpg.Connection[Any], session_id: str, workspace_path: str
) -> None:
    """Minimal live (non-archived) session row — the keep-set / recheck input."""
    await conn.execute(
        """
        INSERT INTO sessions (
            id, agent_id, environment_id, agent_version, title, metadata,
            workspace_volume_path, env, account_id, last_event_seq)
        VALUES ($1, $2, $3, 1, 'live', '{}'::jsonb, $4, '{}'::jsonb, $5, 0)
        """,
        session_id,
        AGENT,
        ENV,
        workspace_path,
        ACCOUNT,
    )


async def _insert_clonable_parent(conn: asyncpg.Connection[Any], parent_id: str) -> str:
    """A live idle parent proven clonable (mirrors test_clone_policy_arms)."""
    await conn.execute(
        "INSERT INTO wf_runs (id, account_id, script, script_sha, environment_id, "
        "host_semantics_epoch) VALUES ('run_parent', $1, 's', 'sha', $2, 0)",
        ACCOUNT,
        ENV,
    )
    await conn.execute(
        """
        INSERT INTO sessions (
            id, agent_id, environment_id, agent_version, title, metadata,
            workspace_volume_path, env, focal_channel, focal_locked, account_id,
            last_event_seq, input_tokens, output_tokens, cost_microusd,
            parent_run_id, origin, surface_frozen, model, litellm_extra,
            tools, mcp_servers, http_servers, snapshot_ref, outbound_suppression,
            archive_when_idle)
        VALUES ($1, $2, $3, 1, 'parent', '{}'::jsonb, '/w/parent', '{}'::jsonb,
            'tg/c1', TRUE, $4, 2, 100, 50, 4242, 'run_parent', 'background', TRUE,
            'test/model', '{}'::jsonb, '[]'::jsonb, '[]'::jsonb, '[]'::jsonb,
            NULL, 'on', TRUE)
        """,
        parent_id,
        AGENT,
        ENV,
        ACCOUNT,
    )
    return parent_id


async def _insert_archived_candidate(
    conn: asyncpg.Connection[Any], session_id: str, workspace_path: str
) -> None:
    await conn.execute(
        """
        INSERT INTO sessions (
            id, agent_id, environment_id, agent_version, title, metadata,
            workspace_volume_path, env, account_id, last_event_seq, archived_at)
        VALUES ($1, $2, $3, 1, 'archived', '{}'::jsonb, $4, '{}'::jsonb, $5, 0,
                now() - interval '2 days')
        """,
        session_id,
        AGENT,
        ENV,
        workspace_path,
        ACCOUNT,
    )


async def test_recheck_matches_realpath_stored_form_not_literal(
    migrated_db_url: str, _reset_db_state: None, tmp_path: Path
) -> None:
    """The recheck's stored-form contract: a normalized stored path matches a
    realpath probe; a literal form (bug shape) whose form differs from its
    realpath does NOT — exactly the miss the clone fix must close."""
    _, _, literal_shared, real_shared = _symlinked_root(tmp_path)[:4]
    conn = await asyncpg.connect(migrated_db_url)
    try:
        await _seed_base(conn)
        await _insert_live_session(conn, "sess_live_norm", real_shared)

        # A live row storing the REALPATH form is caught by the realpath probe.
        assert await queries.unscoped_workspace_path_is_live(conn, real_shared) is True
        # The probe normalizes either form to the same realpath, so passing the
        # literal to the recheck still resolves to the same (caught) answer.
        assert await queries.unscoped_workspace_path_is_live(conn, literal_shared) is True

        # Simulate the pre-fix verbatim storage: the live row now stores the LITERAL
        # form (whose string differs from its realpath). The recheck's clauses all
        # fail against the realpath probe → the false negative that lets the reaper
        # delete a directory a live clone references.
        await conn.execute(
            "UPDATE sessions SET workspace_volume_path = $1 WHERE id = $2",
            literal_shared,
            "sess_live_norm",
        )
        assert await queries.unscoped_workspace_path_is_live(conn, real_shared) is False
    finally:
        await conn.close()


def _reaper_settings(literal_root: Path, migrated_db_url: str, *, dry_run: bool) -> Any:
    """A settings stub targeted at the migrated testcontainer DB.

    Models ``test_sweep_uses_real_pool_locking_path``: the reaper's dedicated
    lock backend dials ``settings.db_url`` directly, so point it at the
    per-testcontainer database rather than the parse-only ``AIOS_DB_URL``.
    """
    settings = get_settings().model_copy(deep=True)
    settings.db_url = migrated_db_url
    settings.workspace_root = literal_root
    settings.workspace_reaper_enabled = True
    settings.workspace_reaper_dry_run = dry_run
    # Drop both floors to 0 so the freshly-seeded candidate (old files) is
    # eligible; the mtime floor is a real defense once the clone writes, but
    # the vulnerable window is exactly the committed-but-not-written sub-case.
    settings.workspace_reaper_min_archived_age_seconds = 0
    settings.workspace_reaper_min_mtime_age_seconds = 0
    return settings


def _make_settings_getter(settings: Any) -> Any:
    """Bind ``settings`` at definition time (avoids B023 loop-variable capture)."""
    return lambda: settings


async def test_full_sweep_keeps_live_clone_shared_dir_with_symlinked_root(
    migrated_db_url: str,
    _reset_db_state: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full reaper control flow (lock → recheck → rmtree) against a testcontainer
    DB with a symlink-ancestor ``workspace_root``: the live clone sharing the
    archived candidate's canonical dir survives both a dry-run and a real sweep.

    Automates the manual end-to-end scenario from the test plan (reaper enabled,
    symlinked ``AIOS_WORKSPACE_ROOT``, archived P + live clone sharing P's
    literal path, committed-but-not-written). Pre-fix the verbatim clone arg
    would defeat the recheck's stored-form match and the reaper would ``rmtree``
    the shared dir; post-fix the normalized stored path is caught and the dir
    survives.
    """
    literal_root, candidate_id, literal_shared, real_shared, shared_dir = _symlinked_root(tmp_path)
    # Patch the GLOBAL settings singleton's workspace_root so
    # ``validate_workspace_path`` (called by ``services.clone_session``) resolves
    # the symlinked literal against the symlinked root, not the parse-only
    # ``AIOS_WORKSPACE_ROOT``. The reaper separately gets its own copy with
    # ``db_url`` + reaper flags via ``_reaper_settings`` below.
    monkeypatch.setattr(get_settings(), "workspace_root", literal_root)
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await _seed_base(conn)
            await _insert_archived_candidate(conn, candidate_id, literal_shared)
            parent_id = make_id(SESSION)
            await _insert_clonable_parent(conn, parent_id)

        clone = await sessions_service.clone_session(
            pool, parent_id, workspace_path=literal_shared, account_id=ACCOUNT
        )

        async with pool.acquire() as conn:
            stored = await conn.fetchval(
                "SELECT workspace_volume_path FROM sessions WHERE id = $1", clone.id
            )
            assert stored == real_shared
        assert shared_dir.exists(), "shared dir present pre-sweep"

        for dry_run in (True, False):
            settings = _reaper_settings(literal_root, migrated_db_url, dry_run=dry_run)
            monkeypatch.setattr(workspace_reaper, "get_settings", _make_settings_getter(settings))
            result = await sweep_archived_workspaces(pool)
            assert result.reaped == 0, (
                f"dry_run={dry_run}: reaper must not delete a live clone's shared dir"
            )
            assert shared_dir.exists(), (
                f"dry_run={dry_run}: the shared directory must survive the sweep"
            )
            assert (shared_dir / "old-file").read_text() == "pre-archive"
    finally:
        await pool.close()
