"""Regression: a failed github clone must NOT shift the mount-snapshot
drift key (issue #1725).

Mechanism of the bug this pins:

- ``_materialize_github_clones`` drops any repo whose clone fails from the
  ``materialized`` list, and ``build_spec_from_session`` used to stamp the
  provision-time ``mount_snapshot`` from that *filtered* list.
- Every step, ``refresh_session_mount_state`` recomputes the snapshot from
  the FULL DB echo set (``list_session_github_repo_echoes`` knows nothing
  about clone outcomes) and releases the sandbox if the snapshot drifted.
- A *permanent* clone failure (e.g. a dead embedded token) makes
  ``frozenset(materialized-only) != frozenset(all-attached)`` FOREVER, so
  the session churns provision↔release on every step indefinitely
  (~13s/step tax, all day).

The fix stamps the provision-time snapshot from the ATTEMPTED (all-attached)
echo set, so a failed clone leaves the drift key untouched. The mount list
still only carries successfully-cloned repos.
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from unittest.mock import AsyncMock

from aios.db.queries import EnvVarCredentialEcho
from aios.models.github_repositories import GithubRepositoryResourceEcho
from aios.sandbox.registry import mount_snapshot_from_echoes
from aios.sandbox.spec import build_spec_from_session
from tests.helpers.sandbox import patch_build_spec_deps

_NOW = datetime(2026, 7, 7, tzinfo=UTC)


def _echo(repo_id: str, mount_path: str) -> GithubRepositoryResourceEcho:
    return GithubRepositoryResourceEcho(
        id=repo_id,
        url=f"https://github.com/acme/{repo_id}.git",
        mount_path=mount_path,
        created_at=_NOW,
        updated_at=_NOW,
    )


async def _build_snapshot(
    *, materialized: list[GithubRepositoryResourceEcho], attempted: list[GithubRepositoryResourceEcho]
) -> frozenset[tuple[str, ...]]:
    clones = AsyncMock(return_value=(materialized, attempted, None))
    with contextlib.ExitStack() as stack:
        for ctx in patch_build_spec_deps(github_clones=clones):
            stack.enter_context(ctx)
        plan = await build_spec_from_session("sess_01TEST")
    return plan.spec.mount_snapshot


async def test_failed_clone_does_not_shrink_drift_key() -> None:
    """The provision-time snapshot must fold EVERY attached repo, even one
    whose clone failed — so it equals the step-time DB echo set and the
    drift detector doesn't release on every step (#1725)."""
    ok = _echo("ok", "/workspace/ok")
    dead = _echo("dead", "/workspace/dead")  # permanently-failing clone

    # Provision: only ``ok`` materialized; both were attempted.
    provision_snapshot = await _build_snapshot(materialized=[ok], attempted=[ok, dead])

    # Step-time: ``refresh_session_mount_state`` folds the FULL DB echo set
    # (all attached repos — it can't see clone outcomes).
    step_snapshot = mount_snapshot_from_echoes([], [ok, dead], [])

    assert provision_snapshot == step_snapshot


async def test_all_clones_succeed_snapshot_unchanged() -> None:
    """When every clone succeeds, attempted == materialized and the snapshot
    is exactly the DB echo set — the fix is a no-op for the happy path."""
    ok = _echo("ok", "/workspace/ok")
    provision_snapshot = await _build_snapshot(materialized=[ok], attempted=[ok])
    step_snapshot = mount_snapshot_from_echoes([], [ok], [])
    assert provision_snapshot == step_snapshot


async def test_failed_clone_snapshot_folds_dead_repo_element() -> None:
    """Concretely: the dead repo's ``(GITHUB_REPOSITORY, id, mount, ts)``
    element is present in the provision snapshot despite the clone failing."""
    from aios.ids import GITHUB_REPOSITORY

    ok = _echo("ok", "/workspace/ok")
    dead = _echo("dead", "/workspace/dead")
    provision_snapshot = await _build_snapshot(materialized=[ok], attempted=[ok, dead])
    assert (
        GITHUB_REPOSITORY,
        dead.id,
        dead.mount_path,
        dead.updated_at.isoformat(),
    ) in provision_snapshot


def test_env_var_echo_import_smoke() -> None:
    """Guard: the step-side fold takes ``EnvVarCredentialEcho`` — ensure the
    import path used by the drift probe stays valid alongside this fix."""
    assert EnvVarCredentialEcho is not None
