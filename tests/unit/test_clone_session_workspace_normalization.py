"""Unit tests for the clone_session explicit-path normalization (reaper recheck).

``services.clone_session``'s explicit ``workspace_path`` arm MUST store the
realpath-normalized form, matching the sibling shared-path writers
(``create_session``'s shared child, ``create_run``'s shared arm). The workspace
reaper's under-lock recheck (``unscoped_workspace_path_is_live``) compares an
``os.path.realpath`` probe against the stored column in SQL; a stored literal
whose form differs from its realpath (a ``..`` segment, or an
``AIOS_WORKSPACE_ROOT`` crossing a symlink ancestor) defeats every clause and
lets the reaper delete a directory a live clone references.

These tests pin the normalization at the service layer (the API entry point)
without the Postgres testcontainer: the raw input first passes
``validate_workspace_path`` (so relative / out-of-jail rejection is preserved),
then is rewritten through ``normalized_workspace_path`` before being handed to
``queries.clone_session``. The DB-backed recheck behavior is covered by the
integration test ``tests/integration/test_clone_session_workspace_normalization.py``.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aios.db import queries
from aios.services import sessions as sessions_service


def _symlinked_root(tmp_path: Path) -> tuple[Path, str, str]:
    """Build a workspace_root whose literal crosses a symlink ancestor.

    Returns ``(literal_root, literal_shared, real_shared)`` where
    ``literal_shared != real_shared`` — the realpath divergence the recheck
    relies on shared-path writers to close by storing ``real_shared``.
    Mirrors the realistic vector from the bug report (``AIOS_WORKSPACE_ROOT``
    whose literal form crosses a symlink ancestor, e.g. ``/var/lib`` ->
    ``/mnt/data/lib``).
    """
    real_root = tmp_path / "data" / "aios" / "workspaces"
    account_dir = real_root / "acc"
    account_dir.mkdir(parents=True)
    (account_dir / "P").mkdir()
    var_dir = tmp_path / "var"
    var_dir.mkdir()
    os.symlink(tmp_path / "data", var_dir / "lib")
    literal_root = var_dir / "lib" / "aios" / "workspaces"
    literal_shared = str(literal_root / "acc" / "P")
    real_shared = os.path.realpath(literal_shared)
    assert literal_shared != real_shared, "setup must produce a realpath divergence"
    return literal_root, literal_shared, real_shared


@pytest.fixture
def symlink_ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Symlinked workspace_root with get_settings patched at both import sites."""
    root, literal_shared, real_shared = _symlinked_root(tmp_path)
    settings = SimpleNamespace(workspace_root=root)
    monkeypatch.setattr(sessions_service, "get_settings", lambda: settings)
    monkeypatch.setattr("aios.sandbox.volumes.get_settings", lambda: settings)
    return {"literal": literal_shared, "real": real_shared, "settings": settings}


def _fake_pool() -> MagicMock:
    """Stand-in pool whose ``async with pool.acquire() as conn`` yields a mock conn."""
    conn = MagicMock()
    conn.execute = AsyncMock()
    tx = MagicMock()
    tx.__aenter__ = AsyncMock(return_value=None)
    tx.__aexit__ = AsyncMock(return_value=None)
    conn.transaction.return_value = tx
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=None)
    pool = MagicMock()
    pool.acquire.return_value = cm
    return pool


def _patch_clone_leaves(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Patch the DB-touching leaves so only validate + normalize run for real.

    Returns a dict capturing the ``workspace_path`` kwarg handed to
    ``queries.clone_session`` — the load-bearing assertion target.
    """
    captured: dict[str, Any] = {}

    async def _fake_clone(
        conn: Any, parent: str, *, account_id: str, workspace_path: str | None
    ) -> MagicMock:
        captured["workspace_path"] = workspace_path
        return MagicMock(name="clone_session")

    monkeypatch.setattr(queries, "clone_session", _fake_clone)
    monkeypatch.setattr(queries, "acquire_workspace_hierarchy_advisory_xact_locks", AsyncMock())
    monkeypatch.setattr(sessions_service, "_enrich_session", AsyncMock())
    return captured


async def test_clone_session_stores_realpath_normalized_symlink_form(
    symlink_ws: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A symlink-crossing explicit path is stored as its realpath form, not the literal."""
    captured = _patch_clone_leaves(monkeypatch)

    await sessions_service.clone_session(
        _fake_pool(),
        "sess_parent",
        workspace_path=symlink_ws["literal"],
        account_id="acc",
    )

    assert captured["workspace_path"] == symlink_ws["real"]
    assert captured["workspace_path"] != symlink_ws["literal"]


async def test_clone_session_explicit_arm_validates_before_normalizing(
    symlink_ws: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An out-of-jail explicit path is rejected before any normalization/store.

    ``validate_workspace_path`` runs first on the raw literal (preserving the
    relative / out-of-jail rejection contract), so ``queries.clone_session`` is
    never reached for an escaping path.
    """
    captured = _patch_clone_leaves(monkeypatch)
    outside = symlink_ws["settings"].workspace_root.parent / "escape"
    outside.mkdir(parents=True)

    from aios.errors import ForbiddenError

    with pytest.raises(ForbiddenError):
        await sessions_service.clone_session(
            _fake_pool(),
            "sess_parent",
            workspace_path=str(outside),
            account_id="acc",
        )
    assert "workspace_path" not in captured, "an out-of-jail path must not reach the store"


async def test_clone_session_default_arm_is_unaffected(
    symlink_ws: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """workspace_path=None bypasses normalization and passes None to the query layer."""
    captured = _patch_clone_leaves(monkeypatch)
    monkeypatch.setattr(
        queries,
        "get_session_workspace_path",
        AsyncMock(return_value=symlink_ws["real"]),
    )

    await sessions_service.clone_session(
        _fake_pool(),
        "sess_parent",
        workspace_path=None,
        account_id="acc",
    )

    assert captured["workspace_path"] is None
