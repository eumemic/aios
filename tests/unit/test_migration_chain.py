"""Verify the on-disk alembic migration chain is linear with a single head.

These tests guard against the production failure where the DB is stamped
``alembic_version = 0055`` but the codebase lacked a resolvable ``0055``
revision (added by #603, removed by the #613/#615 stop-hook revert). With
``0055`` missing, ``aios migrate`` cannot resolve the stamped revision and
every deploy fails.

Pure-Python: reads ``migrations/`` via ``ScriptDirectory``; no DB, no Docker.
"""

from __future__ import annotations

from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory

_MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


def _script_directory() -> ScriptDirectory:
    cfg = Config()
    cfg.set_main_option("script_location", str(_MIGRATIONS_DIR))
    return ScriptDirectory.from_config(cfg)


def test_single_head() -> None:
    """The migration ladder has exactly one head: ``0056``."""
    script = _script_directory()
    assert script.get_heads() == ["0056"]


def test_chain_is_linear_0054_to_0056() -> None:
    """``0054 -> 0055 -> 0056`` is a plain linear chain, no merges/branches."""
    script = _script_directory()

    rev_0056 = script.get_revision("0056")
    rev_0055 = script.get_revision("0055")

    assert rev_0056.down_revision == "0055"
    assert rev_0055.down_revision == "0054"

    # A tuple/list down_revision would mean a merge/branch point.
    assert isinstance(rev_0056.down_revision, str)
    assert isinstance(rev_0055.down_revision, str)


def test_revision_0055_resolvable() -> None:
    """``0055`` resolves — the precise assertion reproducing the prod failure.

    Production DBs are stamped ``alembic_version = 0055``; if the revision is
    missing, ``alembic upgrade`` raises and every deploy fails.
    """
    script = _script_directory()
    revision = script.get_revision("0055")
    assert revision is not None
