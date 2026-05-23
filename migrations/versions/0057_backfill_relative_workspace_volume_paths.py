"""Backfill legacy relative ``sessions.workspace_volume_path`` rows.

Pre-#626 ``AIOS_WORKSPACE_ROOT`` could be configured as a relative path
(e.g. ``./workspaces`` in a dev ``.env``).  ``insert_session`` then stored
``str(workspace_root / account_id / session_id)`` on the row — which, for
a relative ``workspace_root``, yields a relative ``workspaces/<acc>/<sess>``
string.  At sandbox provisioning time ``build_spec_from_session`` re-runs
``validate_workspace_path`` against the stored value; ``Path.resolve()``
resolves the relative input against the worker process's current working
directory, the result lands outside the workspace jail, and every tool
call surfaces ``ForbiddenError`` blamed on whatever path the model just
tried to read or write (see issue #626).

The companion ``_require_absolute_workspace_root`` config validator and
``validate_workspace_path`` absolute-path guard prevent new rows from
landing in this shape; this migration rewrites the existing legacy rows.

Idempotent via the ``NOT LIKE '/%'`` filter — a re-upgrade after the
backfill has already run finds zero matching rows and does nothing.

Operator requirement: ``AIOS_WORKSPACE_ROOT`` must be set and absolute
in the environment when ``alembic upgrade`` runs.  Fail-loud: a missing
or relative value aborts the migration before any UPDATE runs, so the
operator can't silently re-introduce relative paths.

Downgrade reverses the rewrite by stripping the prepended prefix.  This
exists so the migration is symmetric; the resulting relative paths would
fail at provision time per the volumes.py guard, so callers should not
downgrade across this revision on a live system.

Revision ID: 0057
Revises: 0056
Create Date: 2026-05-22
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import PurePosixPath

from alembic import op

revision: str = "0057"
down_revision: str = "0056"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _workspace_root_prefix() -> str:
    """Return ``AIOS_WORKSPACE_ROOT`` normalized to a no-trailing-slash form.

    Asserts the env var is set and absolute — anything else would either
    silently corrupt the rewrite or leave us right back where #626 found
    us.  ``PurePosixPath`` so the migration's text manipulation matches
    the POSIX path strings stored on the row regardless of host OS.
    """
    raw = os.environ.get("AIOS_WORKSPACE_ROOT")
    if not raw:
        raise RuntimeError(
            "AIOS_WORKSPACE_ROOT must be set when running migration 0057 — "
            "the migration needs the operator's workspace_root to prepend to "
            "legacy relative paths"
        )
    path = PurePosixPath(raw)
    if not path.is_absolute():
        raise RuntimeError(
            f"AIOS_WORKSPACE_ROOT must be an absolute path; got '{raw}'. "
            "Migration 0057 prepends this prefix to relative session paths; "
            "a relative prefix would just re-create the bug it's fixing."
        )
    # Trailing slashes break the simple ``prefix || '/' || path`` form
    # below; normalize them away with ``PurePosixPath.as_posix()``.
    return path.as_posix()


def upgrade() -> None:
    prefix = _workspace_root_prefix()
    # Idempotent guard: ``NOT LIKE '/%'`` skips absolute rows on a
    # re-upgrade.  ``LIKE 'workspaces/%'`` narrows to the legacy
    # ``insert_session`` shape so an unrelated relative string a future
    # caller might stash (e.g. a future feature using
    # ``workspace_volume_path`` for something else) isn't accidentally
    # absolutized.
    op.execute(
        f"""
        UPDATE sessions
           SET workspace_volume_path = '{prefix}' || '/' || workspace_volume_path
         WHERE workspace_volume_path NOT LIKE '/%'
           AND workspace_volume_path LIKE 'workspaces/%'
        """
    )


def downgrade() -> None:
    prefix = _workspace_root_prefix()
    # Strip the prepended prefix.  Only touch rows that match the
    # post-upgrade shape (``{prefix}/workspaces/...``) so an absolute
    # path that was already absolute pre-upgrade stays put.
    op.execute(
        f"""
        UPDATE sessions
           SET workspace_volume_path = SUBSTRING(
                   workspace_volume_path FROM CHAR_LENGTH('{prefix}/') + 1
               )
         WHERE workspace_volume_path LIKE '{prefix}/workspaces/%'
        """
    )
