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
from sqlalchemy import text

revision: str = "0057"
down_revision: str = "0056"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Legacy ``insert_session`` stored ``str(workspace_root / acc / sess)`` where
# ``workspace_root`` was the relative string ``'workspaces'``; every legacy
# row therefore begins with this literal segment.  Backfill = strip this
# leading component and replace with the absolute ``AIOS_WORKSPACE_ROOT``.
_LEGACY_PREFIX = "workspaces"


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
    # Skip the env-var requirement when there's nothing to backfill.
    # Fresh-DB test fixtures (no legacy rows) and prod re-runs (already
    # backfilled) both hit this path — the migration becomes a no-op
    # without forcing the operator to thread AIOS_WORKSPACE_ROOT through
    # contexts where it's irrelevant. When real legacy rows exist,
    # _workspace_root_prefix() still fail-louds on missing/relative env.
    bind = op.get_bind()
    legacy_count = bind.execute(
        text(
            """
            SELECT COUNT(*) FROM sessions
             WHERE workspace_volume_path NOT LIKE '/%'
               AND workspace_volume_path LIKE :legacy_pattern
            """
        ).bindparams(legacy_pattern=f"{_LEGACY_PREFIX}/%")
    ).scalar()
    if legacy_count == 0:
        return

    prefix = _workspace_root_prefix()
    # Replace the leading ``workspaces`` component with the absolute prefix
    # so ``workspaces/<acc>/<sess>`` becomes ``<prefix>/<acc>/<sess>`` —
    # exactly what ``str(workspace_root_absolute / acc / sess)`` would have
    # produced if ``AIOS_WORKSPACE_ROOT`` had been absolute at insert time.
    #
    # Idempotent guard: ``NOT LIKE '/%'`` skips absolute rows on a
    # re-upgrade.  ``LIKE 'workspaces/%'`` narrows to the legacy
    # ``insert_session`` shape so an unrelated relative string a future
    # caller might stash (e.g. a future feature using
    # ``workspace_volume_path`` for something else) isn't accidentally
    # absolutized.
    #
    # ``text(...).bindparams`` keeps ``prefix`` out of the SQL string so an
    # exotic ``AIOS_WORKSPACE_ROOT`` (quote chars, etc.) can't break parsing
    # or inject SQL.  ``_workspace_root_prefix`` validates the value is set
    # and absolute but does not escape SQL metacharacters.
    op.execute(
        text(
            """
            UPDATE sessions
               SET workspace_volume_path =
                       :prefix
                       || SUBSTRING(workspace_volume_path FROM :legacy_len + 1)
             WHERE workspace_volume_path NOT LIKE '/%'
               AND workspace_volume_path LIKE :legacy_pattern
            """
        ).bindparams(
            prefix=prefix,
            legacy_len=len(_LEGACY_PREFIX),
            legacy_pattern=f"{_LEGACY_PREFIX}/%",
        )
    )


def downgrade() -> None:
    prefix = _workspace_root_prefix()
    # Reverse the upgrade: strip the absolute prefix and prepend the
    # legacy ``workspaces`` segment so ``<prefix>/<acc>/<sess>`` returns to
    # ``workspaces/<acc>/<sess>``.  The resulting relative path would fail
    # at provision time per the volumes.py guard, so callers should not
    # downgrade across this revision on a live system — this exists only
    # so the migration is symmetric.
    #
    # ``LIKE :prefix_pattern`` matches every row sitting under the
    # configured workspace root.  Pre-upgrade absolute rows that happened
    # to live under the same prefix would also be rewritten here, but
    # downgrading is already destructive (turns absolute paths back into
    # relative ones), so the asymmetry is acceptable.
    #
    # ``AIOS_WORKSPACE_ROOT`` is operator-supplied and may contain LIKE
    # metacharacters (``_``, ``%``, or ``\\``); escape them so the pattern
    # matches the literal prefix rather than over-matching.  The companion
    # ``ESCAPE '\\'`` clause tells Postgres the backslash is the escape
    # character (otherwise its default behavior is implementation-defined
    # and surprising to read).
    escaped_prefix = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    op.execute(
        text(
            r"""
            UPDATE sessions
               SET workspace_volume_path =
                       :legacy_prefix
                       || SUBSTRING(workspace_volume_path FROM :prefix_len + 1)
             WHERE workspace_volume_path LIKE :prefix_pattern ESCAPE '\'
            """
        ).bindparams(
            legacy_prefix=_LEGACY_PREFIX,
            prefix_len=len(prefix),
            prefix_pattern=f"{escaped_prefix}/%",
        )
    )
