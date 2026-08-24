"""In-process alembic runner for the ``test_migrations_*`` modules.

Mirrors the CLI contract the migration tests are written against —
``returncode`` 0/1, captured ``stdout``, failure text in ``stderr`` — while
skipping the ``uv run`` + interpreter + import cost every invocation used to
pay as a subprocess (~1 s x ~150 call sites).  A migration that raises (e.g.
a guarded downgrade) surfaces as ``returncode`` 1 with the traceback in
``stderr``, which is where the tests grep for messages like
``"cannot downgrade"``.
"""

from __future__ import annotations

import io
import os
import subprocess
import traceback
from unittest import mock

from aios.db.migrations import alembic_config


def run_alembic(
    args: list[str], db_url: str, *, extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """In-process ``alembic upgrade|downgrade <revision>`` against ``db_url``.

    ``extra_env`` is for migrations that read configuration from the
    environment (0057 reads ``AIOS_WORKSPACE_ROOT``; 0154 reads
    ``AIOS_VAULT_KEY``).
    """
    from alembic import command

    action, revision = args
    runner = {"upgrade": command.upgrade, "downgrade": command.downgrade}[action]
    out = io.StringIO()
    cfg = alembic_config(stdout=out)
    # ``migrations/env.py`` reads the URL from ``AIOS_DB_URL``.
    with mock.patch.dict(os.environ, {"AIOS_DB_URL": db_url, **(extra_env or {})}):
        try:
            runner(cfg, revision)
        except Exception:
            return subprocess.CompletedProcess(
                ["alembic", *args],
                returncode=1,
                stdout=out.getvalue(),
                stderr=traceback.format_exc(),
            )
    return subprocess.CompletedProcess(
        ["alembic", *args], returncode=0, stdout=out.getvalue(), stderr=""
    )
