"""CLI URL resolution — re-export shim.

The canonical implementation lives in :mod:`aios_sdk.config` so the SDK
package is import-root-clean (no transitive ``aios.cli`` pull-in). The
CLI reads ``AIOS_URL`` / ``AIOS_API_KEY`` / ``AIOS_API_PORT`` from the
process environment via typer's ``envvar=`` — there is deliberately no
second config surface. For ``.env`` loading, callers run
``set -a; source .env; set +a`` before invoking ``aios``, the same
pattern used for the operator commands (alembic, procrastinate).
"""

from __future__ import annotations

from aios_sdk.config import resolve_base_url

__all__ = ["resolve_base_url"]
