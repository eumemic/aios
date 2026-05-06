"""Drift check: ``openapi.json`` must match what ``create_app().openapi()`` produces.

The ``aios.api.app`` import is deferred to the test body because
``aios.harness.procrastinate_app`` runs ``get_settings()`` at module-import
time and the conftest env-var fixture only fires after collection.
"""

from __future__ import annotations

import json
from pathlib import Path


def test_openapi_snapshot_matches_committed() -> None:
    from aios.api.app import create_app

    repo_root = Path(__file__).resolve().parents[2]
    committed_path = repo_root / "openapi.json"
    generated = create_app().openapi()
    committed = json.loads(committed_path.read_text())
    assert generated == committed, (
        "openapi.json is out of date — run scripts/regen-openapi.sh and commit the result"
    )
