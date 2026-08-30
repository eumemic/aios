"""Pure-Python controls for candidate migration admission metadata."""

from __future__ import annotations

from alembic.script import ScriptDirectory

from aios.db import migrations


def test_known_revisions_matches_real_alembic_graph() -> None:
    """Admission recognizes every declaration form accepted by Alembic."""
    scripts = ScriptDirectory.from_config(migrations.alembic_config())
    expected = {revision.revision for revision in scripts.walk_revisions()}

    assert migrations._known_revisions() == expected
    assert {"0174", "0175", "0176", "0177"} <= expected
