"""Schema-drift guard for the ``search_events`` relation allowlist.

``search_events`` runs the model's SQL on the privileged application pool with
only ``SET LOCAL app.session_id`` — there is no Postgres row-level-security
backstop. The sole tenancy guard is ``_ALLOWED_RELATIONS``: every table the
parsed query reads must be one of those relations. That inverts the historical
*denylist* (which failed open every time a table was added to the schema) into
an *allowlist* (fail-closed by construction).

The fail-closed default already covers new *tables* — they are inaccessible
unless someone adds them to the allowlist. This guard covers the other
direction: if someone *does* add a name to ``_ALLOWED_RELATIONS``, it must be a
migration-defined view whose own body scopes rows by
``current_setting('app.session_id')``. Allowlisting a name that is NOT so
scoped would re-open the cross-tenant leak this fix closed.

Pure-Python: scans ``migrations/versions/*.py`` off disk; no DB, no Docker.
"""

from __future__ import annotations

import re
from pathlib import Path

from aios.tools.search_events import _ALLOWED_RELATIONS

_VERSIONS_DIR = Path(__file__).resolve().parents[2] / "migrations" / "versions"

_CREATE_VIEW_RE = re.compile(
    r"CREATE\s+(?:OR\s+REPLACE\s+)?(?:MATERIALIZED\s+)?VIEW\s+(\w+)\s+AS",
    re.IGNORECASE,
)


def _session_scoped_views() -> set[str]:
    """Names of views created in the migrations whose own body references
    ``current_setting('app.session_id')``.

    Each view body is bounded to *its own* ``CREATE ... VIEW <name> AS`` — it
    runs from the match to the next ``CREATE ... VIEW`` in the file or the
    close of the surrounding ``op.execute(...)`` triple-quoted block, whichever
    comes first — so two views defined in one block can't cross-contaminate (a
    later view's ``app.session_id`` filter must not be attributed to an earlier
    unscoped one). A view counts as session-scoped if *any* of its definitions
    (upgrade or downgrade) filters on ``app.session_id``.

    Heuristic caveat: textual presence of ``current_setting('app.session_id'``
    in the body is a proxy — it doesn't prove the setting is used in a
    *filtering* position (a WHERE/JOIN predicate) rather than, say, a projected
    column. It's a drift tripwire, not a proof of scoping; the authoritative
    scoping check is the migration's integration test.
    """
    scoped: set[str] = set()
    for path in _VERSIONS_DIR.glob("*.py"):
        src = path.read_text()
        matches = list(_CREATE_VIEW_RE.finditer(src))
        for i, m in enumerate(matches):
            start = m.end()
            bounds = [len(src)]
            if i + 1 < len(matches):
                bounds.append(matches[i + 1].start())
            close = src.find('"""', start)
            if close != -1:
                bounds.append(close)
            body = src[start : min(bounds)]
            if "current_setting('app.session_id'" in body:
                scoped.add(m.group(1))
    return scoped


# Computed once (globbing the migrations dir is not free) and shared by both
# tests below.
_SESSION_SCOPED_VIEWS = _session_scoped_views()


def test_parser_finds_the_known_session_scoped_views() -> None:
    """Sanity check the scanner against the two views we know exist, so a
    regression that makes it match *nothing* can't vacuously pass the subset
    assertion below."""
    assert "events_search" in _SESSION_SCOPED_VIEWS
    assert "memories_search" in _SESSION_SCOPED_VIEWS


def test_allowlist_is_subset_of_session_scoped_views() -> None:
    """Every allowlisted relation must be a migration-defined, session-scoped
    view. A strict subset is fine — ``memories_search`` is session-scoped but
    intentionally excluded (owned by the ``memory_search`` tool); the guard
    only checks that whatever IS allowlisted enforces ``app.session_id``."""
    unscoped = _ALLOWED_RELATIONS - _SESSION_SCOPED_VIEWS
    assert not unscoped, (
        f"search_events allowlists {sorted(unscoped)}, which are not "
        f"migration-defined views scoped by current_setting('app.session_id'). "
        f"Allowlisting a non-session-scoped relation re-opens the cross-tenant "
        f"leak this guard exists to prevent."
    )
