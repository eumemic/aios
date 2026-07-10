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

# EXPLICIT CARVE-IN (the allowlist scope contract, spelled out rather than
# smuggled past the textual scanner): ``search_views_help`` is allowlisted but
# deliberately NOT per-session scoped. It is a static catalog — a pure VALUES
# list documenting the allowlisted views' schemas, reading NO tables and
# therefore carrying zero tenant data; every session sees the same rows,
# exactly like ``information_schema``. Its
# ``current_setting('app.session_id', true) IS NOT NULL`` predicate is a
# fail-closed gate (no rows on a connection without the tool's GUC
# discipline), NOT row scoping, so this guard must not count it as such.
#
# Anything added here must satisfy the static-catalog contract, enforced by
# ``test_carve_in_views_are_pure_values_catalogs`` below (source-level: the
# view body is a VALUES list with no FROM over a real relation) and by the
# DB-backed tests in
# ``tests/integration/test_migrations_0144_search_views.py`` (depends on zero
# tables per ``information_schema.view_table_usage``; zero rows with the GUC
# unset; byte-identical rows across two different sessions).
_STATIC_CATALOG_RELATIONS = frozenset({"search_views_help"})


def test_parser_finds_the_known_session_scoped_views() -> None:
    """Sanity check the scanner against views we know exist, so a regression
    that makes it match *nothing* can't vacuously pass the subset assertion
    below. Covers the original pair plus the three 0144 data views."""
    for view in (
        "events_search",
        "memories_search",
        "tool_calls_search",
        "spans_search",
        "lifecycle_search",
    ):
        assert view in _SESSION_SCOPED_VIEWS, view


def test_allowlist_is_subset_of_session_scoped_views() -> None:
    """Every allowlisted relation must be a migration-defined, session-scoped
    view — except the explicit static-catalog carve-in documented above. A
    strict subset is fine — ``memories_search`` is session-scoped but
    intentionally excluded (owned by the ``memory_search`` tool); the guard
    only checks that whatever IS allowlisted enforces ``app.session_id``."""
    unscoped = _ALLOWED_RELATIONS - _SESSION_SCOPED_VIEWS - _STATIC_CATALOG_RELATIONS
    assert not unscoped, (
        f"search_events allowlists {sorted(unscoped)}, which are not "
        f"migration-defined views scoped by current_setting('app.session_id') "
        f"and not in the documented static-catalog carve-in. Allowlisting a "
        f"non-session-scoped relation re-opens the cross-tenant leak this "
        f"guard exists to prevent."
    )


def test_carve_in_views_are_pure_values_catalogs() -> None:
    """The static-catalog carve-in is only safe because the view carries zero
    tenant data. Enforce the source-level half of that contract: each carve-in
    view's CREATE must select from a parenthesised VALUES list — not from any
    table or view — so a future edit that joins tenant data into the catalog
    fails this test instead of silently riding the carve-in. (The DB-backed
    half — ``information_schema.view_table_usage`` empty, zero rows with the
    GUC unset, identical rows across sessions — lives in
    ``tests/integration/test_migrations_0144_search_views.py``.)"""
    sources = "\n".join(path.read_text() for path in _VERSIONS_DIR.glob("*.py"))
    for view in _STATIC_CATALOG_RELATIONS:
        pattern = re.compile(
            rf"CREATE\s+VIEW\s+{view}\s+AS\s+SELECT\s+\*\s+FROM\s+\(VALUES\s",
            re.IGNORECASE,
        )
        assert pattern.search(sources), (
            f"carve-in view {view!r} is not defined as a pure VALUES catalog "
            f"(CREATE VIEW {view} AS SELECT * FROM (VALUES ...)); the "
            f"static-catalog carve-in only holds for views that read no tables."
        )
