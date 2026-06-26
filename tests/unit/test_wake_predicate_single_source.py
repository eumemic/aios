"""Structural guard: the sweep wake-decision predicates are *generated from* a
single shared source, so a detector and its action-path counterpart cannot
drift (issue #1065).

Each wake decision runs a *detector* predicate (in ``harness/sweep.py``) that
MUST agree with the corresponding *read/dispatch-path* predicate, or the worker
wakes a session with no work to do (the #155 busy-loop) or skips one that needs
inference. Historically that agreement was kept by developer vigilance — four
hand-maintained copies of two booleans under "MUST stay byte-for-byte in sync"
comments. #1065 replaced the vigilance with structure: one alias-parameterized
SQL-fragment generator per predicate, composed into both sides.

These tests are the structural replacement for the deleted sync-comments: they
fail the instant either side stops composing from the shared generator. Pure
import/introspection — nothing here touches Postgres.
"""

from __future__ import annotations

import inspect
import re

from aios.db import queries
from aios.db.queries import events as events_q
from aios.db.queries import sessions as sessions_q
from aios.harness import sweep


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


# ─── the three generators are the single source, re-exported by identity ──────


def test_generators_reexported_by_identity() -> None:
    """The package root re-exports the SAME generator objects the submodules
    define — so ``sweep`` importing from ``aios.db.queries`` reaches the real
    source, and a patch on either reference is observed by both layers."""
    assert queries.session_active_predicate is sessions_q.session_active_predicate
    assert queries.session_errored_predicate is sessions_q.session_errored_predicate
    assert queries.confirmed_unresolved_predicate is events_q.confirmed_unresolved_predicate


def test_sweep_imports_the_generators_not_inline_copies() -> None:
    """The sweep consumes the shared generators (the correct dependency
    direction: sweep already imports heavily from ``db/queries``)."""
    assert sweep.session_active_predicate is queries.session_active_predicate
    assert sweep.session_errored_predicate is queries.session_errored_predicate
    assert sweep.confirmed_unresolved_predicate is queries.confirmed_unresolved_predicate


# ─── instance 1: active/errored status boolean — one source, both sides ───────


def test_candidate_rows_sql_generated_from_session_active_predicate() -> None:
    """``CANDIDATE_ROWS_SQL`` (the sweep wake detector) is composed FROM
    ``session_active_predicate`` at its ``s`` alias — not a hand-kept re-encode.
    Its read-path twin ``_SESSION_ACTIVE_EXPR`` is the SAME generator at the
    ``sessions`` alias, so the two cannot diverge."""
    candidate = _norm(sweep.CANDIDATE_ROWS_SQL)
    assert _norm(queries.session_active_predicate("s")) in candidate
    # read-path twin is the identical generator, just a different alias
    assert queries.session_active_predicate("sessions") == sessions_q._SESSION_ACTIVE_EXPR


def test_errored_sessions_sql_generated_from_session_errored_predicate() -> None:
    """``ERRORED_SESSIONS_SQL`` is composed FROM ``session_errored_predicate``;
    its read-path twin ``_SESSION_ERRORED_EXPR`` is the same generator."""
    errored = _norm(sweep.ERRORED_SESSIONS_SQL)
    assert _norm(queries.session_errored_predicate("s")) in errored
    assert queries.session_errored_predicate("sessions") == sessions_q._SESSION_ERRORED_EXPR


def test_ghost_asst_sql_generated_from_session_errored_predicate() -> None:
    """``GHOST_ASST_SQL`` pushes the errored bound to skip parked sessions; it
    composes the SAME ``session_errored_predicate`` source as
    ``ERRORED_SESSIONS_SQL`` and the read path — this single-source composition
    is what makes the SQL pre-filter the sole load-bearing errored-skip for
    ghost repair."""
    ghost = _norm(sweep.GHOST_ASST_SQL.format(scope_clause=""))
    assert _norm(queries.session_errored_predicate("s")) in ghost


def test_session_active_predicate_is_alias_parameterized() -> None:
    """The generator emits the SAME boolean modulo table alias — the property
    that lets one source serve both the ``s``-aliased sweep and the
    ``sessions``-aliased read path."""
    s_form = queries.session_active_predicate("s")
    sessions_form = queries.session_active_predicate("sessions")
    assert s_form == sessions_form.replace("sessions.", "s.")
    assert s_form != sessions_form  # the alias actually varied


# ─── instance 2: confirmed-dispatch boolean — one source, both sides ──────────


def test_confirmed_rows_sql_generated_from_confirmed_unresolved_predicate() -> None:
    """``CONFIRMED_ROWS_SQL`` (the cross-session wake detector) is composed FROM
    ``confirmed_unresolved_predicate`` — the same generator the per-session
    dispatch resolver ``list_confirmed_unresolved_tool_calls`` composes."""
    # The detector renders the age placeholder as a ``str.format`` field; the
    # generator's output for that field-name survives into the production text.
    detector = _norm(sweep.CONFIRMED_ROWS_SQL.format(scope_clause="", age_param="$1"))
    assert _norm(queries.confirmed_unresolved_predicate("lc", "$1")) in detector


def test_dispatch_resolver_generated_from_confirmed_unresolved_predicate() -> None:
    """The dispatch-side action path ``list_confirmed_unresolved_tool_calls``
    builds its ``lc`` WHERE sub-predicate by composing the SAME generator (at
    its ``$3`` positional age param) — so detection and dispatch resolve the
    identical condition and cannot drift."""
    src = inspect.getsource(events_q.list_confirmed_unresolved_tool_calls)
    assert 'confirmed_unresolved_predicate("lc", "$3")' in src


def test_confirmed_predicate_is_tenant_scoped() -> None:
    """The unified ``NOT EXISTS`` carries the resolver's correct, tenant-scoped
    form (``tr.account_id = lc.account_id``) — #1065 reconciled the sweep copy's
    pre-existing drift (it omitted this) toward the resolver's form."""
    pred = _norm(queries.confirmed_unresolved_predicate("lc", "$1"))
    assert "tr.account_id = lc.account_id" in pred
    # ...and so does the rendered sweep detector, by composition
    detector = _norm(sweep.CONFIRMED_ROWS_SQL.format(scope_clause="", age_param="$1"))
    assert "tr.account_id = lc.account_id" in detector


def test_confirmed_predicate_age_param_is_caller_supplied() -> None:
    """The age placeholder is the caller's — ``$N`` positional for the resolver,
    a ``str.format`` field for the sweep — so each side binds it correctly while
    sharing one boolean body."""
    assert "$3" in queries.confirmed_unresolved_predicate("lc", "$3")
    assert "{age_param}" in queries.confirmed_unresolved_predicate("lc", "{age_param}")
