"""Single-source guard for persisted-enum value sets — the live-``CHECK`` half
(#1081).

Companion to ``tests/unit/test_persisted_enum_writer_drift.py`` (the
writer-signature half). Together they make the Python ``Literal`` the *single*
editable source for every persisted enum's value set, so a forgotten paired
migration can no longer ship a prod-only runtime ``CHECK`` violation.

This module introspects the *live* migrated schema via
``pg_catalog.pg_get_constraintdef`` — the project's sanctioned drift idiom,
modelled on ``tests/unit/test_agent_tooltype_registry_drift.py``'s
``get_args(...)`` set-equality and on the ``pg_get_constraintdef`` reads in
``tests/integration/test_migrations_0110_bindings_cascade.py``. It asserts:

  1. Every value-enum ``CHECK`` that is *kept* (the (C) rows whose writer is a
     hardcoded SQL literal, or a nullable value-enum) still equals
     ``set(get_args(<its Literal>))`` — pinning the surviving CHECK to the
     Literal so the two cannot drift apart and pass CI.

  2. Every value-enum ``CHECK`` that migration 0111 *drops* (the (A)/(B) rows,
     now single-sourced from the typed writer the unit test pins) is actually
     absent — so a re-introduced redundant CHECK is caught here.

The value set is parsed straight out of ``pg_get_constraintdef`` text
(``... IN ('a', 'b', ...)`` / ``ANY (ARRAY['a'::text, ...])``), so no test data
or write round-trip is needed — pure ``get_args`` set algebra against a
``pg_catalog`` read.
"""

from __future__ import annotations

import re
from typing import Any, get_args

import asyncpg
import pytest

from aios.models.memory_stores import ActorType, MemoryOperation
from tests.conftest import needs_docker

# Constraints migration 0111 drops (the (A)/(B) rows). They must be ABSENT.
_DROPPED_CHECKS: tuple[str, ...] = (
    "vault_credentials_auth_type_check",
    "triggers_last_fire_status_check",
    "wf_run_events_type_check",
    "wf_run_signals_kind_check",
    "wf_runs_status_check",
    "session_memory_stores_access_check",
    "memory_versions_created_by_type_check",
)

_QUOTED = re.compile(r"'([^']*)'")


def _check_value_set(constraintdef: str) -> set[str]:
    """Parse the single-quoted literals out of a ``pg_get_constraintdef`` body
    (handles both ``col IN ('a','b')`` and ``col = ANY (ARRAY['a'::text,...])``
    renderings — both are just single-quoted tokens)."""
    return set(_QUOTED.findall(constraintdef))


async def _constraintdef(conn: asyncpg.Connection[Any], name: str) -> str | None:
    result: str | None = await conn.fetchval(
        """
        SELECT pg_get_constraintdef(c.oid)
          FROM pg_constraint c
         WHERE c.contype = 'c' AND c.conname = $1
        """,
        name,
    )
    return result


@needs_docker
@pytest.mark.integration
async def test_kept_value_enum_checks_match_their_literal(migrated_db_url: str) -> None:
    """Each surviving value-enum CHECK's value set == ``get_args(<Literal>)``."""
    kept: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("memory_versions_operation_check", get_args(MemoryOperation)),
        ("memory_versions_redacted_by_type_check", get_args(ActorType)),
    )
    conn = await asyncpg.connect(migrated_db_url)
    try:
        for name, literal_args in kept:
            cdef = await _constraintdef(conn, name)
            assert cdef is not None, f"expected CHECK {name} to still exist"
            assert _check_value_set(cdef) == set(literal_args), (
                f"CHECK {name} value set {_check_value_set(cdef)} drifted from "
                f"its Literal {set(literal_args)} — single-source violated"
            )
    finally:
        await conn.close()


@needs_docker
@pytest.mark.integration
async def test_redundant_value_enum_checks_are_dropped(migrated_db_url: str) -> None:
    """The (A)/(B) CHECKs are single-sourced from the typed writer (pinned by
    the unit drift-test); re-introducing one here is a regression."""
    conn = await asyncpg.connect(migrated_db_url)
    try:
        for name in _DROPPED_CHECKS:
            cdef = await _constraintdef(conn, name)
            assert cdef is None, (
                f"CHECK {name} should have been dropped in migration 0111 "
                f"(its column is single-sourced from a Literal-typed writer); "
                f"found: {cdef}"
            )
    finally:
        await conn.close()
