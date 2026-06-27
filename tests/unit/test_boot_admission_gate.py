"""Tests for the fail-closed boot-admission gate (#1575, epic #1572).

The gate (:func:`aios.retirements.boot_gate.assert_retirements_admissible`) is
the keystone provability invariant: it runs at api + worker startup, BEFORE
readiness flips green, and refuses readiness unless the live DB is proven safe
for this code. These tests pin the four acceptance branches with a fake asyncpg
pool/connection so no Postgres is needed:

* alembic_version >= contract_rev (behind-DB ⇒ refuse, the process loops),
* per-surface live residue == 0 (nonzero ⇒ refuse + algedonic alert),
* fail-closed (no DB / connection error ⇒ refuse),
* admissible (rev satisfied + zero residue ⇒ return cleanly).

The proof reads the contracted descriptors and their surface predicates from
the registry, so the assertions exercise the real seeded surfaces.
"""

from __future__ import annotations

from typing import Any

import pytest
import structlog

from aios.retirements import Retirement, Surface, boot_gate
from aios.retirements.boot_gate import (
    DatabaseBehindContract,
    DatabaseUnavailable,
    LiveResidueDetected,
    RetirementsNotAdmissible,
    assert_retirements_admissible,
)

# The highest contract_rev among the seeded descriptors; the DB must be at or
# past this to even reach the residue scan.
_AHEAD_REV = "9999"
_BEHIND_REV = "0001"


class _FakeConn:
    """A minimal asyncpg-conn stub driving the gate's two query shapes.

    * ``SELECT to_regclass('alembic_version')`` → a non-None truthy unless
      ``alembic_table_missing``.
    * ``SELECT version_num FROM alembic_version ...`` → ``version``.
    * ``SELECT count(*) FROM <table> WHERE ...`` → ``residue_counts[table]``
      (default 0), recording the bound token for assertions.

    A ``raise_on`` substring makes the matching ``fetchval`` raise the supplied
    exception, modelling a DB error mid-proof.
    """

    def __init__(
        self,
        *,
        version: str | None,
        alembic_table_missing: bool = False,
        residue_counts: dict[str, int] | None = None,
        raise_on: tuple[str, Exception] | None = None,
    ) -> None:
        self.version = version
        self.alembic_table_missing = alembic_table_missing
        self.residue_counts = residue_counts or {}
        self.raise_on = raise_on
        self.count_queries: list[tuple[str, Any]] = []

    async def fetchval(self, sql: str, *args: Any) -> Any:
        if self.raise_on is not None and self.raise_on[0] in sql:
            raise self.raise_on[1]
        if "to_regclass('alembic_version')" in sql:
            return None if self.alembic_table_missing else "alembic_version"
        if "version_num FROM alembic_version" in sql:
            return self.version
        if sql.startswith("SELECT count(*) FROM "):
            self.count_queries.append((sql, args[0] if args else None))
            # table name is the token after FROM: SELECT count(*) FROM <t> WHERE ...
            tokens = sql.split()
            table = tokens[tokens.index("FROM") + 1]
            return self.residue_counts.get(table, 0)
        raise AssertionError(f"unexpected query: {sql!r}")


class _FakePool:
    """Acquire-context wrapper around a single :class:`_FakeConn`.

    ``acquire_error`` makes ``acquire()`` itself raise, modelling a pool that
    can't hand out a connection (the fail-closed no-DB case).
    """

    def __init__(
        self, conn: _FakeConn | None = None, *, acquire_error: Exception | None = None
    ) -> None:
        self._conn = conn
        self._acquire_error = acquire_error

    def acquire(self) -> Any:
        pool = self

        class _Ctx:
            async def __aenter__(self) -> _FakeConn:
                if pool._acquire_error is not None:
                    raise pool._acquire_error
                assert pool._conn is not None
                return pool._conn

            async def __aexit__(self, *exc: Any) -> None:
                return None

        return _Ctx()


def _single_contracted() -> Retirement:
    """A one-surface, one-token contracted descriptor for focused assertions."""
    return Retirement(
        domain="tool_surface",
        action="drop",
        surfaces=(
            Surface(
                table="agents",
                jsonb_col="tools",
                predicate_sql=(
                    "EXISTS(SELECT 1 FROM jsonb_array_elements(tools) e WHERE e->>'type' = :token)"
                ),
            ),
        ),
        introduced_rev="0500",
        contract_rev="0500",
        token="complete_goal",
    )


@pytest.fixture
def patch_registry(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Return a helper that swaps :data:`boot_gate.REGISTRY` for a test set."""

    def _set(registry: tuple[Retirement, ...]) -> None:
        monkeypatch.setattr(boot_gate, "REGISTRY", registry)

    return _set


async def test_behind_db_refuses_readiness(patch_registry: Any) -> None:
    """alembic_version < contract_rev ⇒ DatabaseBehindContract (the loop case)."""
    patch_registry((_single_contracted(),))
    pool = _FakePool(_FakeConn(version=_BEHIND_REV))
    with pytest.raises(DatabaseBehindContract) as exc:
        await assert_retirements_admissible(pool)
    assert isinstance(exc.value, RetirementsNotAdmissible)
    # The residue scan must NOT have run while the DB is behind.
    conn = pool._conn
    assert conn is not None
    assert conn.count_queries == []


async def test_unmigrated_db_is_behind(patch_registry: Any) -> None:
    """No alembic_version table at all ⇒ behind (refuse), never a crash."""
    patch_registry((_single_contracted(),))
    pool = _FakePool(_FakeConn(version=None, alembic_table_missing=True))
    with pytest.raises(DatabaseBehindContract):
        await assert_retirements_admissible(pool)


async def test_nonzero_residue_refuses_and_alerts(patch_registry: Any) -> None:
    """Residue on a surface ⇒ LiveResidueDetected + an algedonic alert."""
    patch_registry((_single_contracted(),))
    conn = _FakeConn(version=_AHEAD_REV, residue_counts={"agents": 3})
    with structlog.testing.capture_logs() as logs, pytest.raises(LiveResidueDetected):
        await assert_retirements_admissible(_FakePool(conn))
    # The token was bound into the predicate, not interpolated.
    assert any(token == "complete_goal" for _sql, token in conn.count_queries)
    # An algedonic alert was emitted.
    alerts = [
        r for r in logs if r.get("event") == "boot_gate.live_residue" and r.get("algedonic") is True
    ]
    assert alerts, logs


async def test_no_db_connection_fails_closed(patch_registry: Any) -> None:
    """A pool that can't acquire ⇒ DatabaseUnavailable (fail-closed)."""
    patch_registry((_single_contracted(),))
    pool = _FakePool(acquire_error=ConnectionError("no route to db"))
    with pytest.raises(DatabaseUnavailable) as exc:
        await assert_retirements_admissible(pool)
    assert isinstance(exc.value, RetirementsNotAdmissible)


async def test_query_error_mid_proof_fails_closed(patch_registry: Any) -> None:
    """A query raising mid-proof ⇒ DatabaseUnavailable, not an escaping error."""
    patch_registry((_single_contracted(),))
    conn = _FakeConn(
        version=_AHEAD_REV,
        raise_on=("count(*)", ConnectionResetError("dropped mid-scan")),
    )
    with pytest.raises(DatabaseUnavailable):
        await assert_retirements_admissible(_FakePool(conn))


async def test_admissible_when_ahead_and_clean(patch_registry: Any) -> None:
    """Rev satisfied + zero residue everywhere ⇒ returns cleanly."""
    patch_registry((_single_contracted(),))
    conn = _FakeConn(version=_AHEAD_REV)
    await assert_retirements_admissible(_FakePool(conn))
    # The residue scan ran (and found nothing).
    assert conn.count_queries


async def test_uncontracted_descriptor_is_skipped(patch_registry: Any) -> None:
    """A descriptor with no contract_rev is not gated — nothing to prove yet."""
    uncontracted = Retirement(
        domain="tool_surface",
        action="drop",
        surfaces=(
            Surface(
                table="agents",
                jsonb_col="tools",
                predicate_sql="EXISTS(SELECT 1 FROM jsonb_array_elements(tools) e "
                "WHERE e->>'type' = :token)",
            ),
        ),
        introduced_rev="0500",
        contract_rev=None,
        token="not_yet",
    )
    patch_registry((uncontracted,))
    # Even a behind/empty DB is admissible because nothing is contracted; the
    # gate must not even query.
    conn = _FakeConn(version=_BEHIND_REV)
    await assert_retirements_admissible(_FakePool(conn))
    assert conn.count_queries == []


async def test_nullable_surface_guards_is_not_null(
    patch_registry: Any,
) -> None:
    """A nullable surface's residue query guards the column with IS NOT NULL."""
    nullable_desc = Retirement(
        domain="tool_surface",
        action="drop",
        surfaces=(
            Surface(
                table="sessions",
                jsonb_col="tools",
                predicate_sql="EXISTS(SELECT 1 FROM jsonb_array_elements(tools) e "
                "WHERE e->>'type' = :token)",
                nullable=True,
            ),
        ),
        introduced_rev="0500",
        contract_rev="0500",
        token="complete_goal",
    )
    patch_registry((nullable_desc,))
    conn = _FakeConn(version=_AHEAD_REV)
    await assert_retirements_admissible(_FakePool(conn))
    assert any("IS NOT NULL" in sql for sql, _token in conn.count_queries)
