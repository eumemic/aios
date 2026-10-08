"""Unit regression tests for migration 0194 (#2513, fix round).

Two defects were parked against this migration and are pinned here without a
live Postgres, by capturing the ``op.execute`` statements the migration emits:

* The VALIDATE of the ``NOT VALID`` CHECK runs inside ``autocommit_block()``
  (so the table scan does not block run inserts), but the ``SET LOCAL
  lock_timeout`` set up front is transaction-scoped and is committed away on
  entering the block. VALIDATE would then run with PostgreSQL's default
  (unbounded) lock wait and a conflicting vacuum/DDL could hang the deploy.
  The fix re-arms a session-scoped ``SET lock_timeout`` inside the block,
  before VALIDATE, and resets it afterward.

The over-correction guard: the degenerate way to "bound the VALIDATE lock"
is to pull VALIDATE out of the autocommit block into the opening transaction,
where the ``SET LOCAL`` is still live -- but that re-introduces the blocking
full-table scan the autocommit block exists to avoid. These tests assert the
VALIDATE is STILL inside the autocommit block AND is now lock-bounded.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

_MIGRATION = (
    Path(__file__).parents[2] / "migrations" / "versions" / "0194_wf_runs_visibility_check.py"
)

_ENTER = "<<AUTOCOMMIT ENTER>>"
_EXIT = "<<AUTOCOMMIT EXIT>>"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_migration_0194", _MIGRATION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _capture(operation: str) -> tuple[list[str], Mock]:
    """Return the ordered SQL statements, with markers around the autocommit block."""
    migration = _load()
    statements: list[str] = []

    class _Block:
        def __enter__(self) -> _Block:
            statements.append(_ENTER)
            return self

        def __exit__(self, *exc: object) -> None:
            statements.append(_EXIT)

    context = Mock()
    context.autocommit_block.return_value = _Block()
    migration.op.get_context = Mock(return_value=context)
    migration.op.execute = statements.append

    getattr(migration, operation)()
    return statements, context


def test_upgrade_adds_not_valid_then_validates_in_autocommit_block() -> None:
    statements, context = _capture("upgrade")

    assert statements == [
        "SET LOCAL lock_timeout = '5s'",
        "ALTER TABLE wf_runs DROP CONSTRAINT IF EXISTS wf_runs_visibility_known",
        "ALTER TABLE wf_runs ADD CONSTRAINT wf_runs_visibility_known "
        "CHECK (visibility IN ('account', 'session')) NOT VALID",
        _ENTER,
        "SET lock_timeout = '5s'",
        "ALTER TABLE wf_runs VALIDATE CONSTRAINT wf_runs_visibility_known",
        "SET lock_timeout = DEFAULT",
        _EXIT,
    ]
    context.autocommit_block.assert_called_once_with()


def test_validate_runs_under_a_bounded_lock_wait_inside_the_block() -> None:
    """Defect #2: a lock bound must be live when VALIDATE runs.

    The opening ``SET LOCAL`` is transaction-scoped and gone after the block
    commits, so a *session*-scoped ``SET lock_timeout`` (not LOCAL) must be
    armed inside the block before VALIDATE. Asserting against the pre-fix
    migration (no SET inside the block) fails here.
    """
    statements, _ = _capture("upgrade")

    enter = statements.index(_ENTER)
    exit_ = statements.index(_EXIT)
    validate = statements.index("ALTER TABLE wf_runs VALIDATE CONSTRAINT wf_runs_visibility_known")

    # Over-correction guard: VALIDATE is still inside the autocommit block, so the
    # scan does not block run inserts -- the fix did not "bound the lock" by
    # dragging VALIDATE back into the opening transaction.
    assert enter < validate < exit_

    in_block = statements[enter + 1 : exit_]
    # A session-scoped bound (not transaction-local, which the block discards)
    # is set before VALIDATE.
    assert "SET lock_timeout = '5s'" in in_block
    assert in_block.index("SET lock_timeout = '5s'") < in_block.index(
        "ALTER TABLE wf_runs VALIDATE CONSTRAINT wf_runs_visibility_known"
    )
    # It must be a plain (session) SET, not SET LOCAL, which would be discarded
    # before the next statement in the autocommit block.
    assert "SET LOCAL lock_timeout = '5s'" not in in_block
    # The bound is released after validation so it does not leak to later work
    # on the connection.
    assert in_block[-1] == "SET lock_timeout = DEFAULT"


def test_downgrade_drops_the_constraint_under_a_bounded_lock_wait() -> None:
    statements, _ = _capture("downgrade")

    assert statements == [
        "SET LOCAL lock_timeout = '5s'",
        "ALTER TABLE wf_runs DROP CONSTRAINT IF EXISTS wf_runs_visibility_known",
    ]


def test_migration_is_parented_on_masters_head() -> None:
    """0194 sits on top of master's 0193 (``0192_wf_runs_budget_run`` ->
    ``0193_wf_runs_agent_principal``), so the alembic history keeps a single head.
    Parenting it on 0191 (where this branch was cut) would fork it into two heads,
    0193 and 0194, at merge time (#2513)."""
    migration = _load()
    assert migration.revision == "0194"
    assert migration.down_revision == "0193"
