"""Integration test: the clone policy covers EVERY column of each copied table.

This is the #1676 completeness gate.  ``clone_session`` generates its four
``INSERT … SELECT`` projections from the per-column policy tables in
``clone_policy.py``.  Historically the column lists were hand-enumerated and
silently drifted from the schema — migration 0127 added five ``cumulative_*``
columns to ``events`` and the clone never learned them, copying NULL class-mass
counters on every clone (the LIVE drift the issue was filed on).

This test pins each policy's ``keys()`` to the live ``information_schema``
columns of its relation on the migrated testcontainer schema.  A future
migration that adds a column without classifying it fails here, deterministically,
before merge — the defect class becomes unauthorable rather than merely guarded.

Named non-goal (recorded, not gated here): this closes COLUMN drift, not TABLE
drift.  A future fifth session-attachment table is still an un-gated edit site
(the #580 class) — it would need its own policy + entry in ``CLONE_POLICIES``.
"""

from __future__ import annotations

import asyncpg
import pytest

from aios.db.queries.clone_policy import CLONE_POLICIES

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("table_name", sorted(CLONE_POLICIES))
async def test_clone_policy_matches_schema(
    migrated_db_url: str, _reset_db_state: None, table_name: str
) -> None:
    """Every column of ``table_name`` has exactly one policy arm — no more, no less.

    ``policy.keys() == information_schema.columns`` for the relation.  A
    superset (policy names a column the table lacks) means a stale/renamed
    arm; a subset (table has a column the policy omits) is the silent-drift
    defect the whole refactor exists to foreclose.
    """
    policy = CLONE_POLICIES[table_name]
    conn = await asyncpg.connect(migrated_db_url)
    try:
        rows = await conn.fetch(
            """
            SELECT column_name
              FROM information_schema.columns
             WHERE table_schema = 'public' AND table_name = $1
            """,
            table_name,
        )
    finally:
        await conn.close()

    schema_columns = {r["column_name"] for r in rows}
    assert schema_columns, f"no columns found for table {table_name!r} — migration gap?"

    policy_columns = set(policy)

    missing_from_policy = schema_columns - policy_columns
    extra_in_policy = policy_columns - schema_columns
    assert not missing_from_policy, (
        f"{table_name}: columns present in the schema but MISSING a clone-policy "
        f"arm (silent-drift defect — classify them in clone_policy.py): "
        f"{sorted(missing_from_policy)}"
    )
    assert not extra_in_policy, (
        f"{table_name}: clone-policy names columns the schema does not have "
        f"(stale/renamed arm — remove or rename in clone_policy.py): "
        f"{sorted(extra_in_policy)}"
    )
