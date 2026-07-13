"""Unit tests for the retirement migration generator (#1578, epic #1572).

``bin/aios-retire <descriptor>`` emits the three lifecycle migrations
(expand/backfill/contract) from a registry descriptor. These tests assert the
acceptance criteria as pure-Python checks over the generated source — no DB, no
Docker:

* all three migrations are generated in chain order on top of a given head;
* the backfill is EXISTS-pre-checked, batched on high-cardinality surfaces,
  re-runnable, and stamps the epoch;
* the contract carries the in-transaction abort-on-nonzero
  guard;
* every emitted file is valid Python that declares the linear chain pointers and
  matches the 0116/0120 shape;
* the whole on-disk ladder (with a generated chain written in) keeps a single
  linear head — the migration-chain invariant.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory

from aios.retirements import Retirement
from aios.retirements.migration_gen import (
    EPOCH_COLUMN,
    HIGH_CARDINALITY_TABLES,
    GeneratedChain,
    current_head,
    generate,
    next_revisions,
    resolve_descriptor,
)
from aios.retirements.registry import (
    LEGACY_BUILTIN_RENAMES,
    REGISTRY,
    RETIRED_GOAL_OUTCOME_BUILTINS,
)

_MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


def _chain(retirement: Retirement, head: str = "0123") -> GeneratedChain:
    return generate(retirement, head, descriptor_name="UNDER_TEST")


def _module(source: str) -> ast.Module:
    """Parse generated source, failing the test with a clear message on a syntax error."""

    try:
        return ast.parse(source)
    except SyntaxError as exc:  # pragma: no cover - failure path
        pytest.fail(f"generated migration is not valid Python: {exc}\n\n{source}")


# ── Chain shape & revision wiring ─────────────────────────────────────────────


def test_next_revisions_extends_numeric_ladder() -> None:
    assert next_revisions("0123", 3) == ["0124", "0125", "0126"]
    assert next_revisions("0009", 3) == ["0010", "0011", "0012"]


def test_next_revisions_rejects_non_numeric_head() -> None:
    with pytest.raises(ValueError):
        next_revisions("0046_reencrypt", 3)


def test_generates_three_migrations_in_chain_order() -> None:
    chain = _chain(LEGACY_BUILTIN_RENAMES, head="0123")
    assert [m.revision for m in chain] == ["0124", "0125", "0126"]
    # Chain pointers: expand→head, backfill→expand, contract→backfill.
    assert 'down_revision: str | None = "0123"' in chain.expand.source
    assert 'revision: str = "0124"' in chain.expand.source
    assert 'down_revision: str | None = "0124"' in chain.backfill.source
    assert 'revision: str = "0125"' in chain.backfill.source
    assert 'down_revision: str | None = "0125"' in chain.contract.source
    assert 'revision: str = "0126"' in chain.contract.source


def test_filenames_carry_phase_and_revision() -> None:
    chain = _chain(LEGACY_BUILTIN_RENAMES)
    assert chain.expand.filename.endswith("_expand.py")
    assert chain.backfill.filename.endswith("_backfill.py")
    assert chain.contract.filename.endswith("_contract.py")
    for m in chain:
        assert m.filename.startswith(m.revision + "_")


@pytest.mark.parametrize("retirement", REGISTRY)
def test_all_generated_sources_are_valid_python(retirement: Retirement) -> None:
    for migration in _chain(retirement):
        mod = _module(migration.source)
        funcs = {n.name for n in mod.body if isinstance(n, ast.FunctionDef)}
        assert {"upgrade", "downgrade"} <= funcs


@pytest.mark.parametrize("retirement", REGISTRY)
def test_downgrade_is_a_noop_forward_only(retirement: Retirement) -> None:
    # Every emitted migration is forward-only: downgrade() is a bare pass.
    for migration in _chain(retirement):
        mod = _module(migration.source)
        (downgrade,) = [
            n for n in mod.body if isinstance(n, ast.FunctionDef) and n.name == "downgrade"
        ]
        assert all(isinstance(stmt, (ast.Pass, ast.Expr)) for stmt in downgrade.body)
        assert not any(isinstance(stmt, ast.Call) for stmt in downgrade.body)


# ── Expand ────────────────────────────────────────────────────────────────────


def test_expand_is_a_genuine_noop_without_ledger_dml() -> None:
    src = _chain(LEGACY_BUILTIN_RENAMES).expand.source
    assert "retirement_ledger" not in src
    assert "op.execute" not in src
    assert "from alembic import op" not in src
    assert "jsonb_set" not in src


# ── Backfill ──────────────────────────────────────────────────────────────────


def test_backfill_covers_every_surface() -> None:
    src = _chain(LEGACY_BUILTIN_RENAMES).backfill.source
    for surface in LEGACY_BUILTIN_RENAMES.surfaces:
        # Each surface's table+column appears in an UPDATE.
        assert re.search(rf"UPDATE {surface.table} SET {surface.jsonb_col} =", src), (
            f"missing UPDATE for surface {surface.table}.{surface.jsonb_col}"
        )


def test_backfill_is_exists_pre_checked() -> None:
    src = _chain(LEGACY_BUILTIN_RENAMES).backfill.source
    # Every surface UPDATE is guarded by an EXISTS over its jsonb column.
    for surface in LEGACY_BUILTIN_RENAMES.surfaces:
        assert f"jsonb_array_elements({surface.jsonb_col})" in src
    assert "EXISTS(" in src


def test_backfill_batches_high_cardinality_surfaces_only() -> None:
    src = _chain(LEGACY_BUILTIN_RENAMES).backfill.source
    # High-cardinality surfaces use the bounded ctid LIMIT loop.
    for surface in LEGACY_BUILTIN_RENAMES.surfaces:
        block = _surface_block(src, surface.table)
        if surface.table in HIGH_CARDINALITY_TABLES:
            assert "LOOP" in block and "LIMIT" in block and "ctid" in block, (
                f"high-cardinality surface {surface.table} must be batched"
            )
        else:
            assert "LOOP" not in block, (
                f"low-cardinality surface {surface.table} should be a single UPDATE"
            )


def test_backfill_stamps_the_epoch() -> None:
    chain = _chain(LEGACY_BUILTIN_RENAMES, head="0123")
    src = chain.backfill.source
    # Epoch stamped to the backfill's own revision number on every surface: one
    # SET-clause occurrence per surface (the docstring mention is not a SET).
    set_clause = f", {EPOCH_COLUMN} = {chain.backfill.revision}"
    assert src.count(set_clause) == len(LEGACY_BUILTIN_RENAMES.surfaces)


def test_backfill_rename_maps_tokens_to_successors() -> None:
    src = _chain(LEGACY_BUILTIN_RENAMES).backfill.source
    for token, successor in LEGACY_BUILTIN_RENAMES.token_map().items():
        assert successor is not None
        assert f"WHEN '{token}' THEN '{successor}'" in src
    # Order-preserving dedupe, 0116 shape.
    assert "jsonb_agg(elem ORDER BY ord)" in src
    assert "DISTINCT ON (dedup_key)" in src


def test_backfill_drop_filters_tokens() -> None:
    src = _chain(RETIRED_GOAL_OUTCOME_BUILTINS).backfill.source
    # 0120/0122 drop shape: filter out retired elements, no jsonb_set/CASE remap.
    assert "WHERE elem->>'type' NOT IN (" in src
    assert "jsonb_set" not in src
    for token in RETIRED_GOAL_OUTCOME_BUILTINS.tokens:
        assert f"'{token}'" in src


def test_backfill_guards_nullable_surfaces() -> None:
    src = _chain(LEGACY_BUILTIN_RENAMES).backfill.source
    nullable = [s for s in LEGACY_BUILTIN_RENAMES.surfaces if s.nullable]
    assert nullable, "fixture should have a nullable surface to exercise the guard"
    for surface in nullable:
        block = _surface_block(src, surface.table)
        assert f"{surface.jsonb_col} IS NOT NULL" in block


def test_backfill_creates_and_drops_its_transform_function() -> None:
    src = _chain(LEGACY_BUILTIN_RENAMES).backfill.source
    assert "CREATE FUNCTION" in src
    assert "DROP FUNCTION" in src


# ── Contract ──────────────────────────────────────────────────────────────────


def test_contract_has_no_ledger_dml() -> None:
    src = _chain(LEGACY_BUILTIN_RENAMES).contract.source
    assert "retirement_ledger" not in src
    assert "contract_rev =" not in src
    assert "phase = 'contract'" not in src


def test_contract_has_in_transaction_abort_on_nonzero_guard() -> None:
    chain = _chain(LEGACY_BUILTIN_RENAMES)
    src = chain.contract.source
    # Residue scan over every surface.
    for surface in LEGACY_BUILTIN_RENAMES.surfaces:
        assert f"SELECT count(*) FROM {surface.table}" in src
    # Abort-on-nonzero inside the migration transaction (RAISE EXCEPTION in a DO
    # block runs in the migration txn, so alembic_version never reaches the
    # contract rev on abort).
    assert "RAISE EXCEPTION" in src
    assert "residue > 0" in src
    # The guard remains the only and therefore final operation in upgrade().
    module = _module(src)
    (upgrade,) = [
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "upgrade"
    ]
    assert len(upgrade.body) == 1
    assert isinstance(upgrade.body[-1], ast.Expr)
    assert isinstance(upgrade.body[-1].value, ast.Call)
    assert ast.unparse(upgrade.body[-1].value.func) == "op.execute"
    assert "RAISE EXCEPTION" in ast.literal_eval(upgrade.body[-1].value.args[0])


def test_contract_guard_references_the_backfill_revision() -> None:
    chain = _chain(LEGACY_BUILTIN_RENAMES)
    # The abort message points operators back at the backfill rev to re-run.
    assert chain.backfill.revision in chain.contract.source


# ── Descriptor resolution ─────────────────────────────────────────────────────


def test_resolve_descriptor_by_constant_name() -> None:
    retirement, name = resolve_descriptor("LEGACY_BUILTIN_RENAMES")
    assert retirement is LEGACY_BUILTIN_RENAMES
    assert name == "LEGACY_BUILTIN_RENAMES"


def test_resolve_descriptor_unknown_lists_available() -> None:
    with pytest.raises(KeyError) as exc:
        resolve_descriptor("NOPE")
    assert "LEGACY_BUILTIN_RENAMES" in str(exc.value)


# ── Ladder integration: a generated chain keeps the ladder single-headed ──────


def test_current_head_matches_alembic() -> None:
    cfg = Config()
    cfg.set_main_option("script_location", str(_MIGRATIONS_DIR))
    script = ScriptDirectory.from_config(cfg)
    (expected,) = script.get_heads()
    assert current_head(str(_MIGRATIONS_DIR)) == expected


def test_generated_chain_stays_single_headed_and_linear(tmp_path: Path) -> None:
    """Writing a generated chain onto a copy of the ladder keeps one linear head."""

    # Copy the real versions dir into a temp ladder.
    src_versions = _MIGRATIONS_DIR / "versions"
    dst = tmp_path / "migrations"
    dst.mkdir()
    (dst / "versions").mkdir()
    # Minimal env so ScriptDirectory can load; copy script.py.mako + env.py.
    for fname in ("env.py", "script.py.mako"):
        (dst / fname).write_text((_MIGRATIONS_DIR / fname).read_text())
    for f in src_versions.glob("*.py"):
        (dst / "versions" / f.name).write_text(f.read_text())

    head = current_head(str(dst))
    chain = generate(LEGACY_BUILTIN_RENAMES, head, descriptor_name="LEGACY_BUILTIN_RENAMES")
    for migration in chain:
        (dst / "versions" / migration.filename).write_text(migration.source)

    cfg = Config()
    cfg.set_main_option("script_location", str(dst))
    script = ScriptDirectory.from_config(cfg)
    heads = list(script.get_heads())
    assert heads == [chain.contract.revision], heads

    # Whole chain linear: every revision has a single (str|None) parent.
    for rev in script.walk_revisions():
        assert rev.down_revision is None or isinstance(rev.down_revision, str)
    bases = [r.revision for r in script.walk_revisions() if r.down_revision is None]
    assert len(bases) == 1


# ── Helpers ───────────────────────────────────────────────────────────────────


def _surface_block(src: str, table: str) -> str:
    """The slice of backfill source belonging to one surface's UPDATE block.

    Each surface is rendered as one ``op.execute(...)`` call; we return the
    op.execute call whose body mentions ``UPDATE <table> ``. Splitting on the
    call boundary (rather than on the comment) is robust to multi-line comments.
    """

    needle = f"UPDATE {table} SET"
    pos = src.index(needle)
    # Walk back to the start of the enclosing op.execute( call ...
    start = src.rindex("op.execute(", 0, pos)
    # ... and forward to the next op.execute( or the DROP FUNCTION line.
    after = src.find("op.execute(", pos)
    end = after if after != -1 else len(src)
    return src[start:end]
