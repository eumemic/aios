"""Migration generator — emit expand/backfill/contract migrations from a descriptor.

This is the *data-migration generator* arm of the pattern-retirement lifecycle
(#1578, epic #1572). Given a single :class:`~aios.retirements.Retirement`
descriptor (declared as data in :mod:`aios.retirements.registry`, #1573), it
emits the **three lifecycle migrations** so each new retirement is a
fill-in-the-descriptor exercise, not three hand-written migrations:

1. **Expand** (rev ``N``) — no-op on data; inserts the descriptor's ledger row
   (``phase = 'expand'``, ``contract_rev = NULL``, ``sla_days``). This is the
   span during which the registry-driven read-tolerance shim
   (:func:`aios.retirements.registry.tolerated_rename_map`, gated on
   ``contract_rev IS NULL``) keeps remapping the retired tokens on read.

2. **Backfill** (rev ``N+1``) — the real data rewrite. ``EXISTS``-pre-checked,
   **batched** set-based ``UPDATE`` / jsonb edit across *all* the descriptor's
   surfaces. High-cardinality surfaces (``wf_runs``, ``agent_versions``, …) are
   rewritten in bounded ``ctid``-keyed batches to cap how long any single
   statement holds its row locks — the 0066 lesson (a single unbounded
   ``UPDATE`` over a high-cardinality table holds locks for the whole table and
   can wedge concurrent writers). Re-runnable: every statement is
   ``EXISTS``-guarded so it no-ops on already-fixed rows, and a partial run that
   is resumed simply finishes the remaining rows. Stamps
   ``tools_vocab_epoch = N+1`` on each rewritten surface (the epoch column is the
   additive #1576 migration; the live exercise sequences it before this runs).

3. **Contract** (rev ``N+2``) — stamps the ledger row's ``contract_rev = N+2``
   (which removes the descriptor's tokens from the read-tolerance map: the shim
   stops remapping and a stale legacy value would then correctly fail
   validation) and carries the in-transaction **abort-guard**: a residue scan
   over every surface that, on finding *any* unmigrated row, raises inside the
   migration transaction. Because the raise is in-transaction,
   ``alembic_version`` never advances to ``N+2`` (the contract rev), so the boot
   gate (#1575), which refuses to tear down a shim whose ``contract_rev`` has not
   been reached everywhere, holds the line. The guard is a *belt* over the
   backfill's *braces*: the backfill should have left zero residue, but a row
   written between backfill and contract (or missed by a surface bug) must abort
   rather than silently contract.

The emitted SQL deliberately matches the merged 0116/0120/0122 shape: a single
``op.execute`` per surface, ``jsonb_array_elements`` predicates, order-preserving
``jsonb_agg``, a ``rename``-vs-``drop`` element transform, and a no-op
``downgrade`` (these rewrites are forward-only). The chain pointers are wired so
the three migrations apply in order on top of the current head.

This module is pure (no DB, no Docker): it reads the descriptor and the on-disk
migration ladder and returns file contents as strings. ``bin/aios-retire`` is a
thin CLI over :func:`generate`.

Ledger schema contract
----------------------
The generated migrations read/write a ``retirement_ledger`` table — the
boot-gate's source of truth (#1575) — with at least these columns::

    domain        text     -- the retirement's domain
    token         text     -- one row per retired token
    phase         text     -- 'expand' then 'contract'
    contract_rev  text     -- NULL until the contract migration stamps it
    sla_days      integer  -- teardown grace after contract_rev runs everywhere

The expand migration inserts one row per token (``phase='expand'``,
``contract_rev=NULL``); the contract migration updates those rows to
``phase='contract'``, ``contract_rev=<N+2>``. The ledger table itself is created
by the registry/boot-gate issue (#1575); this generator only writes rows. The
``INSERT`` is ``ON CONFLICT DO NOTHING`` and the ``UPDATE`` is idempotent so a
re-run of either migration is a no-op.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from aios.retirements import Retirement, Surface

# ── Tunables ──────────────────────────────────────────────────────────────────

#: Surfaces whose tables are high-cardinality enough that a single unbounded
#: ``UPDATE`` would hold row locks too long (the 0066 lesson). The backfill
#: rewrites these in bounded ``ctid`` batches; everything else gets one
#: ``EXISTS``-guarded statement. Keyed by table name (a surface's column does
#: not change its cardinality class).
HIGH_CARDINALITY_TABLES: frozenset[str] = frozenset({"wf_runs", "agent_versions", "sessions"})

#: Rows per batch on a high-cardinality surface. Bounds the lock-hold window;
#: small enough to keep each statement short, large enough to keep the loop
#: count sane on a big table.
DEFAULT_BATCH_SIZE = 5000

#: The ledger table the boot-gate (#1575) owns; this generator only writes rows.
LEDGER_TABLE = "retirement_ledger"

#: The epoch column the backfill stamps (additive migration #1576).
EPOCH_COLUMN = "tools_vocab_epoch"


# ── Result type ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GeneratedMigration:
    """One emitted migration file: its revision, filename, and full source."""

    revision: str
    filename: str
    source: str


@dataclass(frozen=True)
class GeneratedChain:
    """The three lifecycle migrations for one descriptor, in chain order."""

    expand: GeneratedMigration
    backfill: GeneratedMigration
    contract: GeneratedMigration

    def __iter__(self):  # type: ignore[no-untyped-def]
        yield self.expand
        yield self.backfill
        yield self.contract


# ── Revision helpers ──────────────────────────────────────────────────────────


def next_revisions(head: str, count: int = 3) -> list[str]:
    """The next ``count`` zero-padded numeric revisions after ``head``.

    The aios ladder uses zero-padded integer revisions (``0116``, ``0123``, …);
    we continue that scheme. ``head`` must be such a numeric revision.
    """

    m = re.fullmatch(r"(\d+)", head)
    if not m:
        raise ValueError(
            f"head revision {head!r} is not a zero-padded integer; "
            "the generator only extends the numeric aios ladder"
        )
    width = len(head)
    start = int(head)
    return [str(start + i).zfill(width) for i in range(1, count + 1)]


# ── SQL fragment builders ─────────────────────────────────────────────────────


def _quote_sql_str(value: str) -> str:
    """Single-quote a SQL string literal, escaping embedded quotes."""

    return "'" + value.replace("'", "''") + "'"


def _token_list_sql(retirement: Retirement) -> str:
    """``'a', 'b', …`` — the retired tokens as a SQL ``IN`` list."""

    return ", ".join(_quote_sql_str(t) for t in retirement.tokens)


def _exists_predicate(surface: Surface, retirement: Retirement) -> str:
    """A surface's ``EXISTS(...)`` residue predicate for *any* retired token.

    The descriptor's shared ``predicate_sql`` is parameterised by ``:token``;
    for a multi-token retirement we widen it to an ``IN (...)`` over all tokens
    by rewriting the predicate's ``e->>'type' = :token`` into an ``IN`` list.
    This keeps the single shared predicate template authoritative (the column is
    already substituted) while covering every token in one scan.
    """

    token_list = _token_list_sql(retirement)
    # The shared tool_surface predicate compares ``e->>'type' = :token``; widen
    # the bound single token into the full IN list. Fall back to a literal
    # substitution for any other predicate shape.
    widened = re.sub(
        r"=\s*:token\b",
        f"IN ({token_list})",
        surface.predicate_sql,
    )
    if widened == surface.predicate_sql:
        # Predicate did not use the canonical ``= :token`` form; bind the first
        # token literally so the fragment is at least valid for single-token
        # retirements.
        widened = surface.predicate_sql.replace(":token", _quote_sql_str(retirement.tokens[0]))
    return widened


def _element_transform_sql(retirement: Retirement) -> str:
    """The per-element jsonb transform: rename → ``jsonb_set``, drop → filter.

    Returns a SQL ``SELECT ... FROM jsonb_array_elements(...)`` expression body
    that, given a ``tools`` jsonb array, returns the rewritten array. For a
    ``rename`` it maps each retired ``type`` to its successor and dedupes by the
    resulting builtin name (custom/mcp_toolset never deduped), first-occurrence
    wins, order otherwise preserved — the 0116 shape. For a ``drop`` it filters
    out every retired element, order preserved — the 0120/0122 shape.
    """

    if retirement.action == "rename":
        case_arms = "\n".join(
            f"                            WHEN {_quote_sql_str(tok)} THEN {_quote_sql_str(succ)}"
            for tok, succ in retirement.token_map().items()
            if succ is not None
        )
        return f"""
            SELECT coalesce(jsonb_agg(elem ORDER BY ord), '[]'::jsonb)
            FROM (
                SELECT DISTINCT ON (dedup_key) elem, ord
                FROM (
                    SELECT
                        jsonb_set(e, '{{type}}', to_jsonb(new_type)) AS elem,
                        ord,
                        CASE WHEN new_type IN ('custom', 'mcp_toolset')
                             THEN 'pos:' || ord::text
                             ELSE new_type END AS dedup_key
                    FROM jsonb_array_elements(tools) WITH ORDINALITY AS arr(e, ord)
                    CROSS JOIN LATERAL (
                        SELECT CASE e->>'type'
{case_arms}
                            ELSE e->>'type'
                        END AS new_type
                    ) mapped
                ) m
                ORDER BY dedup_key, ord
            ) deduped"""
    # drop
    token_list = _token_list_sql(retirement)
    return f"""
            SELECT coalesce(jsonb_agg(elem ORDER BY ord), '[]'::jsonb)
            FROM jsonb_array_elements(tools) WITH ORDINALITY AS arr(elem, ord)
            WHERE elem->>'type' NOT IN ({token_list})"""


def _function_name(retirement: Retirement, revision: str) -> str:
    """A unique pg function name for this descriptor's backfill transform."""

    slug = re.sub(r"[^a-z0-9]+", "_", retirement.domain.lower()).strip("_")
    verb = "rename" if retirement.action == "rename" else "drop"
    return f"_aios_retire_{slug}_{verb}_{revision}"


# ── Per-migration source builders ─────────────────────────────────────────────


def _module_header(message: str, revision: str, down_revision: str, docstring: str) -> str:
    return f'''"""{message}

{docstring}

Revision ID: {revision}
Revises: {down_revision}

Generated by ``bin/aios-retire`` from the registry descriptor — do not hand-edit;
regenerate from the descriptor instead (#1578, epic #1572).
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "{revision}"
down_revision: str | None = "{down_revision}"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None
'''


def build_expand(retirement: Retirement, revision: str, down_revision: str) -> str:
    """Source for the expand migration: insert the ledger row(s), no data edit."""

    rows = ",\n".join(
        "            ("
        f"{_quote_sql_str(retirement.domain)}, {_quote_sql_str(tok)}, "
        f"'expand', NULL, {int(retirement.sla_days)})"
        for tok in retirement.tokens
    )
    header = _module_header(
        f"Expand: open the {retirement.domain!r} retirement ledger (no data change).",
        revision,
        down_revision,
        "No-op on data. Inserts one ``retirement_ledger`` row per retired token with\n"
        "``phase='expand'``, ``contract_rev=NULL``, and the descriptor's ``sla_days``.\n"
        "While these rows have ``contract_rev IS NULL`` the registry-driven read-\n"
        "tolerance shim keeps remapping the tokens on read; the contract migration\n"
        "(rev N+2) stamps ``contract_rev`` to close that span. ``ON CONFLICT DO\n"
        "NOTHING`` makes the insert re-runnable.",
    )
    return f'''{header}

# (domain, token, phase, contract_rev, sla_days) — one row per retired token.
_LEDGER_ROWS_SQL = """
    INSERT INTO {LEDGER_TABLE} (domain, token, phase, contract_rev, sla_days)
    VALUES
{rows}
    ON CONFLICT (domain, token) DO NOTHING
"""


def upgrade() -> None:
    # Expand is a no-op on persisted data: it only records that this retirement
    # has entered its expand span. The read-tolerance shim does the live work.
    op.execute(_LEDGER_ROWS_SQL)


def downgrade() -> None:
    # Forward-only lifecycle: removing the ledger row would re-open a span the
    # backfill/contract may already have closed. Nothing to reverse.
    pass
'''


def _backfill_statement(surface: Surface, retirement: Retirement, fn: str, revision: str) -> str:
    """One surface's backfill ``op.execute`` block (batched if high-cardinality)."""

    exists = _exists_predicate(surface, retirement)
    null_guard = f"{surface.jsonb_col} IS NOT NULL AND " if surface.nullable else ""
    set_clause = f"{surface.jsonb_col} = {fn}({surface.jsonb_col}), {EPOCH_COLUMN} = {revision}"

    if surface.table in HIGH_CARDINALITY_TABLES:
        # Batched: rewrite at most DEFAULT_BATCH_SIZE rows per statement, looping
        # until no residue remains. ctid-keyed so each batch is a bounded,
        # index-free slice; EXISTS-guarded so it is re-runnable and converges.
        return f'''    # {surface.table}.{surface.jsonb_col}: high-cardinality → batched to bound
    # lock-hold (the 0066 lesson). Loop until no residue remains.
    op.execute("""
        DO $do$
        DECLARE
            updated integer;
        BEGIN
            LOOP
                UPDATE {surface.table} SET {set_clause}
                WHERE ctid IN (
                    SELECT ctid FROM {surface.table}
                    WHERE {null_guard}{exists}
                    LIMIT {DEFAULT_BATCH_SIZE}
                );
                GET DIAGNOSTICS updated = ROW_COUNT;
                EXIT WHEN updated = 0;
            END LOOP;
        END
        $do$;
    """)'''
    return f'''    # {surface.table}.{surface.jsonb_col}: single EXISTS-guarded set-based UPDATE.
    op.execute("""
        UPDATE {surface.table} SET {set_clause}
        WHERE {null_guard}{exists}
    """)'''


def build_backfill(retirement: Retirement, revision: str, down_revision: str) -> str:
    """Source for the backfill migration: batched, EXISTS-pre-checked, epoch-stamping."""

    fn = _function_name(retirement, revision)
    transform = _element_transform_sql(retirement)
    statements = "\n\n".join(
        _backfill_statement(s, retirement, fn, revision) for s in retirement.surfaces
    )
    header = _module_header(
        f"Backfill: rewrite persisted {retirement.domain!r} surfaces to canonical.",
        revision,
        down_revision,
        "EXISTS-pre-checked, batched, set-based rewrite across every surface the\n"
        "descriptor declares. High-cardinality surfaces are rewritten in bounded\n"
        "``ctid`` batches to cap lock-hold (the 0066 lesson); every statement is\n"
        "``EXISTS``-guarded so it no-ops on already-fixed rows and is fully\n"
        f"re-runnable. Stamps ``{EPOCH_COLUMN} = {revision}`` on each rewritten\n"
        "surface. ``downgrade`` is a no-op (forward-only).",
    )
    return f'''{header}


def upgrade() -> None:
    # Order-preserving per-element transform, applied set-based per surface.
    op.execute(r"""
        CREATE FUNCTION {fn}(tools jsonb) RETURNS jsonb
        LANGUAGE sql IMMUTABLE AS $fn${transform}
        $fn$
    """)

{statements}

    op.execute("DROP FUNCTION {fn}(jsonb)")


def downgrade() -> None:
    # Forward-only: canonicalised rows cannot be un-rewritten. Nothing to reverse.
    pass
'''


def _contract_residue_predicate(retirement: Retirement) -> str:
    """A single ``OR``-joined residue predicate across all surfaces for the guard."""

    parts = []
    for s in retirement.surfaces:
        exists = _exists_predicate(s, retirement)
        guard = f"{s.jsonb_col} IS NOT NULL AND " if s.nullable else ""
        parts.append(f"        SELECT count(*) FROM {s.table} WHERE {guard}{exists}")
    return "\n        UNION ALL\n".join(parts)


def build_contract(retirement: Retirement, revision: str, down_revision: str) -> str:
    """Source for the contract migration: stamp the ledger + in-txn abort-guard."""

    residue = _contract_residue_predicate(retirement)
    header = _module_header(
        f"Contract: close the {retirement.domain!r} retirement (stamp ledger + abort-guard).",
        revision,
        down_revision,
        "Stamps the ledger rows ``phase='contract'``, ``contract_rev=<this rev>`` —\n"
        "which drops the descriptor's tokens out of the read-tolerance map, so the\n"
        "shim stops remapping them and a stale legacy value would then correctly\n"
        "fail validation. Carries the in-transaction **abort-guard**: a residue scan\n"
        "over every surface that RAISES on any unmigrated row. Because the raise is\n"
        "in-transaction the migration aborts and ``alembic_version`` never reaches\n"
        "this contract rev, so the boot gate (#1575) holds the line. The guard is a\n"
        "belt over the backfill's braces.",
    )
    update_rows = (
        f"UPDATE {LEDGER_TABLE} SET phase = 'contract', contract_rev = '{revision}'\n"
        f"        WHERE domain = {_quote_sql_str(retirement.domain)}\n"
        f"          AND token IN ({_token_list_sql(retirement)})"
    )
    return f'''{header}


def upgrade() -> None:
    # Belt: abort the whole migration in-transaction if ANY surface still holds a
    # retired token. The backfill (rev N+1) should have left zero residue; a row
    # written between backfill and contract, or missed by a surface bug, MUST
    # abort here rather than silently contract. On abort, alembic_version never
    # advances to this contract rev, so the boot gate (#1575) keeps the shim.
    op.execute("""
        DO $guard$
        DECLARE
            residue bigint;
        BEGIN
            SELECT sum(c) INTO residue FROM (
{residue}
            ) scan(c);
            IF residue IS NOT NULL AND residue > 0 THEN
                RAISE EXCEPTION
                    'retirement contract abort: % unmigrated row(s) remain for {retirement.domain}; '
                    'run the backfill (rev {down_revision}) to convergence before contracting',
                    residue;
            END IF;
        END
        $guard$;
    """)

    # Braces held: stamp the ledger. Idempotent (re-running sets the same values).
    op.execute("""
        {update_rows}
    """)


def downgrade() -> None:
    # Forward-only: un-stamping contract_rev would re-open a closed retirement.
    pass
'''


# ── Public entry point ────────────────────────────────────────────────────────


def generate(
    retirement: Retirement,
    head: str,
    *,
    descriptor_name: str | None = None,
) -> GeneratedChain:
    """Emit the three lifecycle migrations for ``retirement`` on top of ``head``.

    ``head`` is the current single head of the migration ladder (the caller
    resolves it via alembic's ``ScriptDirectory`` so the chain stays linear).
    ``descriptor_name`` is only used to name the generated files; it defaults to
    the descriptor's domain.
    """

    expand_rev, backfill_rev, contract_rev = next_revisions(head, 3)
    name = descriptor_name or retirement.domain
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

    expand = GeneratedMigration(
        revision=expand_rev,
        filename=f"{expand_rev}_retire_{slug}_expand.py",
        source=build_expand(retirement, expand_rev, head),
    )
    backfill = GeneratedMigration(
        revision=backfill_rev,
        filename=f"{backfill_rev}_retire_{slug}_backfill.py",
        source=build_backfill(retirement, backfill_rev, expand_rev),
    )
    contract = GeneratedMigration(
        revision=contract_rev,
        filename=f"{contract_rev}_retire_{slug}_contract.py",
        source=build_contract(retirement, contract_rev, backfill_rev),
    )
    return GeneratedChain(expand=expand, backfill=backfill, contract=contract)


def resolve_descriptor(name: str) -> tuple[Retirement, str]:
    """Resolve a registry descriptor by its module-level constant name.

    ``bin/aios-retire <descriptor>`` names one of the constants in
    :mod:`aios.retirements.registry` (e.g. ``LEGACY_BUILTIN_RENAMES``). Returns
    the descriptor and its canonical name. Raises ``KeyError`` with the available
    names if ``name`` is not a registered descriptor.
    """

    from aios.retirements import registry as _registry

    candidate = getattr(_registry, name, None)
    if isinstance(candidate, Retirement):
        return candidate, name
    available = sorted(
        attr for attr in dir(_registry) if isinstance(getattr(_registry, attr), Retirement)
    )
    raise KeyError(f"no retirement descriptor named {name!r}; available: {', '.join(available)}")


def current_head(migrations_dir: str) -> str:
    """The single head revision of the on-disk migration ladder.

    Resolves via alembic's ``ScriptDirectory`` (the same source the migration-
    chain test trusts) so the generated chain extends the real head and the
    ladder stays linear. Raises if the ladder is not single-headed.
    """

    from alembic.config import Config
    from alembic.script import ScriptDirectory

    cfg = Config()
    cfg.set_main_option("script_location", migrations_dir)
    script = ScriptDirectory.from_config(cfg)
    heads: Sequence[str] = script.get_heads()
    if len(heads) != 1:
        raise RuntimeError(f"expected a single migration head, found {heads}")
    return heads[0]
