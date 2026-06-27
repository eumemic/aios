"""Pattern-retirement lifecycle — the declarative core (#1573, epic #1572).

A :class:`Retirement` is the single load-bearing object of the whole
pattern-retirement mechanism: everything downstream — the read-tolerance
validator, the boot-gate, and the data-migration generator (built in dependent
issues) — is *generated or checked from* a descriptor, never hand-maintained in
parallel. Build the descriptor first; the rest plugs in.

A retirement names a *token* (a value that used to be legal in some persisted
domain) that is going away, the *action* taken on persisted occurrences
(``rename`` to a successor, or ``drop`` outright), and the exhaustive list of
*surfaces* — concrete ``(table, jsonb_col, predicate_sql)`` triples — where that
token can be sitting in the database. The surface list is what makes a
retirement total: the original ad-hoc tool-rename retirement (#1419 / migration
0116) enumerated only six tool surfaces and silently missed the seventh
(``connectors.tools_schema``); declaring surfaces *as data* on the descriptor
makes that hole impossible to leave open by omission.

This module is the declarative core only — importing it has no behavioral
effect. See :mod:`aios.retirements.registry` for the seeded descriptors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

RetirementAction = Literal["drop", "rename"]
"""What a retirement does to a persisted occurrence of its token.

``rename``
    The token has a canonical successor; persisted occurrences are rewritten in
    place to :attr:`Retirement.successor` (e.g. ``invoke``→``call_session``).
``drop``
    The token has *no* successor; persisted occurrences are removed outright
    (e.g. ``complete_goal``/``fail_goal``, which fold into the general
    ``return``/``error`` step verbs, not into any model-listed builtin).
"""


@dataclass(frozen=True)
class Surface:
    """One concrete place in the database where a retired token can be sitting.

    A surface is a ``(table, jsonb_col, predicate_sql)`` triple:

    * ``table`` — the relation, e.g. ``agents`` or ``connectors``.
    * ``jsonb_col`` — the JSONB column holding the domain's values, e.g.
      ``tools`` or ``tools_schema``.
    * ``predicate_sql`` — a SQL boolean fragment, parameterised by ``:token``,
      that is true for a row IFF the token occurs in ``jsonb_col``. The
      generator substitutes ``<jsonb_col>`` and binds ``:token`` so a single
      shared predicate template serves every surface of a domain.

    ``nullable`` records that ``jsonb_col`` may be ``NULL`` (e.g.
    ``sessions.tools``, populated only on frozen child sessions); the migration
    generator must guard such columns with ``IS NOT NULL`` before iterating.
    """

    table: str
    jsonb_col: str
    predicate_sql: str
    nullable: bool = False

    def __post_init__(self) -> None:
        if not self.table:
            raise ValueError("Surface.table must be non-empty")
        if not self.jsonb_col:
            raise ValueError("Surface.jsonb_col must be non-empty")
        if not self.predicate_sql:
            raise ValueError("Surface.predicate_sql must be non-empty")


@dataclass(frozen=True)
class Retirement:
    """A declared, data-only description of a pattern being retired.

    Fields:

    * ``domain`` — the value space the token lives in (e.g. ``"tool_surface"``
      for persisted ``ToolSpec`` ``type`` values). A domain registers its
      surfaces + predicate template *once*; every retirement in that domain
      reuses them. New domains (a schema field, a config key, an enum value)
      register their own surfaces + predicate the same way, so any pattern plugs
      in without bespoke retirement code.
    * ``token`` — the value being retired. For a multi-token retirement (e.g.
      the #1419 rename batch) see :attr:`tokens`; ``token`` is the single-token
      convenience for ``drop`` retirements with no successor.
    * ``action`` — :data:`RetirementAction`.
    * ``successor`` — the canonical replacement for a ``rename``; ``None`` for a
      ``drop``.
    * ``surfaces`` — the exhaustive list of :class:`Surface` triples where the
      token can be persisted. Validated non-empty.
    * ``introduced_rev`` — the migration revision that first made this
      retirement's read-tolerance live (the shim ships with it).
    * ``contract_rev`` — the migration revision that rewrites persisted rows to
      canonical (the data migration). ``None`` while the contract migration has
      not been written yet; the boot-gate uses it to decide when the shim may be
      torn down.
    * ``sla_days`` — how long the read-tolerance shim must remain after
      ``contract_rev`` has run everywhere before teardown is allowed.

    ``rename`` retirements describe a *batch* of ``token → successor`` mappings
    via :attr:`mappings`; the single ``token``/``successor`` pair is the common
    case. A retirement is internally consistent: a ``rename`` has a successor
    for every token and a ``drop`` has none.
    """

    domain: str
    action: RetirementAction
    surfaces: tuple[Surface, ...]
    introduced_rev: str
    contract_rev: str | None = None
    sla_days: int = 0
    # Single-token form (drop, or a one-token rename). For a multi-token rename
    # batch, populate ``mappings`` instead and leave these ``None``.
    token: str | None = None
    successor: str | None = None
    # Multi-token rename batch: ordered ``token -> successor`` (or ``token ->
    # None`` for drop). The seeded #1419 rename retirement carries five tokens
    # that fold into three successors here.
    mappings: tuple[tuple[str, str | None], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.domain:
            raise ValueError("Retirement.domain must be non-empty")
        if not self.surfaces:
            raise ValueError(
                f"Retirement({self.domain!r}) must register at least one surface"
            )
        if self.token is None and not self.mappings:
            raise ValueError(
                f"Retirement({self.domain!r}) must declare a token or mappings"
            )
        if self.token is not None and self.mappings:
            raise ValueError(
                f"Retirement({self.domain!r}) declares both a single token and "
                "a mappings batch; use one form"
            )
        # Normalise to the canonical token->successor mapping and re-check
        # action consistency against it.
        for tok, succ in self.token_map().items():
            if self.action == "rename" and succ is None:
                raise ValueError(
                    f"Retirement({self.domain!r}) action=rename but token "
                    f"{tok!r} has no successor"
                )
            if self.action == "drop" and succ is not None:
                raise ValueError(
                    f"Retirement({self.domain!r}) action=drop but token {tok!r} "
                    f"maps to successor {succ!r}"
                )

    def token_map(self) -> dict[str, str | None]:
        """Canonical ``token -> successor`` mapping for this retirement.

        Folds the single-token and batch forms into one dict. For a ``drop``,
        every value is ``None``; for a ``rename``, every value is the canonical
        successor (note multiple tokens may fold to the same successor, e.g.
        ``invoke_workflow``/``create_run``/``await_run`` → ``call_workflow``).
        """

        if self.mappings:
            return dict(self.mappings)
        return {self.token: self.successor}  # type: ignore[dict-item]

    @property
    def tokens(self) -> tuple[str, ...]:
        """All tokens this retirement covers (insertion order preserved)."""

        return tuple(self.token_map().keys())
