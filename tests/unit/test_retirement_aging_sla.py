"""Tests for the aging-SLA anti-rot check (#1580, Part A; epic #1572).

The SLA check is the *forcing function* that kills the #1432 "teardown tracked,
never closed" rot (R5): a descriptor ledger row with ``contract_rev IS NULL``
**actively ages**. Once the read-tolerance shim has sat untorn-down past its
``sla_days`` (measured from the ``introduced_rev`` migration's landing date),
the scheduled check FAILS master and files an issue — the ledger row, not a
comment, is what ages and forces the close-out.

These tests pin the pure, offline aging core. The DB / GitHub side (failing
master + filing the issue) lives in the scheduled workflow; the verdict it acts
on is computed here, deterministically and injectably.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from aios.retirements import Retirement, Surface
from aios.retirements.aging import SlaBreach, iter_sla_breaches, sla_breaches

# A single shared surface; the aging core never inspects surfaces, only the
# contract_rev / introduced_rev / sla_days ledger fields.
_SURF = (Surface(table="agents", jsonb_col="tools", predicate_sql="x = :token"),)


def _rename(
    domain: str,
    *,
    introduced_rev: str,
    contract_rev: str | None,
    sla_days: int,
) -> Retirement:
    return Retirement(
        domain=domain,
        action="rename",
        mappings=(("legacy", "canon"),),
        surfaces=_SURF,
        introduced_rev=introduced_rev,
        contract_rev=contract_rev,
        sla_days=sla_days,
    )


def _now() -> datetime:
    return datetime(2026, 7, 1, tzinfo=UTC)


def _dates(**kv: datetime) -> dict[str, datetime]:
    return dict(kv)


def test_open_descriptor_past_sla_is_a_breach() -> None:
    # contract_rev IS NULL, introduced 40 days ago, sla_days=30 → breach.
    r = _rename("d", introduced_rev="0100", contract_rev=None, sla_days=30)
    introduced = _now() - timedelta(days=40)
    breaches = sla_breaches((r,), rev_dates={"0100": introduced}, now=_now())
    assert len(breaches) == 1
    b = breaches[0]
    assert isinstance(b, SlaBreach)
    assert b.domain == "d"
    assert b.introduced_rev == "0100"
    assert b.sla_days == 30
    assert b.age_days == 40
    assert b.over_by_days == 10


def test_open_descriptor_within_sla_is_not_a_breach() -> None:
    # 20 days old, sla_days=30 → still inside the window, no breach.
    r = _rename("d", introduced_rev="0100", contract_rev=None, sla_days=30)
    introduced = _now() - timedelta(days=20)
    assert sla_breaches((r,), rev_dates={"0100": introduced}, now=_now()) == []


def test_exactly_at_sla_is_not_yet_a_breach() -> None:
    # age == sla_days is the last compliant day; breach is strictly past.
    r = _rename("d", introduced_rev="0100", contract_rev=None, sla_days=30)
    introduced = _now() - timedelta(days=30)
    assert sla_breaches((r,), rev_dates={"0100": introduced}, now=_now()) == []


def test_descriptor_with_contract_rev_never_breaches() -> None:
    # contract_rev IS SET → the data migration is declared, teardown is on the
    # boot-gate's clock, not the aging SLA's. Even ancient, it does not breach.
    r = _rename("d", introduced_rev="0100", contract_rev="0123", sla_days=30)
    introduced = _now() - timedelta(days=9999)
    assert sla_breaches((r,), rev_dates={"0100": introduced}, now=_now()) == []


def test_drop_descriptor_is_subject_to_the_sla_too() -> None:
    # The SLA gates ANY open descriptor, rename or drop — the ledger row ages
    # regardless of action.
    r = Retirement(
        domain="d",
        action="drop",
        mappings=(("legacy", None),),
        surfaces=_SURF,
        introduced_rev="0100",
        contract_rev=None,
        sla_days=15,
    )
    introduced = _now() - timedelta(days=30)
    breaches = sla_breaches((r,), rev_dates={"0100": introduced}, now=_now())
    assert len(breaches) == 1
    assert breaches[0].action == "drop"


def test_missing_rev_date_raises() -> None:
    # A descriptor whose introduced_rev has no resolvable landing date is a
    # configuration error, not a silently-skipped (and thus never-aging) row —
    # fail loud rather than let R5 hide behind a missing date.
    r = _rename("d", introduced_rev="9999", contract_rev=None, sla_days=30)
    with pytest.raises(KeyError):
        sla_breaches((r,), rev_dates={"0100": _now()}, now=_now())


def test_iter_is_lazy_and_sla_breaches_is_the_materialised_list() -> None:
    r1 = _rename("a", introduced_rev="0100", contract_rev=None, sla_days=10)
    r2 = _rename("b", introduced_rev="0101", contract_rev="0200", sla_days=10)
    rev_dates = {
        "0100": _now() - timedelta(days=100),
        "0101": _now() - timedelta(days=100),
    }
    it = iter_sla_breaches((r1, r2), rev_dates=rev_dates, now=_now())
    materialised = list(it)
    assert [b.domain for b in materialised] == ["a"]
    assert sla_breaches((r1, r2), rev_dates=rev_dates, now=_now()) == materialised


def test_default_now_is_utc_now(monkeypatch: pytest.MonkeyPatch) -> None:
    # now defaults to wall-clock UTC; a long-open descriptor breaches without an
    # explicit now.
    r = _rename("d", introduced_rev="0100", contract_rev=None, sla_days=1)
    introduced = datetime.now(UTC) - timedelta(days=5)
    breaches = sla_breaches((r,), rev_dates={"0100": introduced})
    assert len(breaches) == 1


def test_real_registry_is_scannable_with_resolved_dates() -> None:
    # The seeded registry is iterable by the aging core with real per-rev dates;
    # this is the shape the workflow drives (registry + git-resolved rev dates).
    from aios.retirements.registry import REGISTRY

    # Resolve a date for every introduced_rev present in the registry; give them
    # all "now" so nothing breaches — we are pinning that the registry shape
    # feeds the core cleanly, not the live verdict.
    revs = {r.introduced_rev for r in REGISTRY}
    rev_dates = {rev: _now() for rev in revs}
    # Should not raise (every introduced_rev resolves) and produce a list.
    assert sla_breaches(REGISTRY, rev_dates=rev_dates, now=_now()) == []
