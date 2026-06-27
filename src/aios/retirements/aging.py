"""Aging-SLA anti-rot check — the descriptor ledger row that actively ages (#1580).

Part A of the two ops-agent-owned anti-rot checks (epic #1572). This module is
the **forcing function** against the #1432 "teardown tracked, never closed" rot
(R5): a retirement descriptor whose ``contract_rev IS NULL`` is an *open*
teardown — the read-tolerance shim is still load-bearing because the data
migration that would let it be torn down has not been declared. Left untracked,
such a row sits forever; a comment promising "we'll write the migration" rots the
moment the author moves on.

The descriptor ledger row is the tracker, and here it **ages**: measured from the
landing date of its ``introduced_rev`` migration, an open descriptor that has sat
past ``sla_days`` is an SLA breach. The scheduled CI/audit job
(``.github/workflows/retirement-aging-sla.yml``) computes these breaches, FAILS
master, and files an issue — so the ledger row, not a comment, is what forces the
close-out (write the contract migration, stamp ``contract_rev``, or consciously
bump ``sla_days`` with a reason).

This module is pure and offline by design: it takes the registry, a
``rev -> landing-date`` map (the workflow resolves these from git; tests inject
them), and ``now``, and returns the breaches. No DB, no network, no clock except
the injectable ``now`` — so the verdict is deterministic and unit-testable, and
the workflow only has to resolve dates and act on the result.

Why measure from ``introduced_rev`` and not ``contract_rev``: a breach is about a
shim that has stayed *open* too long. ``contract_rev`` is precisely the field
whose being ``None`` defines "open"; once it is set the descriptor has left the
SLA's jurisdiction (teardown timing then rides the boot-gate's post-contract
clock, the ``sla_days``-after-``contract_rev`` window described on the
descriptor). So the aging clock starts when the shim was *introduced* and runs
for exactly as long as ``contract_rev`` stays ``None``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from aios.retirements import Retirement, RetirementAction
from aios.retirements.registry import REGISTRY

#: Migrations live here; a descriptor's ``introduced_rev`` is the numeric prefix
#: of exactly one file (e.g. ``0116`` → ``0116_normalize_legacy_tool_names.py``).
_MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "migrations" / "versions"


@dataclass(frozen=True)
class SlaBreach:
    """One open descriptor that has aged past its ``sla_days``.

    ``age_days`` is whole days from the ``introduced_rev`` landing date to
    ``now``; ``over_by_days`` is ``age_days - sla_days`` (always ``>= 1`` for a
    breach). ``introduced_rev`` is carried so the filed issue can name the
    migration whose shim is overdue.
    """

    domain: str
    action: RetirementAction
    introduced_rev: str
    tokens: tuple[str, ...]
    sla_days: int
    age_days: int
    over_by_days: int


def iter_sla_breaches(
    registry: tuple[Retirement, ...] = REGISTRY,
    *,
    rev_dates: dict[str, datetime],
    now: datetime | None = None,
) -> Iterator[SlaBreach]:
    """Yield an :class:`SlaBreach` for every open descriptor aged past its SLA.

    A descriptor breaches IFF:

    * ``contract_rev IS NULL`` — the shim is still open (no data migration
      declared), so it is on the *aging* clock rather than the post-contract
      teardown clock; and
    * ``age_days > sla_days`` — strictly past, so the SLA day itself is still
      compliant (a breach is being *over* the line, not on it).

    ``rev_dates`` maps each descriptor's ``introduced_rev`` to the wall-clock
    instant that migration landed (the workflow resolves these from git; tests
    inject them). A descriptor whose ``introduced_rev`` is absent from
    ``rev_dates`` raises :class:`KeyError` — an unresolvable landing date is a
    configuration error, never a silently-skipped (and thus never-aging) row;
    failing loud is the whole point of an anti-rot forcing function.

    ``now`` defaults to the current UTC time and is injectable for deterministic
    tests.
    """

    moment = now or datetime.now(UTC)
    for retirement in registry:
        if retirement.contract_rev is not None:
            # contract_rev declared → off the aging SLA's jurisdiction.
            continue
        introduced = rev_dates[retirement.introduced_rev]
        age_days = _whole_days(introduced, moment)
        if age_days <= retirement.sla_days:
            continue
        yield SlaBreach(
            domain=retirement.domain,
            action=retirement.action,
            introduced_rev=retirement.introduced_rev,
            tokens=retirement.tokens,
            sla_days=retirement.sla_days,
            age_days=age_days,
            over_by_days=age_days - retirement.sla_days,
        )


def sla_breaches(
    registry: tuple[Retirement, ...] = REGISTRY,
    *,
    rev_dates: dict[str, datetime],
    now: datetime | None = None,
) -> list[SlaBreach]:
    """Materialised :func:`iter_sla_breaches` — the list the workflow acts on."""

    return list(iter_sla_breaches(registry, rev_dates=rev_dates, now=now))


def _whole_days(start: datetime, end: datetime) -> int:
    """Whole elapsed days from ``start`` to ``end`` (floor of the timedelta).

    Both instants are expected timezone-aware; a naive ``start`` (e.g. a git
    date parsed without an offset) is treated as UTC so the subtraction never
    raises on mixed awareness.
    """

    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)
    return (end - start).days


# ---------------------------------------------------------------------------
# Date resolution + CLI driver for the scheduled SLA workflow.
#
# The aging core above is pure (it takes ``rev_dates``). The workflow needs to
# resolve those dates from the repo: a descriptor's ``introduced_rev`` is the
# numeric prefix of exactly one migration file, and that file's *git add* date
# (first commit touching it) is when the shim landed. We use the committer date
# of the earliest commit on the file's history — the moment the read-tolerance
# became live on master.
# ---------------------------------------------------------------------------


def _migration_path_for_rev(rev: str, *, migrations_dir: Path = _MIGRATIONS_DIR) -> Path:
    """The single migration file whose name begins ``<rev>_``.

    Raises :class:`FileNotFoundError` if no file (an unresolvable rev is a
    configuration error, never a silently-skipped row) or :class:`ValueError`
    if more than one matches (an ambiguous prefix).
    """

    matches = sorted(migrations_dir.glob(f"{rev}_*.py"))
    if not matches:
        raise FileNotFoundError(f"no migration file for introduced_rev {rev!r} in {migrations_dir}")
    if len(matches) > 1:
        raise ValueError(f"ambiguous introduced_rev {rev!r}: {[m.name for m in matches]}")
    return matches[0]


def _git_landed_date(path: Path) -> datetime:
    """Committer date of the earliest commit touching ``path`` (when it landed).

    ``git log --reverse`` lists history oldest-first; the first line is the
    commit that introduced the file. Returns a timezone-aware UTC datetime.
    """

    out = subprocess.run(
        ["git", "log", "--reverse", "--format=%cI", "--", str(path)],
        cwd=path.parent,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    first = out.splitlines()[0] if out else ""
    if not first:
        raise RuntimeError(f"no git history for {path}")
    return datetime.fromisoformat(first).astimezone(UTC)


def resolve_rev_dates(
    registry: tuple[Retirement, ...] = REGISTRY,
    *,
    migrations_dir: Path = _MIGRATIONS_DIR,
) -> dict[str, datetime]:
    """``introduced_rev -> landing date`` for every descriptor in ``registry``.

    Resolves each ``introduced_rev`` to its migration file and that file's git
    landing date. This is the bridge the workflow uses to feed the pure aging
    core; it is the only part that touches the repo/git.
    """

    revs = {r.introduced_rev for r in registry}
    return {rev: _git_landed_date(_migration_path_for_rev(rev, migrations_dir=migrations_dir)) for rev in revs}


def _main(argv: list[str]) -> int:
    """Emit the SLA verdict as JSON and exit nonzero on any breach.

    Drives the scheduled ``retirement-aging-sla`` workflow: resolves landing
    dates from git, computes breaches against the live registry, prints
    ``{"breaches": [...]}``, and returns ``1`` if any descriptor has aged past
    its SLA (so the workflow step fails master) or ``0`` if all are compliant.
    """

    rev_dates = resolve_rev_dates()
    breaches = sla_breaches(REGISTRY, rev_dates=rev_dates)
    payload = {"breaches": [asdict(b) for b in breaches]}
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 1 if breaches else 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
