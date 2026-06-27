"""The tools-vocab epoch — the per-row raw-restore belt (#1576, epic #1572).

Every persisted tool-surface row carries a ``tools_vocab_epoch smallint`` stamp
recording the most recent **backfill** (the data migration that rewrites
persisted ``ToolSpec`` rows to canonical vocabulary) the row's blob was known to
satisfy. The stamp is the *raw-restore belt*: it travels INSIDE a
``pg_restore``/volume-swap snapshot, so a chain-bypassing restore of an old DB
self-describes its staleness (its rows read ``epoch < latest-backfill-rev``)
even when it bypasses ``aios migrate`` entirely. It is a belt, not the primary
authority — the content-predicate boot scan (the boot-admission gate) is the
authority; the stamp lets that scan fast-path to ``MIN(epoch) >= backfill_rev``
per table instead of re-scanning every blob.

The single source of truth for "the latest backfill rev" is the **retirement
registry** itself: a backfill is exactly a retirement's ``contract_rev`` (the
data migration that rewrites persisted rows to canonical). The latest such rev
across the ``tool_surface`` domain *is* the current tools-vocab epoch — there is
no parallel constant to drift. :data:`TOOLS_VOCAB_EPOCH` is the integer form of
that rev (alembic revs are zero-padded numeric strings, e.g. ``"0122"`` → 122),
small enough for a ``smallint`` column and monotonic with the migration chain.

The column **default is 0** (see migration 0124): a row written before this
mechanism existed — or restored raw from an old snapshot — reads as epoch 0,
which is ``< latest-backfill-rev`` for any real backfill, so it correctly
self-describes as stale. New write paths stamp :data:`TOOLS_VOCAB_EPOCH`
explicitly so a fresh row is born current.
"""

from __future__ import annotations

from aios.retirements import Retirement
from aios.retirements.registry import REGISTRY, TOOL_SURFACE_DOMAIN


def _rev_to_epoch(rev: str) -> int:
    """Convert an alembic revision id to its integer epoch form.

    AIOS migration revisions are zero-padded decimal strings (``"0116"``,
    ``"0122"``); the epoch is the plain integer (116, 122). Raising on a
    non-numeric rev is deliberate: a backfill that wants to participate in the
    epoch belt MUST be reachable as a ``smallint`` here, so a bespoke / hashed
    rev id is a declaration error, caught at import.
    """

    try:
        return int(rev)
    except ValueError as exc:  # pragma: no cover - guards a declaration error
        raise ValueError(
            f"backfill revision {rev!r} is not a numeric alembic rev; the "
            "tools-vocab epoch belt requires numeric revs so it fits a smallint"
        ) from exc


def latest_backfill_rev(
    domain: str = TOOL_SURFACE_DOMAIN,
    *,
    registry: tuple[Retirement, ...] = REGISTRY,
) -> int:
    """The integer epoch of the latest declared backfill in ``domain``.

    A *backfill* is a retirement's ``contract_rev`` — the data migration that
    rewrites persisted rows of the domain to canonical vocabulary. This returns
    the maximum such rev (as an integer epoch) across every retirement in
    ``domain`` whose contract migration has been declared; retirements still in
    their expand span (``contract_rev IS NULL``) contribute no backfill and are
    skipped. With no declared backfill yet, the epoch floor is ``0`` — the same
    value the column defaults to, so nothing reads as stale before the first
    backfill exists.

    ``registry`` is injectable so tests can scope the lookup to a synthetic set
    of descriptors.
    """

    revs = [
        _rev_to_epoch(r.contract_rev)
        for r in registry
        if r.domain == domain and r.contract_rev is not None
    ]
    return max(revs, default=0)


#: The current tools-vocab epoch: the integer epoch of the latest ``tool_surface``
#: backfill. Write paths stamp this onto ``tools_vocab_epoch`` so fresh rows are
#: born current; the boot scan compares persisted ``MIN(epoch)`` against it.
TOOLS_VOCAB_EPOCH: int = latest_backfill_rev()
