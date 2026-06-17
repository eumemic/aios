"""Queries for the de-Goodharted detection-residue gauge (#1328).

The durable substrate behind Plane D of the substrate-different-verdict
invariant (epic #1330, roadmap item 9). ``residue_events`` rows are
CLASSIFICATIONS the ops-agent writes; this module is the ONLY write/read path to
that table. The load-bearing properties, expressed in code:

- **Append-only.** This module exposes INSERT + SELECT only. There is NO
  ``UPDATE``/``DELETE`` against ``residue_events`` anywhere here; the migration's
  ``BEFORE UPDATE OR DELETE`` trigger backstops it structurally.
- **Axis segregation (the de-Goodhart).** Every aggregate read is axis-SCOPED:
  the count helpers and the found-by-finder breakdown take an ``axis`` parameter
  and filter ``WHERE axis = $axis``. NO query in this module ``SUM``s or
  ``count``s across BOTH axes — a rising axis-1 (machine-found mechanical waste)
  is NEVER combined with axis-2 (class migration) into a single ratio, because a
  rising axis-1 is FORBIDDEN as an autonomy-increment justification. A unit test
  greps this module and fails if a cross-axis aggregate appears.
- **residue_kind never inferred.** The axis-2 ingest copies ``residue_kind``
  VERBATIM from a stored ``gate_resume`` signal's ``result`` payload (the
  finder's own stamp, written by ``dev_pipeline.py``'s ``human_gate()`` wrapper);
  it CLASSIFIES (axis, dedupe) but does not DEFINE the kind. No prose parsing.
- **Denominator from an uncorrelated substrate.** The denominators are computed
  here (NOT stored as rows): the axis-1 universe is the host-emitted terminal-run
  count from ``wf_runs``; the axis-2 universe is the GitHub merged-PR set, read by
  the caller (``dev_pipeline.py``) and passed in as a ``MergedPrUniverse`` so the
  fleet-didn't-author-it git substrate, not the ops-agent's account, sets the
  denominator.
- **At-least-once idempotency.** Every INSERT is ON CONFLICT DO NOTHING on
  ``UNIQUE(account_id, idempotency_key)``; the key is ``hash(source_run_id ‖
  residue_kind)`` for observer rows / ``hash(source_gate_nonce)`` for gate rows,
  so a replay inserts ONE row.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, NamedTuple

import asyncpg

from aios.ids import RESIDUE_EVENT, make_id

# ─── frozen vocabularies (mirror the migration CHECKs) ───────────────────────

# The segregated axes. NEVER summed.
AXIS_OBSERVER = 1  # machine-found mechanical waste (axis-1, from the observer)
AXIS_CLASS_MIGRATION = 2  # class migration (axis-2, from gate-resolve)

# The finder vocabulary, frozen in the migration CHECK.
FINDERS = (
    "internal-armed-check",
    "external-world",
    "chairman",
    "seat-incidental",
)

# The four human-in-loop KINDS plus the OPEN ``other`` sentinel. The enum is
# deliberately NOT CHECK-frozen in the table; ``other``-bucket growth is the
# render-side ALARM. ``cannot-determine`` is a SEPARATE fail-loud verdict (a null
# telemetry field), never folded into a clean/ok bucket.
RESIDUE_KINDS = (
    "uncorrelated-detection",
    "deadlock-break",
    "design-judgment",
    "ladder-top-approval",
)
OTHER_KIND = "other"
CANNOT_DETERMINE_KIND = "cannot-determine"
# The accepted set the human_gate() wrapper (dev_pipeline.py) enforces: the four
# kinds plus ``other``. Authored here so the wrapper and the ingest share one
# source of truth.
ACCEPTED_GATE_RESIDUE_KINDS = frozenset(RESIDUE_KINDS) | {OTHER_KIND}

# kind_source provenance (mirrors the migration CHECK).
KIND_SOURCE_GATE = "gate-resolve-payload"
KIND_SOURCE_OBSERVER = "observer"
KIND_SOURCE_MANUAL = "manual"


def _idempotency_key(*parts: str | None) -> str:
    """The at-least-once dedupe key: a stable hash over the provenance parts.

    Gate rows key on ``hash(source_gate_nonce)``; observer rows key on
    ``hash(source_run_id ‖ residue_kind)``. A NUL separator keeps the parts
    unambiguous, and ``None`` parts are skipped so an absent field never collides
    with an empty string.
    """
    joined = "\x00".join(p for p in parts if p is not None)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


# ─── INSERT (append-only; ON CONFLICT DO NOTHING) ────────────────────────────


async def insert_residue_event(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    axis: int,
    finder: str,
    residue_kind: str,
    kind_source: str,
    signature: dict[str, Any],
    idempotency_key: str,
    source_run_id: str | None = None,
    source_gate_nonce: str | None = None,
    issue_ref: str | None = None,
    class_closed_issue: str | None = None,
) -> bool:
    """Append one residue classification, idempotently.

    Returns ``True`` if a row was inserted, ``False`` if a row with the same
    ``(account_id, idempotency_key)`` already existed (ON CONFLICT DO NOTHING —
    the at-least-once replay contract; a re-ingest never double-counts). The
    CHECK constraints (axis/finder/kind_source) reject out-of-vocabulary values
    as an IntegrityError, never a silent coercion.
    """
    row = await conn.fetchrow(
        """
        INSERT INTO residue_events (
            id, account_id, axis, finder, residue_kind, kind_source, signature,
            source_run_id, source_gate_nonce, issue_ref, class_closed_issue,
            idempotency_key
        ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, $10, $11, $12)
        ON CONFLICT (account_id, idempotency_key) DO NOTHING
        RETURNING id
        """,
        make_id(RESIDUE_EVENT),
        account_id,
        axis,
        finder,
        residue_kind,
        kind_source,
        json.dumps(signature),
        source_run_id,
        source_gate_nonce,
        issue_ref,
        class_closed_issue,
        idempotency_key,
    )
    return row is not None


# ─── axis-2 ingest: copy the finder's stamp from a gate_resume signal ────────

# The dev_pipeline gate kinds that are human-in-loop (the human_gate() wrapper
# demands a residue_kind for exactly these). Mirrored from dev_pipeline.py.
HUMAN_IN_LOOP_GATE_KINDS = frozenset({"merge_approval", "design"})


def gate_result_residue_kind(result: Any) -> str | None:
    """Read ``residue_kind`` VERBATIM from a stored ``gate_resume`` ``result``.

    The finder (seat/chairman) stamps the kind at the moment of human judgment
    via the ``human_gate()`` wrapper; this reads it back, copying — never
    inferring from prose. Returns the stamped kind, or ``None`` if absent.
    """
    if isinstance(result, dict):
        kind = result.get("residue_kind")
        if isinstance(kind, str) and kind:
            return kind
    return None


async def ingest_gate_resume_axis2(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    source_gate_nonce: str,
    gate_kind: str,
    result: Any,
    finder: str,
    source_run_id: str | None = None,
    issue_ref: str | None = None,
) -> bool:
    """Ingest one human-in-loop ``gate_resume`` as an axis-2 (class-migration) row.

    Copies ``residue_kind`` from the stored ``result`` payload (the
    stamped-at-source value); ``kind_source`` is therefore
    ``gate-resolve-payload``. Idempotent on ``hash(source_gate_nonce)`` so a
    gate-resume replay inserts ONE row. Returns ``True`` if inserted.

    Only the human-in-loop kinds carry a residue stamp; a non-human-in-loop gate
    (``verify``/``merge_guard``) is not an axis-2 residue event and is skipped
    (returns ``False``). A human-in-loop resume whose ``result`` is MISSING the
    stamp is a contract violation the ``human_gate()`` wrapper foreclosed
    upstream; defensively we refuse to fabricate a kind and raise.
    """
    if gate_kind not in HUMAN_IN_LOOP_GATE_KINDS:
        return False
    kind = gate_result_residue_kind(result)
    if kind is None:
        raise ValueError(
            "gate_resume for human-in-loop kind %r is missing residue_kind — the "
            "human_gate() wrapper should have rejected this resolve; refusing to "
            "infer a kind (#1328)." % (gate_kind,)
        )
    signature = {"gate_kind": gate_kind, "source_gate_nonce": source_gate_nonce}
    return await insert_residue_event(
        conn,
        account_id=account_id,
        axis=AXIS_CLASS_MIGRATION,
        finder=finder,
        residue_kind=kind,
        kind_source=KIND_SOURCE_GATE,
        signature=signature,
        idempotency_key=_idempotency_key(source_gate_nonce),
        source_gate_nonce=source_gate_nonce,
        source_run_id=source_run_id,
        issue_ref=issue_ref,
    )


# ─── axis-1 ingest: the machine-observer verdict (inert until item 8+2) ──────


async def ingest_observer_verdict_axis1(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    source_run_id: str,
    verdict: str,
    signature: dict[str, Any],
) -> bool:
    """Ingest one machine-observer verdict as an axis-1 (mechanical-waste) row.

    The observer (roadmap item 8, reading per-run telemetry from item 2) emits a
    verdict per terminal run:

    - ``anomaly`` → an axis-1 row, ``finder='internal-armed-check'``,
      ``kind_source='observer'``, ``residue_kind='uncorrelated-detection'``,
      ``signature`` = the anomaly fingerprint. Idempotent on
      ``hash(source_run_id ‖ residue_kind)``.
    - ``cannot-determine`` (any null telemetry field — the ``vault_ids:null``
      disease) → a ``cannot-determine`` row so absence is FAIL-LOUD, NEVER
      silently counted as clean. It writes NO ``ok`` row.
    - ``ok`` → NO row (a clean run is the absence of a residue event, not a row).

    Returns ``True`` if a row was inserted. Until the observer ships this path is
    simply never called (the ingest is wired but inert: no telemetry → no
    axis-1 rows), and the denominator is still computed from ``wf_runs``.
    """
    if verdict == "ok":
        return False
    if verdict == "anomaly":
        residue_kind = "uncorrelated-detection"
    elif verdict == "cannot-determine":
        residue_kind = CANNOT_DETERMINE_KIND
    else:
        raise ValueError(
            "unknown observer verdict %r (expected ok/anomaly/cannot-determine) "
            "(#1328)" % (verdict,)
        )
    return await insert_residue_event(
        conn,
        account_id=account_id,
        axis=AXIS_OBSERVER,
        finder="internal-armed-check",
        residue_kind=residue_kind,
        kind_source=KIND_SOURCE_OBSERVER,
        signature=signature,
        idempotency_key=_idempotency_key(source_run_id, residue_kind),
        source_run_id=source_run_id,
    )


# ─── denominators (computed live; NEVER stored; uncorrelated substrate) ──────

# The host-emitted terminal-run statuses — the SAME truth the RunCompletionSource
# matcher fires on. Imported lazily to keep this query module import-light.


async def run_table_denominator(
    conn: asyncpg.Connection[Any], *, account_id: str, window_start: Any
) -> int:
    """The axis-1 denominator: count of terminal ``wf_runs`` over the window.

    "Of N terminal runs the HOST recorded, M had an observer-detected mechanical
    anomaly." The count comes from ``wf_runs`` — the host-emitted substrate the
    fleet did not author — NOT from how many events the ops-agent ingested, so
    the gauge can never be maker==checker. ``window_start`` is a timezone-aware
    datetime lower bound.
    """
    from aios.models.workflows import TERMINAL_RUN_STATUSES

    statuses = list(TERMINAL_RUN_STATUSES)
    row = await conn.fetchrow(
        """
        SELECT count(*) AS n
          FROM wf_runs
         WHERE account_id = $1
           AND status = ANY($2::text[])
           AND created_at >= $3
        """,
        account_id,
        statuses,
        window_start,
    )
    assert row is not None
    return int(row["n"])


class MergedPrUniverse(NamedTuple):
    """The axis-2 denominator carrier: the GitHub merged-PR universe over the
    window, OR a cannot-determine verdict.

    The caller (``dev_pipeline.py``) reads the merged-PR set via ``http_request``
    riding the no-silent-degrade list-read invariant (#1323: a truncated list →
    ``GitHubListIncomplete``). When the read can prove completeness it passes
    ``count`` with ``determinable=True``; when it cannot (truncation / a dangling
    ``rel="next"`` / a GraphQL ``len(nodes) != totalCount``) it passes
    ``determinable=False`` and the render shows ``cannot-determine`` for axis-2 —
    NEVER an implicit short count (the look-green-while-doing-less guard)."""

    count: int | None
    determinable: bool

    @classmethod
    def determined(cls, count: int) -> MergedPrUniverse:
        return cls(count=count, determinable=True)

    @classmethod
    def cannot_determine(cls) -> MergedPrUniverse:
        return cls(count=None, determinable=False)


# ─── axis-scoped found-by-finder breakdowns (NEVER cross-axis) ───────────────


async def found_by_finder(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    axis: int,
    window_start: Any,
) -> dict[str, int]:
    """The found-by-finder breakdown for a SINGLE axis over the window.

    Returns ``{finder: count}`` for the given axis ONLY. The ``WHERE axis = $2``
    is load-bearing: this never aggregates across both axes, so a caller cannot
    accidentally combine axis-1 and axis-2 into one ratio. ``cannot-determine``
    rows are EXCLUDED from the finder breakdown (they are surfaced on their own
    line) so they are never silently counted as a clean finder hit.
    """
    rows = await conn.fetch(
        """
        SELECT finder, count(*) AS n
          FROM residue_events
         WHERE account_id = $1
           AND axis = $2
           AND created_at >= $3
           AND residue_kind <> $4
         GROUP BY finder
        """,
        account_id,
        axis,
        window_start,
        CANNOT_DETERMINE_KIND,
    )
    return {r["finder"]: int(r["n"]) for r in rows}


async def kind_count(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    axis: int,
    residue_kind: str,
    window_start: Any,
) -> int:
    """Count rows of one ``residue_kind`` within a SINGLE axis over the window.

    Used for the ``other``-bucket growth alarm and the ``cannot-determine``
    fail-loud line — each scoped to one axis, never summed across axes.
    """
    row = await conn.fetchrow(
        """
        SELECT count(*) AS n
          FROM residue_events
         WHERE account_id = $1
           AND axis = $2
           AND residue_kind = $3
           AND created_at >= $4
        """,
        account_id,
        axis,
        residue_kind,
        window_start,
    )
    assert row is not None
    return int(row["n"])
