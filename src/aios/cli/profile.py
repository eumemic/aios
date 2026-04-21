"""Session profiler — reconstructs per-phase latency from span events.

Consumed by ``aios sessions profile <session_id>`` (issue #132). Pure
functions over the span event stream; no HTTP or I/O.

The session step has two distinct time domains:

* **Inside-step** — strictly nested within ``step_start``/``step_end``:
  ``sweep`` (site=entry), ``context_build``, ``model_request``. These
  shares sum to ~100% of step time.
* **Between-step** — the gap between consecutive steps (``step_end`` →
  next ``step_start``). ``tool_execute`` and ``sweep`` (site=tail) run
  here async, overlapping procrastinate queue wait. The gap's cause is
  **not** the next step's ``cause`` field (Procrastinate coalescing
  can desynchronise them); instead, use the earliest ``wake_deferred``
  since the previous ``step_end`` — that's the wake that actually fired.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from statistics import median
from typing import Any

# Causes that start a new user-or-time-initiated turn. Their between-step
# gap preceding them represents user idle / network wait, not glue overhead.
_TURN_STARTING_CAUSES = frozenset({"message", "inbound_message", "initial_message", "scheduled"})
# Causes whose gap represents intentional delay — subtract delay_seconds.
_DELAYED_CAUSES = frozenset({"scheduled", "reschedule"})


@dataclass(frozen=True, slots=True)
class Span:
    id: str
    seq: int
    event: str
    created_at: datetime
    data: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Pair:
    start: Span
    end: Span
    duration_s: float


@dataclass(frozen=True, slots=True)
class PhaseStats:
    phase: str
    n: int
    total_s: float
    p50_s: float
    p95_s: float
    share_of_step: float  # 0..1


@dataclass(frozen=True, slots=True)
class GapCategory:
    """Between-step gaps grouped by the earliest wake_deferred.cause."""

    name: str  # "same-turn", "cross-turn", "scheduled/reschedule"
    n: int
    total_s: float
    p50_s: float
    # Async work that ran in these gaps (tail sweep overlaps the next queue wait).
    tool_execute_s: float
    sweep_tail_s: float
    # For scheduled/reschedule gaps only: the intentional delay that should
    # be subtracted when attributing orchestration cost.
    intentional_delay_s: float = 0.0


@dataclass(frozen=True, slots=True)
class Profile:
    n_steps: int
    inside: list[PhaseStats]
    step_total_s: float
    gaps: list[GapCategory]
    elapsed_s: float
    user_idle_s: float  # cross-turn gap total
    orchestration_s: float  # non-tool, non-sweep, non-idle residual
    # Unpaired / malformed span diagnostics (non-fatal).
    warnings: list[str] = field(default_factory=list)


# ─── parsing and pairing ─────────────────────────────────────────────────────


def _parse_ts(value: Any) -> datetime:
    """Accept either datetime or ISO 8601 string. FastAPI returns the latter."""
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _parse_spans(raw_events: list[dict[str, Any]]) -> list[Span]:
    spans: list[Span] = []
    for ev in raw_events:
        if ev.get("kind") != "span":
            continue
        data = ev.get("data") or {}
        name = data.get("event")
        if not isinstance(name, str):
            continue
        spans.append(
            Span(
                id=str(ev["id"]),
                seq=int(ev["seq"]),
                event=name,
                created_at=_parse_ts(ev["created_at"]),
                data=data,
            )
        )
    spans.sort(key=lambda s: s.seq)
    return spans


def _pair_spans(spans: list[Span]) -> tuple[dict[str, Pair], list[str]]:
    """Return (pairs_by_start_id, warnings).

    Pairs ``*_end`` back to ``*_start`` via the backpointer convention
    (``<phase>_start_id`` lives on the end event). ``wake_deferred`` is
    singular with no pair. Unpaired ends are reported as warnings but do
    not fail the profile.
    """
    by_id = {s.id: s for s in spans}
    pairs: dict[str, Pair] = {}
    warnings: list[str] = []
    for s in spans:
        if not s.event.endswith("_end"):
            continue
        field_name = f"{s.event.removesuffix('_end')}_start_id"
        start_id = s.data.get(field_name)
        if not isinstance(start_id, str):
            warnings.append(f"seq={s.seq} {s.event}: missing {field_name}")
            continue
        start = by_id.get(start_id)
        if start is None:
            warnings.append(f"seq={s.seq} {s.event}: start id {start_id} not in window")
            continue
        pairs[start_id] = Pair(
            start=start,
            end=s,
            duration_s=(s.created_at - start.created_at).total_seconds(),
        )
    return pairs, warnings


# ─── turn slicing ────────────────────────────────────────────────────────────


def _slice_last_turns(step_pairs: list[Pair], turns: int) -> list[Pair]:
    """Keep only the last ``turns`` turns of steps.

    A turn starts at a ``step_start`` whose ``cause`` is user- or
    time-initiated (message, inbound_message, initial_message, scheduled).
    Subsequent continuation steps (cause=sweep / tool_confirmation /
    custom_tool_result / reschedule) belong to the same turn.
    """
    boundaries = [
        i for i, p in enumerate(step_pairs) if p.start.data.get("cause") in _TURN_STARTING_CAUSES
    ]
    if not boundaries:
        return step_pairs
    cut_at = boundaries[-turns] if turns <= len(boundaries) else boundaries[0]
    return step_pairs[cut_at:]


# ─── gap classification ──────────────────────────────────────────────────────


def _earliest_wake_cause_in(gap_spans: list[Span]) -> tuple[str | None, float]:
    """Earliest wake_deferred.cause in the gap, plus its delay_seconds (0 if absent).

    Multiple wake_deferred events can land in one gap (Procrastinate coalescing);
    the earliest one is the wake that actually fired the next step.
    """
    for s in gap_spans:
        if s.event == "wake_deferred":
            cause = s.data.get("cause")
            delay = s.data.get("delay_seconds") or 0
            return (str(cause) if cause else None, float(delay))
    return (None, 0.0)


def _gap_category_name(cause: str | None) -> str:
    if cause is None:
        return "(no wake)"
    # Delayed causes win over turn-starting: "scheduled" appears in both sets
    # (it starts a new turn AND carries intentional delay), and for gap
    # classification the delay semantics are what matter.
    if cause in _DELAYED_CAUSES:
        return "scheduled/reschedule"
    if cause in _TURN_STARTING_CAUSES:
        return "cross-turn"
    return "same-turn"


# ─── main entry point ───────────────────────────────────────────────────────


def compute_profile(
    raw_events: list[dict[str, Any]],
    *,
    turns: int | None = None,
) -> Profile:
    """Analyse span events and return a :class:`Profile` breakdown."""
    spans = _parse_spans(raw_events)
    pairs, warnings = _pair_spans(spans)

    step_pairs = sorted(
        (p for p in pairs.values() if p.start.event == "step_start"),
        key=lambda p: p.start.seq,
    )
    if turns is not None:
        step_pairs = _slice_last_turns(step_pairs, turns)

    if not step_pairs:
        return Profile(
            n_steps=0,
            inside=[],
            step_total_s=0.0,
            gaps=[],
            elapsed_s=0.0,
            user_idle_s=0.0,
            orchestration_s=0.0,
            warnings=warnings,
        )

    step_total_s = sum(p.duration_s for p in step_pairs)

    # Inside-step phases: pairs whose start sits between a selected step's
    # start and end (by seq).
    step_ranges = [(p.start.seq, p.end.seq) for p in step_pairs]
    inside_durations: dict[str, list[float]] = defaultdict(list)
    for pair in pairs.values():
        if pair.start.event == "step_start":
            continue
        if pair.start.event == "sandbox_provision_start":
            # Rare; skip for the primary phase list but still accounted in totals.
            continue
        if any(lo <= pair.start.seq <= hi for lo, hi in step_ranges):
            phase = pair.start.event.removesuffix("_start")
            inside_durations[phase].append(pair.duration_s)

    inside_stats: list[PhaseStats] = []
    for phase in ("sweep", "context_build", "model_request"):
        durations = inside_durations.get(phase, [])
        if not durations:
            continue
        total = sum(durations)
        inside_stats.append(
            PhaseStats(
                phase=phase,
                n=len(durations),
                total_s=total,
                p50_s=median(durations),
                p95_s=_p95(durations),
                share_of_step=(total / step_total_s) if step_total_s > 0 else 0.0,
            )
        )

    # Between-step gaps: one per consecutive (step_end, next step_start).
    gap_buckets: dict[str, _GapAccumulator] = defaultdict(_GapAccumulator)
    cross_turn_total = 0.0
    scheduled_delay_total = 0.0
    for i in range(len(step_pairs) - 1):
        gap_start = step_pairs[i].end
        gap_end = step_pairs[i + 1].start
        gap_span = [
            s
            for s in spans
            if gap_start.seq < s.seq < gap_end.seq
            and gap_start.created_at <= s.created_at <= gap_end.created_at
        ]
        duration = (gap_end.created_at - gap_start.created_at).total_seconds()
        cause, delay = _earliest_wake_cause_in(gap_span)
        category = _gap_category_name(cause)

        tool_s = sum(
            pairs[s.id].duration_s
            for s in gap_span
            if s.event == "tool_execute_start" and s.id in pairs
        )
        sweep_s = sum(
            pairs[s.id].duration_s
            for s in gap_span
            if s.event == "sweep_start" and s.data.get("site") == "tail" and s.id in pairs
        )

        bucket = gap_buckets[category]
        bucket.durations.append(duration)
        bucket.tool_execute_s += tool_s
        bucket.sweep_tail_s += sweep_s
        bucket.intentional_delay_s += delay if cause in _DELAYED_CAUSES else 0.0
        if category == "cross-turn":
            cross_turn_total += duration
        if cause in _DELAYED_CAUSES:
            scheduled_delay_total += delay

    gap_cats: list[GapCategory] = []
    for name in ("same-turn", "cross-turn", "scheduled/reschedule", "(no wake)"):
        if name not in gap_buckets:
            continue
        cat_bucket = gap_buckets[name]
        if not cat_bucket.durations:
            continue
        gap_cats.append(
            GapCategory(
                name=name,
                n=len(cat_bucket.durations),
                total_s=sum(cat_bucket.durations),
                p50_s=median(cat_bucket.durations),
                tool_execute_s=cat_bucket.tool_execute_s,
                sweep_tail_s=cat_bucket.sweep_tail_s,
                intentional_delay_s=cat_bucket.intentional_delay_s,
            )
        )

    first = step_pairs[0].start
    last = step_pairs[-1].end
    elapsed_s = (last.created_at - first.created_at).total_seconds()

    same_turn_total = gap_buckets.get("same-turn", _GapAccumulator())
    same_turn_duration = sum(same_turn_total.durations)
    orchestration_s = max(
        0.0,
        same_turn_duration - same_turn_total.tool_execute_s - same_turn_total.sweep_tail_s,
    )

    return Profile(
        n_steps=len(step_pairs),
        inside=inside_stats,
        step_total_s=step_total_s,
        gaps=gap_cats,
        elapsed_s=elapsed_s,
        user_idle_s=cross_turn_total,
        orchestration_s=orchestration_s,
        warnings=warnings,
    )


@dataclass
class _GapAccumulator:
    durations: list[float] = field(default_factory=list)
    tool_execute_s: float = 0.0
    sweep_tail_s: float = 0.0
    intentional_delay_s: float = 0.0


def _p95(values: list[float]) -> float:
    """Nearest-rank 95th percentile; tiny samples collapse to max."""
    if not values:
        return 0.0
    ranked = sorted(values)
    idx = min(len(ranked) - 1, round(0.95 * (len(ranked) - 1)))
    return ranked[idx]


# ─── rendering ───────────────────────────────────────────────────────────────


def format_duration(seconds: float) -> str:
    """Human-readable duration: '9.45s', '120ms', '0.02s' — choose the unit
    that keeps the number two-to-four digits where possible."""
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.0f}µs"
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


def render_profile(profile: Profile) -> str:
    """Render a :class:`Profile` as a human-readable multi-section report."""
    if profile.n_steps == 0:
        return "(no steps in range — session may be empty or spans missing)\n"

    lines: list[str] = []

    lines.append(f"INSIDE-STEP (n={profile.n_steps})")
    if not profile.inside:
        lines.append("  (no inside-step phases detected)")
    else:
        for p in profile.inside:
            lines.append(
                f"  {p.phase:<16} n={p.n:<4} p50={format_duration(p.p50_s):<8}"
                f" p95={format_duration(p.p95_s):<8}"
                f" total={format_duration(p.total_s):<8}"
                f" {p.share_of_step * 100:5.1f}%"
            )
        lines.append(
            f"  {'step total':<16} n={profile.n_steps:<4}"
            f" total={format_duration(profile.step_total_s)}"
        )

    lines.append("")
    lines.append("BETWEEN-STEP")
    if not profile.gaps:
        lines.append("  (only one step — no gaps)")
    else:
        for g in profile.gaps:
            suffix = ""
            if g.name == "cross-turn":
                suffix = " (user idle)"
            lines.append(
                f"  {g.name + ':':<24} {g.n} gap(s), total={format_duration(g.total_s)},"
                f" p50={format_duration(g.p50_s)}{suffix}"
            )
            if g.tool_execute_s > 0:
                lines.append(f"    of which tool_execute: {format_duration(g.tool_execute_s)}")
            if g.sweep_tail_s > 0:
                lines.append(f"    of which sweep (tail): {format_duration(g.sweep_tail_s)}")
            if g.intentional_delay_s > 0:
                lines.append(
                    f"    intentional delay (schedule_wake/retry): "
                    f"{format_duration(g.intentional_delay_s)}"
                )

    lines.append("")
    lines.append("WALL-CLOCK")
    elapsed = profile.elapsed_s
    lines.append(f"  total elapsed:   {format_duration(elapsed)}")
    pct = (profile.step_total_s / elapsed * 100) if elapsed > 0 else 0.0
    lines.append(f"  inside steps:    {format_duration(profile.step_total_s)} ({pct:.1f}%)")
    between_total = elapsed - profile.step_total_s
    pct_b = (between_total / elapsed * 100) if elapsed > 0 else 0.0
    lines.append(f"  between steps:   {format_duration(between_total)} ({pct_b:.1f}%)")
    if profile.user_idle_s > 0:
        pct_u = (profile.user_idle_s / elapsed * 100) if elapsed > 0 else 0.0
        lines.append(f"  user idle:       {format_duration(profile.user_idle_s)} ({pct_u:.1f}%)")
    pct_o = (profile.orchestration_s / elapsed * 100) if elapsed > 0 else 0.0
    lines.append(f"  orchestration:   {format_duration(profile.orchestration_s)} ({pct_o:.1f}%)")

    if profile.warnings:
        lines.append("")
        lines.append(f"warnings ({len(profile.warnings)}):")
        for w in profile.warnings[:10]:
            lines.append(f"  {w}")
        if len(profile.warnings) > 10:
            lines.append(f"  … {len(profile.warnings) - 10} more")

    return "\n".join(lines) + "\n"


def profile_to_dict(profile: Profile) -> dict[str, Any]:
    """JSON-mode rendering — mirrors the dataclass structure."""
    return {
        "n_steps": profile.n_steps,
        "elapsed_s": profile.elapsed_s,
        "step_total_s": profile.step_total_s,
        "user_idle_s": profile.user_idle_s,
        "orchestration_s": profile.orchestration_s,
        "inside": [
            {
                "phase": p.phase,
                "n": p.n,
                "total_s": p.total_s,
                "p50_s": p.p50_s,
                "p95_s": p.p95_s,
                "share_of_step": p.share_of_step,
            }
            for p in profile.inside
        ],
        "gaps": [
            {
                "category": g.name,
                "n": g.n,
                "total_s": g.total_s,
                "p50_s": g.p50_s,
                "tool_execute_s": g.tool_execute_s,
                "sweep_tail_s": g.sweep_tail_s,
                "intentional_delay_s": g.intentional_delay_s,
            }
            for g in profile.gaps
        ],
        "warnings": profile.warnings,
    }
