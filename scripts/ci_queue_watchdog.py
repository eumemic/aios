"""Evaluate GitHub workflow runs for an abnormally old master CI run."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: Exit code for "could not read the runs payload" -- distinct from 0 (healthy)
#: and 1 (breach). A blind watchdog must never be indistinguishable from either (#2317).
EXIT_BLIND = 2

_NONTERMINAL = {"queued", "in_progress", "pending", "requested", "waiting"}


@dataclass(frozen=True)
class Breach:
    run_id: int
    age_seconds: int
    p95_seconds: int
    threshold_seconds: int
    html_url: str


@dataclass(frozen=True)
class InsufficientHistory:
    """An unknown verdict caused by too few completed runs."""

    status: str
    completed_runs: int
    required_runs: int


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def evaluate_runs(
    runs: list[dict[str, Any]], *, now: datetime | None = None, sample_size: int = 20
) -> Breach | InsufficientHistory | None:
    """Return a failure verdict for a breach or an indeterminate threshold."""
    now = now or datetime.now(UTC)
    master = [run for run in runs if run.get("head_branch") == "master"]
    pending = [run for run in master if run.get("status") in _NONTERMINAL]
    if not pending:
        return None

    completed = sorted(
        (
            run
            for run in master
            if run.get("status") == "completed" and run.get("created_at") and run.get("updated_at")
        ),
        key=lambda run: _timestamp(run["updated_at"]),
        reverse=True,
    )[:sample_size]
    if len(completed) < sample_size:
        return InsufficientHistory(
            status="unknown",
            completed_runs=len(completed),
            required_runs=sample_size,
        )

    durations = sorted(
        int((_timestamp(run["updated_at"]) - _timestamp(run["created_at"])).total_seconds())
        for run in completed
    )
    p95 = durations[math.ceil(0.95 * len(durations)) - 1]
    oldest = min(pending, key=lambda run: _timestamp(run["created_at"]))
    age = int((now - _timestamp(oldest["created_at"])).total_seconds())
    threshold = 2 * p95
    if age <= threshold:
        return None
    return Breach(
        run_id=int(oldest["id"]),
        age_seconds=age,
        p95_seconds=p95,
        threshold_seconds=threshold,
        html_url=str(oldest.get("html_url", "")),
    )


def _load_runs(path: Path) -> tuple[list[dict[str, Any]] | None, str]:
    """Parse the runs payload; ``(None, reason)`` when it is not a readable runs list.

    GitHub error bodies (401 ``Bad credentials``, rate limit) are VALID JSON
    without ``workflow_runs``. An absent required field is a failed read, never
    an empty result -- defaulting it would assert health that was never
    established (#2317).
    """
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return None, f"runs payload unparseable: {exc}"
    runs = payload.get("workflow_runs") if isinstance(payload, dict) else None
    if not isinstance(runs, list):
        message = payload.get("message") if isinstance(payload, dict) else None
        return None, f"runs payload has no workflow_runs list: {message}"
    return [run for run in runs if isinstance(run, dict)], ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs, reason = _load_runs(args.runs)
    if runs is None:
        # Cannot-determine is NOT "nothing pending" and NOT "breach": say so loudly.
        print(
            f"BLIND: CI queue watchdog could not read runs; check did NOT run: {reason}",
            file=sys.stderr,
        )
        args.output.write_text(json.dumps({"status": "unreadable", "reason": reason}))
        return EXIT_BLIND
    verdict = evaluate_runs(runs)
    args.output.write_text(json.dumps(asdict(verdict) if verdict else None))
    if isinstance(verdict, InsufficientHistory):
        return 2
    return int(isinstance(verdict, Breach))


if __name__ == "__main__":
    raise SystemExit(main())
