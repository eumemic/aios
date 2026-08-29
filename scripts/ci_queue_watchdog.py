"""Evaluate GitHub workflow runs for an abnormally old master CI run."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.runs.read_text())
    verdict = evaluate_runs(payload["workflow_runs"])
    args.output.write_text(json.dumps(asdict(verdict) if verdict else None))
    if isinstance(verdict, InsufficientHistory):
        return 2
    return int(isinstance(verdict, Breach))


if __name__ == "__main__":
    raise SystemExit(main())
