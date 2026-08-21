"""Contract tests for finite Code Validation jobs and its queue watchdog."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml
from scripts.ci_queue_watchdog import InsufficientHistory, evaluate_runs

_ROOT = Path(__file__).parents[2]


def test_every_code_validation_job_has_a_finite_timeout() -> None:
    workflow = yaml.safe_load((_ROOT / ".github/workflows/code-validation.yml").read_text())
    assert {name: job.get("timeout-minutes") for name, job in workflow["jobs"].items()} == {
        "detect": 10,
        "lint": 20,
        "unit": 40,
        "connectors": 30,
        "integration": 40,
        "e2e": 40,
    }


def test_watchdog_flags_oldest_nonterminal_master_run_past_twice_p95() -> None:
    now = datetime(2026, 8, 20, 12, tzinfo=UTC)
    completed = [
        {
            "id": n,
            "status": "completed",
            "created_at": (now - timedelta(minutes=30 + n)).isoformat(),
            "updated_at": (now - timedelta(minutes=10 + n)).isoformat(),
            "head_branch": "master",
        }
        for n in range(1, 21)
    ]
    pending = {
        "id": 99,
        "status": "in_progress",
        "created_at": (now - timedelta(minutes=41)).isoformat(),
        "updated_at": now.isoformat(),
        "head_branch": "master",
        "html_url": "https://github.example/runs/99",
    }

    verdict = evaluate_runs([pending, *completed], now=now)

    assert verdict is not None
    assert verdict.run_id == 99
    assert verdict.age_seconds == 41 * 60
    assert verdict.p95_seconds == 20 * 60


def test_watchdog_ignores_healthy_and_non_master_pending_runs() -> None:
    now = datetime(2026, 8, 20, 12, tzinfo=UTC)
    completed = [
        {
            "id": n,
            "status": "completed",
            "created_at": (now - timedelta(minutes=30)).isoformat(),
            "updated_at": (now - timedelta(minutes=10)).isoformat(),
            "head_branch": "master",
        }
        for n in range(20)
    ]
    runs = [
        {
            "id": 90,
            "status": "queued",
            "created_at": (now - timedelta(minutes=39)).isoformat(),
            "head_branch": "master",
        },
        {
            "id": 91,
            "status": "in_progress",
            "created_at": (now - timedelta(hours=2)).isoformat(),
            "head_branch": "feature",
        },
        *completed,
    ]

    assert evaluate_runs(runs, now=now) is None


def test_watchdog_reports_unknown_when_pending_run_has_insufficient_history() -> None:
    now = datetime(2026, 8, 20, 12, tzinfo=UTC)
    completed = [
        {
            "id": n,
            "status": "completed",
            "created_at": (now - timedelta(minutes=30 + n)).isoformat(),
            "updated_at": (now - timedelta(minutes=10 + n)).isoformat(),
            "head_branch": "master",
        }
        for n in range(1, 20)
    ]
    pending = {
        "id": 99,
        "status": "queued",
        "created_at": (now - timedelta(hours=24)).isoformat(),
        "head_branch": "master",
    }

    verdict = evaluate_runs([pending, *completed], now=now)

    assert verdict == InsufficientHistory(
        status="unknown", completed_runs=19, required_runs=20
    )


def test_watchdog_cli_fails_and_writes_unknown_for_insufficient_history(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    runs = [
        {
            "id": n,
            "status": "completed",
            "created_at": (now - timedelta(minutes=30 + n)).isoformat(),
            "updated_at": (now - timedelta(minutes=10 + n)).isoformat(),
            "head_branch": "master",
        }
        for n in range(1, 20)
    ]
    runs.append(
        {
            "id": 99,
            "status": "queued",
            "created_at": (now - timedelta(hours=24)).isoformat(),
            "head_branch": "master",
        }
    )
    input_path = tmp_path / "runs.json"
    output_path = tmp_path / "result.json"
    input_path.write_text(json.dumps({"workflow_runs": runs}))

    result = subprocess.run(
        [
            sys.executable,
            str(_ROOT / "scripts/ci_queue_watchdog.py"),
            str(input_path),
            "--output",
            str(output_path),
        ],
        check=False,
    )

    assert result.returncode == 2
    assert json.loads(output_path.read_text()) == {
        "status": "unknown",
        "completed_runs": 19,
        "required_runs": 20,
    }
