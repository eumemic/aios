"""Database-free behavioural tests for run-journal deletion eligibility."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from aios.db.queries.prune import prune_archived_runs


@dataclass
class _Run:
    archived_age_days: int | None
    status: str
    has_summary: bool
    journal_pruned: bool = False
    events: int = 1
    signals: int = 0


class _PruneConnection:
    """Small executable model of the wf_runs predicates used by the prune SQL.

    Unlike an AsyncMock SQL-string assertion, this lets the negative fixtures flow
    through the real prune control flow. Omitting a predicate from either query
    omits it from this model too, making guard-removal mutations observable.
    """

    def __init__(self, runs: dict[str, _Run], *, mutate_after_fetch: bool = False) -> None:
        self.runs = runs
        self.mutate_after_fetch = mutate_after_fetch
        self.candidates: list[str] = []
        self.delete_attempts: list[str] = []

    @staticmethod
    def _eligible(sql: str, run: _Run, retention_days: int) -> bool:
        checks = (
            ("archived_at IS NOT NULL", run.archived_age_days is not None),
            (
                "archived_at < now() - make_interval(days => $1)",
                run.archived_age_days is not None and run.archived_age_days > retention_days,
            ),
            ("terminal_summary IS NOT NULL", run.has_summary),
            ("journal_pruned_at IS NULL", not run.journal_pruned),
            (
                "status IN ('completed','errored','cancelled')",
                run.status in {"completed", "errored", "cancelled"},
            ),
        )
        return all(value for predicate, value in checks if predicate in sql)

    async def fetch(self, sql: str, retention_days: int, row_limit: int) -> list[dict[str, str]]:
        self.candidates = [
            run_id for run_id, run in self.runs.items() if self._eligible(sql, run, retention_days)
        ][:row_limit]
        if self.mutate_after_fetch:
            # Model rows becoming unsafe between candidate selection and child DELETE.
            self.runs[self.candidates[0]].archived_age_days = None
            self.runs[self.candidates[1]].status = "running"
            self.runs[self.candidates[2]].archived_age_days = 1
            self.runs[self.candidates[3]].has_summary = False
        return [{"id": run_id} for run_id in self.candidates]

    async def execute(self, sql: str, *args: Any) -> str:
        if sql.startswith("DELETE"):
            run_id, retention_days, row_limit = args
            self.delete_attempts.append(run_id)
            run = self.runs[run_id]
            # DELETE uses $2 for retention, while the candidate query uses $1.
            normalized_sql = sql.replace("days => $2", "days => $1")
            deleted = 0
            if self._eligible(normalized_sql, run, retention_days):
                field = "events" if "wf_run_events" in sql else "signals"
                deleted = min(getattr(run, field), row_limit)
                setattr(run, field, getattr(run, field) - deleted)
            return f"DELETE {deleted}"
        return "UPDATE 1"

    async def fetchval(self, _sql: str, run_id: str) -> bool:
        run = self.runs[run_id]
        return bool(run.events or run.signals)


def _unsafe_runs() -> dict[str, _Run]:
    return {
        "not-archived": _Run(None, "completed", True),
        "not-terminal": _Run(90, "running", True),
        "too-young": _Run(1, "completed", True),
        "summary-less": _Run(90, "completed", False),
    }


@pytest.mark.asyncio
async def test_run_journal_prune_refuses_each_ineligible_run_without_postgres() -> None:
    runs = {**_unsafe_runs(), "eligible": _Run(90, "completed", True)}
    conn = _PruneConnection(runs)

    assert await prune_archived_runs(conn, retention_days=30) == 1

    assert conn.candidates == ["eligible"]
    assert all(runs[run_id].events == 1 for run_id in _unsafe_runs())
    assert runs["eligible"].events == 0


@pytest.mark.asyncio
async def test_child_delete_rechecks_eligibility_if_candidates_become_unsafe() -> None:
    runs = {
        "becomes-unarchived": _Run(90, "completed", True),
        "becomes-running": _Run(90, "completed", True),
        "becomes-young": _Run(90, "completed", True),
        "loses-summary": _Run(90, "completed", True),
    }
    conn = _PruneConnection(runs, mutate_after_fetch=True)

    assert await prune_archived_runs(conn, retention_days=30) == 0

    assert set(conn.delete_attempts) == set(runs)
    assert all(run.events == 1 for run in runs.values())
