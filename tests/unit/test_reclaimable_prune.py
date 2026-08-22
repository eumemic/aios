from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from aios.db import queries
from aios.harness import reclaimable_prune


@pytest.mark.asyncio
async def test_archival_feeder_failure_degrades_sweep_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = SimpleNamespace(
        reclaimable_prune_enabled=True,
        wf_runs_retention_days=30,
        archived_definition_retention_days=30,
        wf_runs_archive_grace_days=7,
        reclaimable_prune_batch_rows=100,
    )
    monkeypatch.setattr(reclaimable_prune, "get_settings", lambda: settings)

    @asynccontextmanager
    async def acquire() -> Any:
        yield object()

    pool = SimpleNamespace(acquire=acquire)
    calls: list[str] = []

    async def fail_archival(*_args: Any, **_kwargs: Any) -> int:
        calls.append("runs_archival")
        raise RuntimeError("archival feeder unavailable")

    def successful(family: str, count: int) -> Any:
        async def prune(*_args: Any, **_kwargs: Any) -> int:
            calls.append(family)
            return count

        return prune

    monkeypatch.setattr(queries, "reconcile_terminal_archival_batch", fail_archival)
    monkeypatch.setattr(queries, "prune_archived_runs", successful("runs", 2))
    monkeypatch.setattr(
        queries,
        "prune_unpinned_archived_agents",
        successful("agents", 3),
    )
    monkeypatch.setattr(
        queries,
        "prune_unpinned_archived_workflows",
        successful("workflows", 5),
    )
    monkeypatch.setattr(
        queries,
        "prune_unpinned_archived_skills",
        successful("skills", 7),
    )

    result = await reclaimable_prune.sweep_reclaimable_ephemera(pool)

    assert calls == ["runs_archival", "runs", "agents", "workflows", "skills"]
    assert result.total == 17
    assert result.degraded
    assert result.failed_families == ("runs_archival",)
    assert result != reclaimable_prune.PruneResult(2, 3, 5, 7)
