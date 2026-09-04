from __future__ import annotations

from pathlib import Path

import pytest

from aios.api.routers.health import health
from aios.harness.production_watchdogs import ProductionWatchdogState


@pytest.mark.asyncio
async def test_gc_failures_are_visible_on_real_http_health_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot = tmp_path / "watchdog-health.json"
    monkeypatch.setenv("AIOS_WORKER_WATCHDOG_HEALTH_FILE", str(snapshot))

    ProductionWatchdogState().record_gc(None, 2)

    payload = await health()
    assert payload["gc_health_status"] == "failing"
    assert payload["gc_consecutive_failures"] == 2
    assert payload["gc_last_success_at"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("snapshot_state", ["missing", "unreadable", "malformed"])
async def test_gc_snapshot_read_failure_is_unknown_on_real_http_health_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, snapshot_state: str
) -> None:
    snapshot = tmp_path / "watchdog-health.json"
    monkeypatch.setenv("AIOS_WORKER_WATCHDOG_HEALTH_FILE", str(snapshot))
    if snapshot_state == "unreadable":
        snapshot.mkdir()
    elif snapshot_state == "malformed":
        snapshot.write_text("{not-json", encoding="utf-8")

    payload = await health()

    assert payload["gc_health_status"] == "unknown"
    assert payload["gc_consecutive_failures"] is None
    assert payload["gc_last_success_at"] is None
