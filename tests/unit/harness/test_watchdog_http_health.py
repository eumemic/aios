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
    assert payload["gc_consecutive_failures"] == 2
    assert payload["gc_last_success_at"] is None
