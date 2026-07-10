from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aios.api.routers import health
from aios.db.queries.events import calibration_telemetry
from aios.harness.tokens import CONTENT_CLASSES


class _Conn:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.query = ""

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.query = query
        return self.rows


async def test_calibration_telemetry_fitted_r_eff_is_composition_weighted(
    monkeypatch: Any,
) -> None:
    """``fitted_r_eff`` must be the composition-weighted blend the windower
    consumes, NOT the unweighted mean of the per-class coefficients.

    Uses NONUNIFORM coefficients AND a skewed composition so the two values
    diverge sharply — the arithmetic-mean bug (invisible under the prior
    symmetric-coefficient test) would fail this assertion.
    """
    # Skewed mix: ~all local mass in "text", a little in "tool_result".
    mass = {c: 0.0 for c in CONTENT_CLASSES}
    mass["text"] = 900.0
    mass["tool_result"] = 100.0
    row: dict[str, Any] = {
        "model": "model-a",
        "mean_ratio": 1.8,
        "p50_ratio": 1.75,
        "n_samples": 42,
    }
    row.update({f"mass_{c}": m for c, m in mass.items()})
    conn = _Conn([row])

    # Nonuniform coefficients. Unweighted mean = 6.5 / 6 ≈ 1.0833; the correct
    # composition-weighted blend over the skewed mix
    # = (2.0*900 + 0.5*100) / 1000 = 1.85 — close to the measured 1.8, whereas
    # the old mean (1.0833) would falsely trip the ops-agent's divergence band.
    coefficients = {
        "system": 1.0,
        "tools": 1.0,
        "text": 2.0,
        "tool_result": 0.5,
        "thinking": 1.0,
        "tool_use": 1.0,
    }

    async def _fit(_conn: Any, model: str, *, account_id: str) -> tuple[dict[str, float], int]:
        assert model == "model-a"
        return dict(coefficients), 19

    monkeypatch.setattr("aios.db.queries.events.model_token_class_ratio_fit", _fit)
    result = await calibration_telemetry(conn)

    assert "created_at >= now() - interval '24 hours'" in conn.query
    assert "input_tokens')::bigint > 0" in conn.query

    entry = result["model-a"]
    assert entry["fitted_r_eff"] == pytest.approx(1.85)
    # Regression guard: reject a relapse to the unweighted arithmetic mean.
    unweighted_mean = sum(coefficients.values()) / len(coefficients)
    assert entry["fitted_r_eff"] != pytest.approx(unweighted_mean)
    assert entry["fitted_coefficients"] == coefficients
    assert entry["measured_ratio"] == {"mean": 1.8, "p50": 1.75}
    assert entry["n_samples"] == {"fitted": 19, "measured": 42}


async def test_calibration_telemetry_neutral_fit_is_scale_stable(monkeypatch: Any) -> None:
    """Below-threshold / degenerate fits fold to all-1.0 coefficients, whose
    blend is 1.0 for any composition — no divergence introduced by the fix."""
    mass = {c: 0.0 for c in CONTENT_CLASSES}
    mass["text"] = 500.0
    mass["tool_use"] = 500.0
    row: dict[str, Any] = {
        "model": "model-b",
        "mean_ratio": 1.05,
        "p50_ratio": 1.0,
        "n_samples": 3,
    }
    row.update({f"mass_{c}": m for c, m in mass.items()})
    conn = _Conn([row])

    async def _fit(_conn: Any, model: str, *, account_id: str) -> tuple[None, int]:
        return None, 3

    monkeypatch.setattr("aios.db.queries.events.model_token_class_ratio_fit", _fit)
    result = await calibration_telemetry(conn)

    assert result["model-b"]["fitted_r_eff"] == pytest.approx(1.0)
    assert result["model-b"]["fitted_coefficients"] == {c: 1.0 for c in CONTENT_CLASSES}


def test_calibration_telemetry_route_uses_pool(monkeypatch: Any) -> None:
    expected = {"model-a": {"fitted_r_eff": 1.0}}

    async def _telemetry(_conn: Any) -> dict[str, Any]:
        return expected

    monkeypatch.setattr("aios.api.routers.health.calibration_telemetry", _telemetry)

    class _Acquire:
        async def __aenter__(self) -> object:
            return object()

        async def __aexit__(self, *args: Any) -> None:
            return None

    class _Pool:
        def acquire(self) -> _Acquire:
            return _Acquire()

    app = FastAPI()
    app.state.pool = _Pool()
    app.include_router(health.router)
    response = TestClient(app).get("/telemetry/calibration")
    assert response.status_code == 200
    assert response.json() == {"models": expected, "window_hours": 24}
