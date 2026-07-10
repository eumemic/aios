from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from aios.api.routers import health
from aios.db.queries.events import calibration_telemetry


class _Conn:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.query = ""

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.query = query
        return self.rows


async def test_calibration_telemetry_aggregates_recent_successes(monkeypatch: Any) -> None:
    conn = _Conn(
        [
            {"model": "model-a", "mean_ratio": 1.2, "p50_ratio": 1.1, "n_samples": 7},
        ]
    )

    async def _fit(_conn: Any, model: str, *, account_id: str) -> tuple[dict[str, float], int]:
        assert model == "model-a"
        return {"message": 0.9, "tool_result": 1.1}, 19

    monkeypatch.setattr("aios.db.queries.events.model_token_class_ratio_fit", _fit)
    result = await calibration_telemetry(conn)

    assert "created_at >= now() - interval '24 hours'" in conn.query
    assert "input_tokens')::bigint > 0" in conn.query
    assert result == {
        "model-a": {
            "fitted_r_eff": 1.0,
            "fitted_coefficients": {"message": 0.9, "tool_result": 1.1},
            "measured_ratio": {"mean": 1.2, "p50": 1.1},
            "n_samples": {"fitted": 19, "measured": 7},
        }
    }


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
