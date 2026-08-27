"""Route-level coverage for the browser takeover API (jarbot#106 §5.7).

TestClient + dependency overrides (the ``test_sse_preflight`` pattern), with
``submit_browser_call`` and the query layer stubbed: the trust boundary
(scoped grant read → 404 before any stream/file access), the input epoch and
spool-cap gates, heartbeat rowcount 404/409, the error-currency → HTTP map,
and the frames-manifest containment + poll behavior.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aios.api.deps import get_account_id, get_db_url, get_pool
from aios.api.routers import browser as browser_router
from aios.config import get_settings
from aios.db import queries as queries_module
from aios.errors import install_exception_handlers
from aios.services import sessions as sessions_service

_ACCOUNT = "acc_01ROUTETEST0000000000000000"
_OTHER_GRANT = {
    "id": "bgr_1",
    "account_id": _ACCOUNT,
    "session_id": "sess_1",
    "status": "open",
    "reason": "auth",
    "boot": "01BOOT",
    "epoch": 5,
    "target": {"url": "https://x.example"},
}


def _build_app() -> FastAPI:
    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(browser_router.router)

    async def _pool() -> Any:
        return MagicMock()

    async def _db_url() -> str:
        return "postgresql://stub/aios"

    async def _account() -> str:
        return _ACCOUNT

    app.dependency_overrides[get_pool] = _pool
    app.dependency_overrides[get_db_url] = _db_url
    app.dependency_overrides[get_account_id] = _account
    return app


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(_build_app()) as c:
        yield c


def _stub_conn(monkeypatch: pytest.MonkeyPatch, **query_returns: Any) -> None:
    """Patch queries.* the routes call, and give pool.acquire a stub conn."""
    for name, value in query_returns.items():
        monkeypatch.setattr(queries_module, name, AsyncMock(return_value=value), raising=False)


class TestTrustBoundary:
    def test_input_on_unknown_grant_404s_before_any_write(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_conn(monkeypatch, get_browser_grant=None)
        resp = client.post(
            "/v1/browser/takeover/bgr_missing/input",
            json={"epoch": 5, "seq": 1, "events": [{"type": "pointer_move", "x": 1, "y": 2}]},
        )
        assert resp.status_code == 404
        assert resp.json()["error"]["type"] == "not_found"

    def test_frames_on_unknown_grant_404s_before_opening_the_stream(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_conn(monkeypatch, get_browser_grant=None)
        resp = client.get("/v1/browser/takeover/bgr_missing/frames")
        assert resp.status_code == 404


class TestInput:
    def test_stale_epoch_is_409(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_conn(monkeypatch, get_browser_grant=_OTHER_GRANT)
        resp = client.post(
            "/v1/browser/takeover/bgr_1/input",
            json={"epoch": 4, "seq": 1, "events": [{"type": "pointer_move", "x": 1, "y": 2}]},
        )
        assert resp.status_code == 409
        assert resp.json()["error"]["detail"]["code"] == "stale_epoch"

    def test_closed_grant_is_409(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_conn(monkeypatch, get_browser_grant={**_OTHER_GRANT, "status": "closed"})
        resp = client.post(
            "/v1/browser/takeover/bgr_1/input",
            json={"epoch": 5, "seq": 1, "events": [{"type": "pointer_move", "x": 1, "y": 2}]},
        )
        assert resp.status_code == 409

    def test_accepted_input_appends_one_jsonl_line(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _stub_conn(monkeypatch, get_browser_grant=_OTHER_GRANT)
        monkeypatch.setattr(get_settings(), "workspace_root", tmp_path)
        from aios.sandbox.volumes import ensure_browser_plane_dir

        plane = ensure_browser_plane_dir(_ACCOUNT)

        resp = client.post(
            "/v1/browser/takeover/bgr_1/input",
            json={"epoch": 5, "seq": 7, "events": [{"type": "text", "text": "hi"}]},
        )
        assert resp.status_code == 204
        spool = plane / "input" / "spool.jsonl"
        lines = spool.read_text().strip().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["grant_id"] == "bgr_1" and rec["epoch"] == 5 and rec["seq"] == 7
        assert rec["events"] == [{"type": "text", "text": "hi"}]

    def test_over_cap_spool_is_413(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _stub_conn(monkeypatch, get_browser_grant=_OTHER_GRANT)
        settings = get_settings()
        monkeypatch.setattr(settings, "workspace_root", tmp_path)
        monkeypatch.setattr(settings, "sandbox_browser_input_spool_max_bytes", 10)
        from aios.sandbox.volumes import ensure_browser_plane_dir

        ensure_browser_plane_dir(_ACCOUNT)
        resp = client.post(
            "/v1/browser/takeover/bgr_1/input",
            json={"epoch": 5, "seq": 1, "events": [{"type": "text", "text": "x" * 100}]},
        )
        assert resp.status_code == 413


class TestHeartbeat:
    def test_unknown_grant_404(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_conn(monkeypatch, touch_browser_grant_heartbeat=False, get_browser_grant=None)
        resp = client.post("/v1/browser/takeover/bgr_x/heartbeat")
        assert resp.status_code == 404

    def test_closed_grant_409(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_conn(
            monkeypatch,
            touch_browser_grant_heartbeat=False,
            get_browser_grant={**_OTHER_GRANT, "status": "expired"},
        )
        resp = client.post("/v1/browser/takeover/bgr_1/heartbeat")
        assert resp.status_code == 409

    def test_open_grant_204(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_conn(monkeypatch, touch_browser_grant_heartbeat=True)
        resp = client.post("/v1/browser/takeover/bgr_1/heartbeat")
        assert resp.status_code == 204


class TestControlErrorCurrency:
    def test_takeover_in_progress_is_409(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            browser_router,
            "submit_browser_call",
            AsyncMock(return_value=({"code": "takeover_in_progress", "message": "busy"}, True)),
        )
        monkeypatch.setattr(
            sessions_service,
            "get_session_basic",
            AsyncMock(return_value=MagicMock()),
        )
        resp = client.post("/v1/browser/takeover", json={"session_id": "sess_1", "reason": "auth"})
        assert resp.status_code == 409
        assert resp.json()["error"]["detail"]["code"] == "takeover_in_progress"

    def test_browser_unavailable_is_503(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            browser_router,
            "submit_browser_call",
            AsyncMock(return_value=({"code": "browser_unavailable", "message": "down"}, True)),
        )
        monkeypatch.setattr(
            sessions_service,
            "get_session_basic",
            AsyncMock(return_value=MagicMock()),
        )
        resp = client.post("/v1/browser/takeover", json={"session_id": "sess_1"})
        assert resp.status_code == 503

    def test_open_success_returns_the_grant(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            browser_router,
            "submit_browser_call",
            AsyncMock(
                return_value=(
                    {
                        "grant_id": "bgr_new",
                        "target": {"url": "https://x"},
                        "boot": "01B",
                        "epoch": 9,
                        "ttl_seconds": 300,
                    },
                    False,
                )
            ),
        )
        monkeypatch.setattr(
            sessions_service,
            "get_session_basic",
            AsyncMock(return_value=MagicMock()),
        )
        resp = client.post("/v1/browser/takeover", json={"session_id": "sess_1"})
        assert resp.status_code == 200
        assert resp.json()["grant_id"] == "bgr_new" and resp.json()["epoch"] == 9


class TestFrameLoading:
    def test_hostile_manifest_file_is_refused(self, tmp_path: Path) -> None:
        frames = tmp_path / "frames"
        frames.mkdir()
        (tmp_path / "secret.jpg").write_bytes(b"nope")
        manifest = {"seq": 1, "file": "../secret.jpg"}
        assert browser_router._load_frame(frames, manifest, _ACCOUNT) is None

    def test_valid_manifest_forwards_the_trusted_chrome_envelope(self, tmp_path: Path) -> None:
        frames = tmp_path / "frames"
        frames.mkdir()
        (frames / "0.jpg").write_bytes(b"\xff\xd8jpeg")
        manifest = {
            "seq": 3,
            "file": "0.jpg",
            "ts_ms": 100,
            "epoch": 7,
            "boot": "01B",
            "origin": "https://accounts.example.com",
            "security": "secure",
            "w": 1280,
            "h": 800,
        }
        frame = browser_router._load_frame(frames, manifest, _ACCOUNT)
        assert frame is not None
        assert frame["origin"] == "https://accounts.example.com"
        assert frame["security"] == "secure"
        assert frame["epoch"] == 7 and frame["boot"] == "01B"
        assert "account" not in frame  # never crosses to the client
        import base64

        assert frame["jpeg_b64"] == base64.b64encode(b"\xff\xd8jpeg").decode()

    def test_absent_manifest_reads_as_none(self, tmp_path: Path) -> None:
        assert browser_router._read_manifest(tmp_path / "frames") is None
