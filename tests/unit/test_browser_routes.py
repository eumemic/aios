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

    def test_frames_on_closed_grant_is_409_not_a_stale_stream(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A screencast must not open on an already-terminal grant (it would
        emit one stale frame before the recheck ends it) — reject at open."""
        _stub_conn(monkeypatch, get_browser_grant={**_OTHER_GRANT, "status": "closed"})
        resp = client.get("/v1/browser/takeover/bgr_1/frames")
        assert resp.status_code == 409


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

    def test_symlinked_spool_is_not_followed(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A compromised container plants a symlink at input/spool.jsonl aimed
        at another account's plane; O_NOFOLLOW makes the append fail rather than
        write into the victim's file."""
        _stub_conn(monkeypatch, get_browser_grant=_OTHER_GRANT)
        monkeypatch.setattr(get_settings(), "workspace_root", tmp_path)
        from aios.sandbox.volumes import ensure_browser_plane_dir

        plane = ensure_browser_plane_dir(_ACCOUNT)
        victim = tmp_path / "acc_VICTIM" / "input" / "spool.jsonl"
        victim.parent.mkdir(parents=True)
        victim.write_text("")
        (plane / "input" / "spool.jsonl").symlink_to(victim)

        # O_NOFOLLOW fails ELOOP rather than following into the victim; the
        # route lets it propagate (a 500 in prod, re-raised by the TestClient).
        # The load-bearing assertion is that the victim's file is never written.
        with pytest.raises(OSError):
            client.post(
                "/v1/browser/takeover/bgr_1/input",
                json={"epoch": 5, "seq": 1, "events": [{"type": "text", "text": "attack"}]},
            )
        assert victim.read_text() == ""  # the victim's spool was never written


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

    def test_open_grant_touches_the_liveness_marker(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _stub_conn(monkeypatch, touch_browser_grant_heartbeat=True)
        monkeypatch.setattr(get_settings(), "workspace_root", tmp_path)
        from aios.sandbox.volumes import ensure_browser_plane_dir

        plane = ensure_browser_plane_dir(_ACCOUNT)
        marker = plane / "input" / ".heartbeat"
        assert not marker.exists()

        resp = client.post("/v1/browser/takeover/bgr_1/heartbeat")

        assert resp.status_code == 204
        # A watching (heartbeating) but not-typing viewer's liveness reaches the
        # driver's idle watchdog via this marker, not just the DB heartbeat_at.
        assert marker.exists()

    def test_marker_failure_does_not_fail_the_heartbeat(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Plane input dir absent → marker.touch() raises, but the DB heartbeat
        # already succeeded, so the viewer still gets 204 (best-effort marker).
        _stub_conn(monkeypatch, touch_browser_grant_heartbeat=True)
        monkeypatch.setattr(get_settings(), "workspace_root", tmp_path)  # no plane created
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

    def test_browser_crashed_is_503(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A dead Chromium host is transient exactly like an unreachable
        container — retryable 503, never a 409 that reads as a state
        conflict (a permanently-crashing browser must not 409-loop)."""
        monkeypatch.setattr(
            browser_router,
            "submit_browser_call",
            AsyncMock(return_value=({"code": "browser_crashed", "message": "chromium died"}, True)),
        )
        monkeypatch.setattr(
            sessions_service, "get_session_basic", AsyncMock(return_value=MagicMock())
        )
        resp = client.post("/v1/browser/takeover", json={"session_id": "sess_1"})
        assert resp.status_code == 503
        assert resp.json()["error"]["detail"]["code"] == "browser_crashed"

    def test_internal_error_is_500_not_409(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The executor's generic ``internal`` backstop is a worker-side bug —
        it must surface as 500, not a 409 that reads as a caller-actionable
        conflict."""
        monkeypatch.setattr(
            browser_router,
            "submit_browser_call",
            AsyncMock(return_value=({"code": "internal", "message": "boom"}, True)),
        )
        monkeypatch.setattr(
            sessions_service, "get_session_basic", AsyncMock(return_value=MagicMock())
        )
        resp = client.post("/v1/browser/takeover", json={"session_id": "sess_1"})
        assert resp.status_code == 500

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


class TestHandbackShot:
    """The close handback inlines the driver's screenshot from the plane — the
    third no-follow read site. It must read a real shot and refuse a symlinked
    or escaping shot_path (the bytes land in the product handback)."""

    def _plane(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        monkeypatch.setattr(get_settings(), "workspace_root", tmp_path)
        from aios.sandbox.volumes import ensure_browser_plane_dir

        return ensure_browser_plane_dir(_ACCOUNT)

    def test_real_shot_becomes_a_data_url(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        plane = self._plane(tmp_path, monkeypatch)
        (plane / "shots" / "handback.png").write_bytes(b"\x89PNGshot")
        payload = browser_router._handback_payload({"shot_path": "shots/handback.png"}, _ACCOUNT)
        import base64

        assert payload.screenshot_data_url == (
            "data:image/png;base64," + base64.b64encode(b"\x89PNGshot").decode()
        )

    def test_symlinked_shot_yields_no_data_url(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        plane = self._plane(tmp_path, monkeypatch)
        victim = tmp_path / "acc_VICTIM" / "profile" / "Cookies"
        victim.parent.mkdir(parents=True)
        victim.write_bytes(b"cookie-jar")
        (plane / "shots" / "handback.png").symlink_to(victim)
        payload = browser_router._handback_payload({"shot_path": "shots/handback.png"}, _ACCOUNT)
        assert payload.screenshot_data_url is None

    def test_escaping_shot_path_yields_no_data_url(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._plane(tmp_path, monkeypatch)
        (tmp_path / "secret.png").write_bytes(b"secret")
        payload = browser_router._handback_payload({"shot_path": "../../secret.png"}, _ACCOUNT)
        assert payload.screenshot_data_url is None


class TestFrameLoading:
    def test_hostile_manifest_file_is_refused(self, tmp_path: Path) -> None:
        frames = tmp_path / "frames"
        frames.mkdir()
        (tmp_path / "secret.jpg").write_bytes(b"nope")
        manifest = {"seq": 1, "file": "../secret.jpg"}
        assert browser_router._load_frame(tmp_path, manifest) is None

    def test_symlinked_frames_dir_to_another_plane_is_refused(self, tmp_path: Path) -> None:
        """CRITICAL: a compromised container swaps its OWN frames dir for a
        symlink to ANOTHER account's plane. The no-follow walk refuses the
        symlinked component at open time — closing both the plain escape and
        the TOCTOU variant (a component swapped AFTER any check), which the old
        resolve-then-read sequence left open."""
        victim_frames = tmp_path / "acc_VICTIM" / "frames"
        victim_frames.mkdir(parents=True)
        (victim_frames / "0.jpg").write_bytes(b"\xff\xd8victim-screen")
        (victim_frames / "manifest.json").write_text(json.dumps({"seq": 1, "file": "0.jpg"}))
        attacker_plane = tmp_path / "acc_ATTACKER"
        attacker_plane.mkdir()
        (attacker_plane / "frames").symlink_to(victim_frames)  # escape the plane

        # Neither the manifest read nor the frame load may cross to the victim.
        assert browser_router._read_manifest(attacker_plane) is None
        assert browser_router._load_frame(attacker_plane, {"seq": 1, "file": "0.jpg"}) is None

    def test_symlinked_frame_file_is_refused(self, tmp_path: Path) -> None:
        """The leaf variant of the swap: a real frames dir whose FRAME FILE is
        a symlink into another account's plane (the exact post-check swap the
        TOCTOU exploited). O_NOFOLLOW on the leaf open refuses it."""
        victim = tmp_path / "acc_VICTIM" / "profile" / "Cookies"
        victim.parent.mkdir(parents=True)
        victim.write_bytes(b"cookie-jar")
        plane = tmp_path / "acc_ATTACKER"
        (plane / "frames").mkdir(parents=True)
        (plane / "frames" / "0.jpg").symlink_to(victim)
        assert browser_router._load_frame(plane, {"seq": 1, "file": "0.jpg"}) is None

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
        frame = browser_router._load_frame(tmp_path, manifest)
        assert frame is not None
        assert frame["origin"] == "https://accounts.example.com"
        assert frame["security"] == "secure"
        assert frame["epoch"] == 7 and frame["boot"] == "01B"
        assert "account" not in frame  # never crosses to the client
        import base64

        assert frame["jpeg_b64"] == base64.b64encode(b"\xff\xd8jpeg").decode()

    def test_absent_manifest_reads_as_none(self, tmp_path: Path) -> None:
        assert browser_router._read_manifest(tmp_path) is None

    def test_non_int_seq_reads_as_no_frame(self, tmp_path: Path) -> None:
        """A manifest present but with a null/placeholder seq is 'no frame yet',
        not a TypeError that tears down the SSE stream."""
        frames = tmp_path / "frames"
        frames.mkdir()
        (frames / "manifest.json").write_text(json.dumps({"seq": None, "file": "0.jpg"}))
        assert browser_router._read_manifest(tmp_path) is None


class TestFramesEpochFence:
    """The stream must never forward a frame stamped with a DIFFERENT epoch
    than the grant it serves: after this grant closes and a new takeover opens
    (epoch rotates on both edges), a still-attached viewer would otherwise
    receive the NEXT takeover's frames for up to the grant-recheck window (the
    same-account frame bleed)."""

    def _plane_with_manifest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, epoch: int
    ) -> None:
        monkeypatch.setattr(get_settings(), "workspace_root", tmp_path)
        from aios.sandbox.volumes import ensure_browser_plane_dir

        frames = ensure_browser_plane_dir(_ACCOUNT) / "frames"
        (frames / "frame-1.jpg").write_bytes(b"\xff\xd8jpeg")
        (frames / "manifest.json").write_text(
            json.dumps({"seq": 1, "file": "frame-1.jpg", "boot": "01BOOT", "epoch": epoch})
        )

    def _stream_body(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> str:
        # Call 1 = the route's scope gate (open, epoch 5); call 2 = the first
        # in-loop recheck (closed) — so the poll loop runs exactly one iteration
        # and the stream ends deterministically.
        monkeypatch.setattr(
            queries_module,
            "get_browser_grant",
            AsyncMock(side_effect=[dict(_OTHER_GRANT), {**_OTHER_GRANT, "status": "closed"}]),
            raising=False,
        )
        resp = client.get("/v1/browser/takeover/bgr_1/frames")
        assert resp.status_code == 200
        return resp.text

    def test_mismatched_epoch_frame_is_never_streamed(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._plane_with_manifest(tmp_path, monkeypatch, epoch=6)  # grant epoch is 5
        body = self._stream_body(client, monkeypatch)
        assert "event: frame" not in body
        assert "event: end" in body

    def test_matching_epoch_frame_is_streamed(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Positive control for the fence: the identical setup with the grant's
        OWN epoch DOES stream the frame."""
        self._plane_with_manifest(tmp_path, monkeypatch, epoch=5)
        body = self._stream_body(client, monkeypatch)
        assert "event: frame" in body
        assert "event: end" in body
