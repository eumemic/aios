"""Unit coverage for the browser plane's local upkeep (jarbot#106).

The DB-backed executor/reaper paths live in
``tests/integration/test_browser_control_plane.py``; this file pins the pure
pieces: the plane byte-quota sweep (oldest-first, profile untouchable, spool
truncation keyed on open grants) and the lifecycle renderer's totality.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pytest

from aios.config import get_settings
from aios.harness import browser_control
from aios.harness.browser_control import _driver, _enforce_plane_quotas
from aios.harness.context import _render_browser_lifecycle_notice
from aios.sandbox.browser_protocol import BrowserRequest, BrowserResponse
from aios.sandbox.registry import SandboxRegistry
from aios.sandbox.volumes import ensure_browser_plane_dir

_ACCOUNT_ID = "acc_01QUOTATEST00000000000000000"


class TestPlaneQuotas:
    @pytest.fixture
    def plane(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        settings = get_settings()
        monkeypatch.setattr(settings, "workspace_root", tmp_path)
        monkeypatch.setattr(settings, "sandbox_browser_shots_max_bytes", 1024**2)
        monkeypatch.setattr(settings, "sandbox_browser_frames_max_bytes", 100)
        monkeypatch.setattr(settings, "sandbox_browser_downloads_max_bytes", 1024**2)
        return ensure_browser_plane_dir(_ACCOUNT_ID)

    @staticmethod
    def _write(path: Path, size: int, *, age_s: float) -> None:
        path.write_bytes(b"x" * size)
        stamp = time.time() - age_s
        os.utime(path, (stamp, stamp))

    def test_over_cap_dir_reaps_oldest_first(self, plane: Path) -> None:
        frames = plane / "frames"
        self._write(frames / "old.jpg", 60, age_s=300)
        self._write(frames / "mid.jpg", 60, age_s=200)
        self._write(frames / "new.jpg", 60, age_s=100)

        _enforce_plane_quotas(accounts_with_open_grants={_ACCOUNT_ID})

        # 180 bytes against a 100-byte cap: the two oldest go, newest stays.
        assert not (frames / "old.jpg").exists()
        assert not (frames / "mid.jpg").exists()
        assert (frames / "new.jpg").exists()

    def test_profile_is_never_touched(self, plane: Path) -> None:
        """Real logins live in the profile — no byte cap ever reaps it."""
        cookies = plane / "profile" / "Cookies"
        self._write(cookies, 50_000_000, age_s=10_000)

        _enforce_plane_quotas(accounts_with_open_grants=set())

        assert cookies.exists()

    def test_spool_truncated_only_without_an_open_grant(self, plane: Path) -> None:
        spool = plane / "input" / "spool.jsonl"
        spool.write_text('{"epoch": 1}\n')

        _enforce_plane_quotas(accounts_with_open_grants={_ACCOUNT_ID})
        assert spool.exists()  # a human is driving — never yank their input

        _enforce_plane_quotas(accounts_with_open_grants=set())
        assert not spool.exists()


class TestBrowserLifecycleRenderer:
    """Total by contract: a pure function of ``data`` that never raises."""

    def test_outcome_variants_render(self) -> None:
        for outcome in ("done", "cancelled", "expired", None, "??"):
            text = _render_browser_lifecycle_notice(
                {"event": "browser_takeover_ended", "outcome": outcome, "url": "https://x.example"}
            )
            assert text.startswith("[") and text.endswith("]")
            assert "handed control back" in text

    def test_state_lost_renders(self) -> None:
        text = _render_browser_lifecycle_notice({"event": "browser_state_lost", "cause": "cleared"})
        assert "cleared" in text

    def test_empty_data_never_raises(self) -> None:
        assert isinstance(_render_browser_lifecycle_notice({}), str)


class TestTakeoverTargeting:
    """``takeover_open`` must carry the requesting session's id in the request's
    ``session_id`` FIELD (not args) so the driver takes over that session's
    page (jarbot#106 §5.6). The field already exists on ``BrowserRequest``."""

    async def test_driver_threads_session_id_into_the_request_field(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, BrowserRequest] = {}

        async def _fake_driver_call(
            registry: object, account_id: str, request: BrowserRequest, *, timeout_s: float
        ) -> BrowserResponse:
            seen["request"] = request
            return BrowserResponse(ok=True, boot="01BOOT", epoch=1)

        monkeypatch.setattr(browser_control, "driver_call", _fake_driver_call)

        await _driver(
            cast(SandboxRegistry, MagicMock()),
            "acc_x",
            "takeover_open",
            {"grant_id": "bgr_x", "reason": ""},
            timeout_s=45,
            session_id="sess_target",
        )

        request = seen["request"]
        assert request.session_id == "sess_target"  # threaded via the field…
        assert "session_id" not in request.args  # …never the args dict

    async def test_control_ops_without_a_session_stay_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, BrowserRequest] = {}

        async def _fake_driver_call(
            registry: object, account_id: str, request: BrowserRequest, *, timeout_s: float
        ) -> BrowserResponse:
            seen["request"] = request
            return BrowserResponse(ok=True, boot="01BOOT", epoch=0)

        monkeypatch.setattr(browser_control, "driver_call", _fake_driver_call)

        await _driver(cast(SandboxRegistry, MagicMock()), "acc_x", "status", {}, timeout_s=30)

        assert seen["request"].session_id is None
