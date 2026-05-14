"""E2E coverage for ``POST /v1/sessions/<id>/files`` (#324).

Exercises the multipart upload endpoint against the in-process API
(no Docker required for the api-side contract).  The model-side
visibility at ``/mnt/uploads/<file_id>/<filename>`` is exercised
manually per the plan's Verification section — this test pins the
HTTP-level shape, auth scoping, size enforcement, and durable
landing of bytes on the api filesystem.
"""

from __future__ import annotations

import hashlib
import secrets

import httpx
import pytest

from aios.config import get_settings
from aios.sandbox.volumes import session_uploads_dir
from tests.e2e.harness import Harness


async def _make_session(harness: Harness) -> str:
    account_id = "acc_test_stub"  # PR 3 scaffolding
    from aios.services import agents as agents_service
    from aios.services import environments as env_svc
    from aios.services import sessions as sess_svc

    # Fresh suffix per call — multiple sessions in one test can't collide on
    # agent/env names.
    suffix = secrets.token_hex(4)
    agent = await agents_service.create_agent(
        harness._pool,
        name=f"files-{suffix}",
        model="fake/test",
        system="",
        tools=[],
        description=None,
        metadata={},
        window_min=50_000,
        window_max=150_000,
        account_id=account_id,
    )
    env = await env_svc.create_environment(
        harness._pool, name=f"env-files-{suffix}", account_id=account_id
    )
    session = await sess_svc.create_session(
        harness._pool,
        agent_id=agent.id,
        environment_id=env.id,
        title=None,
        metadata={},
        account_id=account_id,
    )
    return session.id


class TestSessionFilesUpload:
    async def test_operator_upload_happy_path(
        self, http_client: httpx.AsyncClient, harness: Harness
    ) -> None:
        session_id = await _make_session(harness)
        payload = b"\x89PNG\r\n\x1a\nfake-image-bytes"

        r = await http_client.post(
            f"/v1/sessions/{session_id}/files",
            files={"file": ("photo.png", payload, "image/png")},
        )
        assert r.status_code == 201, r.text
        body = r.json()

        assert body["filename"] == "photo.png"
        assert body["content_type"] == "image/png"
        assert body["size"] == len(payload)
        assert body["sha256"] == hashlib.sha256(payload).hexdigest()
        assert body["in_sandbox_path"] == f"/mnt/uploads/{body['file_id']}/photo.png"
        assert body["file_id"].startswith("file_")

        # Bytes durable on the api filesystem at the expected per-file dir.
        host_path = session_uploads_dir(session_id) / body["file_id"] / "photo.png"
        assert host_path.exists()
        assert host_path.read_bytes() == payload

    async def test_unknown_session_returns_404(
        self, http_client: httpx.AsyncClient, harness: Harness
    ) -> None:
        r = await http_client.post(
            "/v1/sessions/sess_does_not_exist/files",
            files={"file": ("x.bin", b"x", "application/octet-stream")},
        )
        assert r.status_code == 404, r.text

    async def test_oversized_returns_413(
        self,
        http_client: httpx.AsyncClient,
        harness: Harness,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session_id = await _make_session(harness)
        settings = get_settings()
        monkeypatch.setattr(settings, "upload_max_size_bytes", 64)

        r = await http_client.post(
            f"/v1/sessions/{session_id}/files",
            files={"file": ("big.bin", b"a" * 256, "application/octet-stream")},
        )
        assert r.status_code == 413, r.text
        # Per-file dir was cleaned up.
        assert list(session_uploads_dir(session_id).iterdir()) == []

    async def test_missing_bearer_returns_401(
        self, http_client: httpx.AsyncClient, harness: Harness
    ) -> None:
        session_id = await _make_session(harness)
        # Force the request through with no Authorization header — strip
        # the one the fixture installs by passing an empty headers dict
        # that overrides per-call.
        r = await http_client.post(
            f"/v1/sessions/{session_id}/files",
            headers={"Authorization": ""},
            files={"file": ("x.bin", b"x", "application/octet-stream")},
        )
        assert r.status_code == 401, r.text
