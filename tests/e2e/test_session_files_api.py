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
from tests.helpers.connections import bearer


async def _make_session(harness: Harness) -> str:
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
    )
    env = await env_svc.create_environment(harness._pool, name=f"env-files-{suffix}")
    session = await sess_svc.create_session(
        harness._pool,
        agent_id=agent.id,
        environment_id=env.id,
        title=None,
        metadata={},
    )
    return session.id


async def _create_connection_with_session(
    harness: Harness, http_client: httpx.AsyncClient, session_id: str, account_suffix: str
) -> tuple[str, str]:
    """Returns ``(connection_id, token_plaintext)`` attached to ``session_id``."""
    from aios.services import connections as connections_service

    r = await http_client.post(
        "/v1/connections",
        json={"connector": "echo", "account": f"files-{account_suffix}"},
    )
    assert r.status_code == 201, r.text
    connection_id = str(r.json()["id"])
    await connections_service.attach_connection(harness._pool, connection_id, session_id=session_id)
    t = await http_client.post("/v1/connector-tokens", json={"connection_id": connection_id})
    assert t.status_code == 201, t.text
    return connection_id, str(t.json()["plaintext"])


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

    async def test_connector_token_can_upload_to_attached_session(
        self, http_client: httpx.AsyncClient, harness: Harness
    ) -> None:
        session_id = await _make_session(harness)
        _, token = await _create_connection_with_session(harness, http_client, session_id, "ok")

        r = await http_client.post(
            f"/v1/sessions/{session_id}/files",
            headers=bearer(token),
            files={"file": ("doc.txt", b"hello", "text/plain")},
        )
        assert r.status_code == 201, r.text
        assert r.json()["filename"] == "doc.txt"

    async def test_connector_token_rejected_for_other_session(
        self, http_client: httpx.AsyncClient, harness: Harness
    ) -> None:
        attached_session = await _make_session(harness)
        other_session = await _make_session(harness)
        _, token = await _create_connection_with_session(
            harness, http_client, attached_session, "mismatch"
        )

        r = await http_client.post(
            f"/v1/sessions/{other_session}/files",
            headers=bearer(token),
            files={"file": ("nope.bin", b"x", "application/octet-stream")},
        )
        assert r.status_code == 403, r.text
        body = r.json()
        # Error envelope is {"error": {"type", "message", "detail": {...}}}.
        assert body["error"]["type"] == "forbidden"
        assert body["error"]["detail"]["session_id"] == other_session

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
