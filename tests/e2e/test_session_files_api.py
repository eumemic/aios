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

pytestmark = pytest.mark.docker


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


class TestSessionFilesDownload:
    """``GET /v1/sessions/<id>/files/<file_id>`` (#179): a thin authenticated
    read of bytes that already exist on disk, not a CDN — no transformation,
    same scoping contract as every other session-scoped read."""

    async def test_download_roundtrips_uploaded_bytes(
        self, http_client: httpx.AsyncClient, harness: Harness
    ) -> None:
        session_id = await _make_session(harness)
        payload = b"\x89PNG\r\n\x1a\nfake-image-bytes"
        upload = await http_client.post(
            f"/v1/sessions/{session_id}/files",
            files={"file": ("photo.png", payload, "image/png")},
        )
        assert upload.status_code == 201, upload.text
        file_id = upload.json()["file_id"]

        r = await http_client.get(f"/v1/sessions/{session_id}/files/{file_id}")

        assert r.status_code == 200, r.text
        assert r.content == payload
        assert r.headers["content-type"] == "image/png"
        assert "inline" in r.headers["content-disposition"]

    async def test_download_pins_nosniff(
        self, http_client: httpx.AsyncClient, harness: Harness
    ) -> None:
        """``nosniff`` on every response, inline or not.

        Defence-in-depth only — it stops the browser *guessing* a type, and
        does nothing about a declared-and-honoured one.  The allowlist below
        is the actual control.
        """
        session_id = await _make_session(harness)
        upload = await http_client.post(
            f"/v1/sessions/{session_id}/files",
            files={"file": ("photo.png", b"\x89PNG\r\n\x1a\n", "image/png")},
        )
        file_id = upload.json()["file_id"]

        r = await http_client.get(f"/v1/sessions/{session_id}/files/{file_id}")

        assert r.headers["x-content-type-options"] == "nosniff"

    @pytest.mark.parametrize(
        "declared_type",
        [
            "image/svg+xml",
            "text/html",
            "application/xhtml+xml",
            "image/svg+xml; charset=utf-8",
            "IMAGE/SVG+XML",
        ],
    )
    async def test_script_bearing_declared_type_never_renders_in_origin(
        self, http_client: httpx.AsyncClient, harness: Harness, declared_type: str
    ) -> None:
        """The stored content-type is attacker-chosen; never echo it inline.

        ``stage_upload`` takes ``upload.content_type`` verbatim from the
        client's multipart header, so an uploader picks the type their own
        bytes come back as.  ``image/svg+xml`` passes any ``image/*`` prefix
        test and executes script in the serving origin — stored XSS.  Served
        as an octet-stream attachment instead, so the bytes still download
        but never render.  Parameterized over casing/parameter variants
        because the stored value is unnormalized.
        """
        session_id = await _make_session(harness)
        payload = b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>'
        upload = await http_client.post(
            f"/v1/sessions/{session_id}/files",
            files={"file": ("payload.svg", payload, declared_type)},
        )
        assert upload.status_code == 201, upload.text
        file_id = upload.json()["file_id"]

        r = await http_client.get(f"/v1/sessions/{session_id}/files/{file_id}")

        assert r.status_code == 200, r.text
        # Bytes are preserved — this is a rendering control, not a filter.
        assert r.content == payload
        assert r.headers["content-type"] == "application/octet-stream"
        assert "svg" not in r.headers["content-type"]
        assert "attachment" in r.headers["content-disposition"]
        assert r.headers["x-content-type-options"] == "nosniff"

    @pytest.mark.parametrize(
        "declared_type",
        ["image/png", "image/jpeg", "image/gif", "image/webp"],
    )
    async def test_inline_allowlist_still_renders(
        self, http_client: httpx.AsyncClient, harness: Harness, declared_type: str
    ) -> None:
        """The raster types #179 needs keep rendering inline.

        Guards against the fix over-reaching into the feature it protects:
        if this set stops being served inline, composer thumbnails break.
        """
        session_id = await _make_session(harness)
        upload = await http_client.post(
            f"/v1/sessions/{session_id}/files",
            files={"file": ("photo.img", b"fake-raster-bytes", declared_type)},
        )
        file_id = upload.json()["file_id"]

        r = await http_client.get(f"/v1/sessions/{session_id}/files/{file_id}")

        assert r.status_code == 200, r.text
        assert r.headers["content-type"] == declared_type
        assert "inline" in r.headers["content-disposition"]

    async def test_unknown_file_id_returns_404(
        self, http_client: httpx.AsyncClient, harness: Harness
    ) -> None:
        session_id = await _make_session(harness)
        r = await http_client.get(f"/v1/sessions/{session_id}/files/file_does_not_exist")
        assert r.status_code == 404, r.text

    async def test_file_from_another_session_returns_404(
        self, http_client: httpx.AsyncClient, harness: Harness
    ) -> None:
        session_a = await _make_session(harness)
        session_b = await _make_session(harness)
        upload = await http_client.post(
            f"/v1/sessions/{session_a}/files",
            files={"file": ("x.bin", b"x", "application/octet-stream")},
        )
        file_id = upload.json()["file_id"]

        r = await http_client.get(f"/v1/sessions/{session_b}/files/{file_id}")

        assert r.status_code == 404, r.text

    async def test_missing_bearer_returns_401(
        self, http_client: httpx.AsyncClient, harness: Harness
    ) -> None:
        session_id = await _make_session(harness)
        upload = await http_client.post(
            f"/v1/sessions/{session_id}/files",
            files={"file": ("x.bin", b"x", "application/octet-stream")},
        )
        file_id = upload.json()["file_id"]

        r = await http_client.get(
            f"/v1/sessions/{session_id}/files/{file_id}",
            headers={"Authorization": ""},
        )
        assert r.status_code == 401, r.text
