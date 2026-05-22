"""Unit coverage for vision-aware rendering in :func:`render_user_event`.

The renderer's vision policy is delegated to
:mod:`aios.harness.vision`; here we exercise the wiring around it —
host-bytes-read for inlinable images, text-marker fallback for
non-vision models and oversize images, and the legacy-stub path.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import pytest

from aios.config import get_settings
from aios.harness import vision
from aios.harness.context import render_user_event
from aios.sandbox.volumes import session_attachments_dir


@pytest.fixture
def temp_workspace_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    settings = get_settings()
    monkeypatch.setattr(settings, "workspace_root", tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def _stub_supports_vision(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> Any:
    """Default: vision-capable model returns True; non-vision-capable returns False.

    Tests can override individual mappings via ``vision._VISION_OVERRIDES``.
    """
    saved = dict(vision._VISION_OVERRIDES)
    vision._VISION_OVERRIDES.clear()
    vision._VISION_OVERRIDES["model/vision"] = True
    vision._VISION_OVERRIDES["model/text"] = False
    yield
    vision._VISION_OVERRIDES.clear()
    vision._VISION_OVERRIDES.update(saved)


def _stage_image(
    workspace_root: Path, session_id: str, connector: str, name: str, payload: bytes
) -> str:
    """Write a fake staged attachment, return its in-sandbox path."""
    session_dir = session_attachments_dir(session_id) / connector
    session_dir.mkdir(parents=True, exist_ok=True)
    file_path = session_dir / name
    file_path.write_bytes(payload)
    return f"/mnt/attachments/{connector}/{name}"


def _user_event(
    *,
    content: str = "hi",
    channel: str = "echo/acct/chat-1",
    attachments: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"channel": channel}
    if attachments is not None:
        metadata["attachments"] = attachments
    return {"role": "user", "content": content, "metadata": metadata}


class TestVisionAwareRendering:
    def test_inlinable_image_emits_image_url_part(self, temp_workspace_root: Path) -> None:
        sandbox_path = _stage_image(
            temp_workspace_root, "sess-1", "echo", "evt-1-photo.jpg", b"jpegbytes"
        )
        event = _user_event(
            content="hello",
            attachments=[
                {
                    "filename": "photo.jpg",
                    "content_type": "image/jpeg",
                    "size": len(b"jpegbytes"),
                    "in_sandbox_path": sandbox_path,
                }
            ],
        )
        msg = render_user_event(
            event,
            "echo/acct/chat-1",
            "echo/acct/chat-1",
            model="model/vision",
            session_id="sess-1",
        )
        content = msg["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "text"
        assert "hello" in content[0]["text"]
        assert content[1] == {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64.b64encode(b'jpegbytes').decode()}"
            },
        }

    def test_non_dict_attachment_record_is_skipped(self, temp_workspace_root: Path) -> None:
        """A non-dict element in ``metadata.attachments`` must not crash the renderer.

        ``SessionUserMessage.metadata: dict[str, Any]`` accepts any shape;
        Pydantic doesn't drill into the value. A connector that mis-serializes
        attachments — ``["filename.jpg"]`` instead of ``[{...}]`` — or a
        pre-existing event row with corrupt metadata, used to crash
        ``_apply_attachments`` at ``record.get(...)`` with
        ``AttributeError: 'str' object has no attribute 'get'``. Because
        the renderer is called on every wake, the session became
        permanently un-renderable until manual intervention.
        """
        # Deliberately heterogeneous list: stray strings, None, and ints
        # are exactly the shapes a mis-serializing connector could send.
        malformed: list[Any] = ["malformed-string", None, 42]
        event = _user_event(content="hi", attachments=malformed)
        # Today: raises AttributeError mid-loop, bricking the session.
        msg = render_user_event(
            event,
            "echo/acct/chat-1",
            "echo/acct/chat-1",
            model="model/vision",
            session_id="sess-1",
        )
        # No crash; malformed records produce no image parts and no marker.
        assert msg["role"] == "user"
        assert isinstance(msg["content"], str)
        assert "hi" in msg["content"]

    def test_mixed_valid_and_malformed_attachments(self, temp_workspace_root: Path) -> None:
        """A valid record next to malformed ones still renders correctly —
        the malformed entries are skipped, the valid one inlines as usual."""
        sandbox_path = _stage_image(
            temp_workspace_root, "sess-1", "echo", "evt-1-photo.jpg", b"jpegbytes"
        )
        mixed: list[Any] = [
            "stray-string",
            {
                "filename": "photo.jpg",
                "content_type": "image/jpeg",
                "size": len(b"jpegbytes"),
                "in_sandbox_path": sandbox_path,
            },
            None,
        ]
        event = _user_event(content="hello", attachments=mixed)
        msg = render_user_event(
            event,
            "echo/acct/chat-1",
            "echo/acct/chat-1",
            model="model/vision",
            session_id="sess-1",
        )
        content = msg["content"]
        assert isinstance(content, list)
        # The valid image was inlined; the malformed records produced nothing.
        image_parts = [p for p in content if p.get("type") == "image_url"]
        assert len(image_parts) == 1
        assert image_parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_oversize_image_falls_back_to_marker(self, temp_workspace_root: Path) -> None:
        sandbox_path = _stage_image(
            temp_workspace_root, "sess-1", "echo", "evt-1-big.jpg", b"\0" * (3 * 1024 * 1024)
        )
        event = _user_event(
            content="big",
            attachments=[
                {
                    "filename": "big.jpg",
                    "content_type": "image/jpeg",
                    "size": 3 * 1024 * 1024,
                    "in_sandbox_path": sandbox_path,
                }
            ],
        )
        msg = render_user_event(
            event,
            "echo/acct/chat-1",
            "echo/acct/chat-1",
            model="model/vision",
            session_id="sess-1",
        )
        assert isinstance(msg["content"], str)
        assert "[image: big.jpg" in msg["content"]
        assert "/mnt/attachments/echo/evt-1-big.jpg" in msg["content"]

    def test_non_vision_mind_renders_marker(self, temp_workspace_root: Path) -> None:
        sandbox_path = _stage_image(temp_workspace_root, "sess-1", "echo", "evt-1-photo.jpg", b"x")
        event = _user_event(
            content="hi",
            attachments=[
                {
                    "filename": "photo.jpg",
                    "content_type": "image/jpeg",
                    "size": 1,
                    "in_sandbox_path": sandbox_path,
                }
            ],
        )
        msg = render_user_event(
            event,
            "echo/acct/chat-1",
            "echo/acct/chat-1",
            model="model/text",
            session_id="sess-1",
        )
        assert isinstance(msg["content"], str)
        assert "[image: photo.jpg" in msg["content"]

    def test_document_emits_attachment_marker(self, temp_workspace_root: Path) -> None:
        sandbox_path = _stage_image(
            temp_workspace_root, "sess-1", "echo", "evt-1-report.pdf", b"%PDF"
        )
        event = _user_event(
            content="here",
            attachments=[
                {
                    "filename": "report.pdf",
                    "content_type": "application/pdf",
                    "size": 4,
                    "in_sandbox_path": sandbox_path,
                }
            ],
        )
        msg = render_user_event(
            event,
            "echo/acct/chat-1",
            "echo/acct/chat-1",
            model="model/vision",
            session_id="sess-1",
        )
        assert isinstance(msg["content"], str)
        assert "[attachment: report.pdf" in msg["content"]

    def test_legacy_stub_no_in_sandbox_path(self) -> None:
        event = _user_event(
            content="legacy",
            attachments=[
                {
                    "filename": "old.jpg",
                    "content_type": "image/jpeg",
                    "size": 1024,
                }
            ],
        )
        msg = render_user_event(
            event,
            "echo/acct/chat-1",
            "echo/acct/chat-1",
            model="model/vision",
            session_id="sess-1",
        )
        assert isinstance(msg["content"], str)
        assert "[image: old.jpg" in msg["content"]

    def test_no_model_disables_inlining(self, temp_workspace_root: Path) -> None:
        """Append-time call (no model passed) emits text markers only.

        That keeps cum_tokens deterministic without a model lookup.
        """
        sandbox_path = _stage_image(
            temp_workspace_root, "sess-1", "echo", "evt-1-photo.jpg", b"bytes"
        )
        event = _user_event(
            attachments=[
                {
                    "filename": "photo.jpg",
                    "content_type": "image/jpeg",
                    "size": 5,
                    "in_sandbox_path": sandbox_path,
                }
            ],
        )
        msg = render_user_event(event, "echo/acct/chat-1", "echo/acct/chat-1")
        assert isinstance(msg["content"], str)
        assert "[image: photo.jpg" in msg["content"]

    def test_image_only_no_caption_no_header_omits_empty_text(
        self, temp_workspace_root: Path
    ) -> None:
        """An image-only event with no caption AND no channel header must
        NOT emit ``{"type":"text","text":""}`` — Anthropic rejects empty
        text blocks (``text content blocks must be non-empty``).  The
        common path is fine because the channel header populates the
        leading text, but legacy events with metadata-stripped headers
        would 400 against Anthropic-routed models.  Regression test for
        PR #218.
        """
        sandbox_path = _stage_image(
            temp_workspace_root, "sess-1", "echo", "evt-1-photo.jpg", b"PNG"
        )
        # Construct an event whose metadata has attachments but lacks
        # ``channel`` — _format_channel_header returns "" so leading_text
        # stays empty.  Content is also empty (image-only inbound).
        event: dict[str, Any] = {
            "role": "user",
            "content": "",
            "metadata": {
                "channel": "echo/acct/chat-1",
                "attachments": [
                    {
                        "filename": "photo.jpg",
                        "content_type": "image/jpeg",
                        "size": 3,
                        "in_sandbox_path": sandbox_path,
                    }
                ],
            },
        }
        # Strip the channel header by zeroing all the fields _format_channel_header
        # consumes after the bare "channel" key: the header is the channel marker
        # plus the inbound content, both empty here, so leading_text is just the
        # bare ``[channel=...]`` line.  To exercise the no-header branch we drop
        # ``channel`` from metadata entirely below.
        event["metadata"].pop("channel")
        # Re-add ``attachments`` only — _format_channel_header returns "" when
        # channel is absent, leaving leading_text empty for the image-only path.
        msg = render_user_event(
            event,
            "echo/acct/chat-1",
            "echo/acct/chat-1",
            model="model/vision",
            session_id="sess-1",
        )
        content = msg["content"]
        assert isinstance(content, list), (
            "image_only message must still emit content parts when image is inlined"
        )
        # No empty text block in the parts.
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                assert part.get("text"), f"empty text part rejected by Anthropic — got {part!r}"
        # And the image_url part is preserved.
        assert any(p.get("type") == "image_url" for p in content)

    def test_mime_corrected_when_event_declares_wrong_type(self, temp_workspace_root: Path) -> None:
        """#342: a persisted event may carry a wrong content_type (Signal /
        Telegram occasionally label JPEG as image/png).  Without correction
        the rendered data URL declares image/png and Anthropic 400s on
        magic-vs-declared mismatch.  Renderer must sniff and substitute.

        This is the historical-event path — the SDK-boundary correction
        only protects new events; long-running sessions still carry bad
        declarations from before the fix, so the renderer is the layer
        that unwedges them at replay time.
        """
        jpeg_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 32
        sandbox_path = _stage_image(
            temp_workspace_root, "sess-1", "echo", "evt-1-lies.png", jpeg_bytes
        )
        event = _user_event(
            content="historical",
            attachments=[
                {
                    "filename": "lies.png",
                    "content_type": "image/png",  # the lie
                    "size": len(jpeg_bytes),
                    "in_sandbox_path": sandbox_path,
                }
            ],
        )
        msg = render_user_event(
            event,
            "echo/acct/chat-1",
            "echo/acct/chat-1",
            model="model/vision",
            session_id="sess-1",
        )
        content = msg["content"]
        assert isinstance(content, list)
        url = content[1]["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,"), (
            f"renderer should have corrected image/png → image/jpeg, got {url[:50]}"
        )

    def test_post_409_attachment_resolves_via_workspace_path(
        self, temp_workspace_root: Path
    ) -> None:
        """Issue #630: an attachment whose ``in_sandbox_path`` is under
        ``/workspace`` must resolve against the explicitly-supplied
        ``workspace_path`` (the post-#409 bind-mount source), not the
        pre-#409 ``workspace_root/session_id`` synthetic path."""
        nested = (temp_workspace_root / "acct-1" / "sess-1").resolve()
        nested.mkdir(parents=True)
        payload = b"JPGNESTED"
        (nested / "img.png").write_bytes(payload)

        event = _user_event(
            content="hi",
            attachments=[
                {
                    "filename": "img.png",
                    "content_type": "image/png",
                    "size": len(payload),
                    "in_sandbox_path": "/workspace/img.png",
                }
            ],
        )
        msg = render_user_event(
            event,
            "echo/acct/chat-1",
            "echo/acct/chat-1",
            model="model/vision",
            session_id="sess-1",
            workspace_path=nested,
        )
        content = msg["content"]
        assert isinstance(content, list), (
            f"expected inlined image_url part; got text-marker fallback: {content!r}"
        )
        url_parts = [p for p in content if p.get("type") == "image_url"]
        assert len(url_parts) == 1
        assert base64.b64encode(payload).decode() in url_parts[0]["image_url"]["url"]

    def test_attachment_workspace_path_missing_returns_none_for_workspace_sandbox_path(
        self, temp_workspace_root: Path
    ) -> None:
        """Fail-closed: when ``workspace_path`` is None AND the
        ``in_sandbox_path`` is under ``/workspace``, the attachment is
        NOT inlined (degrades to a text marker)."""
        # Stage bytes at a legacy synthetic location to make the contrast
        # explicit — even though the file exists at the pre-#409 layout,
        # the renderer must refuse to inline without a workspace_path.
        legacy = (temp_workspace_root / "sess-1").resolve()
        legacy.mkdir(parents=True)
        (legacy / "img.png").write_bytes(b"LEGACY")

        event = _user_event(
            content="hi",
            attachments=[
                {
                    "filename": "img.png",
                    "content_type": "image/png",
                    "size": 6,
                    "in_sandbox_path": "/workspace/img.png",
                }
            ],
        )
        msg = render_user_event(
            event,
            "echo/acct/chat-1",
            "echo/acct/chat-1",
            model="model/vision",
            session_id="sess-1",
            # workspace_path deliberately not passed.
        )
        # No inlined image — must fall back to a text marker.
        assert isinstance(msg["content"], str), (
            f"expected text-marker fallback; got inlined parts: {msg['content']!r}"
        )
        assert "[image: img.png" in msg["content"]

    def test_multiple_attachments_mixed(self, temp_workspace_root: Path) -> None:
        a = _stage_image(temp_workspace_root, "sess-1", "echo", "evt-1-a.jpg", b"AAA")
        b = _stage_image(temp_workspace_root, "sess-1", "echo", "evt-1-b.pdf", b"PDF")
        event = _user_event(
            content="two",
            attachments=[
                {
                    "filename": "a.jpg",
                    "content_type": "image/jpeg",
                    "size": 3,
                    "in_sandbox_path": a,
                },
                {
                    "filename": "b.pdf",
                    "content_type": "application/pdf",
                    "size": 3,
                    "in_sandbox_path": b,
                },
            ],
        )
        msg = render_user_event(
            event,
            "echo/acct/chat-1",
            "echo/acct/chat-1",
            model="model/vision",
            session_id="sess-1",
        )
        content = msg["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "text"
        assert "[attachment: b.pdf" in content[0]["text"]
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
