"""Unit coverage for vision-aware rendering in :func:`render_user_event`.

The renderer's vision policy is delegated to
:mod:`aios.harness.vision`; here we exercise the wiring around it —
host-bytes-read for inlinable images, text-marker fallback for
non-vision models and oversize images, and the legacy-stub path.
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from aios.config import get_settings
from aios.harness import vision
from aios.harness.context import (
    _apply_attachments,
    build_messages,
)
from aios.harness.context import (
    render_user_event as _render_user_event_impl,
)
from aios.models.events import Event
from aios.sandbox.volumes import session_attachments_dir
from tests.helpers.images import valid_jpeg_bytes, valid_tiff_bytes

# These tests exercise attachment/vision rendering, not the `received=` envelope
# (their assertions are substring checks). Inject a fixed created_at so callers
# needn't thread one through; the resulting `received` field is inert here.
_CREATED_AT = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


_VALID_JPEG = valid_jpeg_bytes()


def render_user_event(
    event_data: dict[str, Any],
    orig_channel: str | None,
    focal_channel_at_arrival: str | None,
    **kwargs: Any,
) -> dict[str, Any]:
    return _render_user_event_impl(
        event_data, orig_channel, focal_channel_at_arrival, _CREATED_AT, **kwargs
    )


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
            temp_workspace_root, "sess-1", "echo", "evt-1-photo.jpg", _VALID_JPEG
        )
        event = _user_event(
            content="hello",
            attachments=[
                {
                    "filename": "photo.jpg",
                    "content_type": "image/jpeg",
                    "size": len(_VALID_JPEG),
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
                "url": f"data:image/jpeg;base64,{base64.b64encode(_VALID_JPEG).decode()}"
            },
        }

    def test_undecodable_under_cap_image_falls_back_to_marker(
        self, temp_workspace_root: Path
    ) -> None:
        """An image-typed attachment under the inline cap whose bytes Pillow
        cannot decode (a corrupt body behind a valid JPEG magic prefix, or a
        zero-byte file) must degrade to a text marker, NOT be inlined.

        Providers full-decode an inline image and 400 on such bytes; the bytes
        are immutable in the event log, so inlining them re-sends the rejected
        part on every wake — bricking the turn into a terminal error the model
        never sees (it's a provider-side 400, not something the model emitted).
        The render boundary must apply the provider's own decodability verdict.
        """
        corrupt = b"\xff\xd8\xff" + b"garbage" * 4  # valid JPEG magic, undecodable body
        sandbox_path = _stage_image(temp_workspace_root, "sess-1", "echo", "evt-1-bad.jpg", corrupt)
        event = _user_event(
            content="look",
            attachments=[
                {
                    "filename": "bad.jpg",
                    "content_type": "image/jpeg",
                    "size": len(corrupt),
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
        # Marker-only render: content collapses to a string (no image_url part).
        assert isinstance(msg["content"], str), "undecodable image must not be inlined"
        assert "[image: bad.jpg" in msg["content"]
        assert "/mnt/attachments/echo/evt-1-bad.jpg" in msg["content"]

    def test_decodable_but_provider_unsupported_format_falls_back_to_marker(
        self, temp_workspace_root: Path
    ) -> None:
        """A TIFF under the cap DECODES in Pillow (so the decode gate passes
        it) but no vision provider accepts TIFF — they take only
        jpeg/png/gif/webp and 400 on the rest, bricking the turn on every wake.
        The render boundary gates on the ACTUAL decoded format (not the
        declared mime, which can lie either way) and degrades an unsupported
        one to a text marker the model can still ``read``.
        """
        payload = valid_tiff_bytes()
        sandbox_path = _stage_image(
            temp_workspace_root, "sess-1", "echo", "evt-1-scan.tiff", payload
        )
        event = _user_event(
            content="scan",
            attachments=[
                {
                    "filename": "scan.tiff",
                    "content_type": "image/tiff",
                    "size": len(payload),
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
        # Unsupported format → marker-only render (content collapses to a str).
        assert isinstance(msg["content"], str), "unsupported image format must not be inlined"
        assert "[image: scan.tiff" in msg["content"]

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
            temp_workspace_root, "sess-1", "echo", "evt-1-photo.jpg", _VALID_JPEG
        )
        mixed: list[Any] = [
            "stray-string",
            {
                "filename": "photo.jpg",
                "content_type": "image/jpeg",
                "size": len(_VALID_JPEG),
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

    def test_oversize_without_inline_falls_back_to_marker(self, temp_workspace_root: Path) -> None:
        """An attachment whose original exceeds the inline cap but has no
        ``inline`` sub-record (e.g. staging-time downsample failed, or the
        event predates the auto-downsample feature) renders as a marker.
        """
        oversize = vision.INLINE_SIZE_CAP_BYTES + 1
        sandbox_path = _stage_image(
            temp_workspace_root, "sess-1", "echo", "evt-1-big.jpg", b"\0" * oversize
        )
        event = _user_event(
            content="big",
            attachments=[
                {
                    "filename": "big.jpg",
                    "content_type": "image/jpeg",
                    "size": oversize,
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

    def test_oversize_with_inline_renders_inline_bytes(self, temp_workspace_root: Path) -> None:
        """When staging produced a downsampled ``inline`` sibling, the
        renderer prefers it: the model sees the resized pixels rather
        than the path marker.
        """
        oversize = vision.INLINE_SIZE_CAP_BYTES + 1
        _stage_image(temp_workspace_root, "sess-1", "echo", "evt-1-big.jpg", b"\0" * oversize)
        inline_payload = _VALID_JPEG
        inline_sandbox_path = _stage_image(
            temp_workspace_root,
            "sess-1",
            "echo",
            "evt-1-big.jpg.inline.jpg",
            inline_payload,
        )
        event = _user_event(
            content="big",
            attachments=[
                {
                    "filename": "big.jpg",
                    "content_type": "image/jpeg",
                    "size": oversize,
                    "in_sandbox_path": "/mnt/attachments/echo/evt-1-big.jpg",
                    "inline": {
                        "in_sandbox_path": inline_sandbox_path,
                        "content_type": "image/jpeg",
                        "size": len(inline_payload),
                        "width": 2000,
                        "height": 1500,
                    },
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
        image_parts = [p for p in content if p.get("type") == "image_url"]
        assert len(image_parts) == 1
        expected_url = f"data:image/jpeg;base64,{base64.b64encode(inline_payload).decode()}"
        assert image_parts[0]["image_url"]["url"] == expected_url

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

    def test_apply_attachments_omits_empty_text_block(self, temp_workspace_root: Path) -> None:
        """``_apply_attachments`` must NOT emit ``{"type":"text","text":""}``
        when the leading text is empty — Anthropic rejects empty text blocks
        (``text content blocks must be non-empty``), which would 400 and wedge
        the session on a caption-less image-only inbound.  Regression test for
        PR #218.

        Tested at ``_apply_attachments``'s own boundary with an empty
        ``leading_text``: the render path always prepends a non-empty
        ``[received=…]`` envelope now, so the function's self-guard must stay
        correct independent of what its caller happens to prepend.
        """
        sandbox_path = _stage_image(
            temp_workspace_root, "sess-1", "echo", "evt-1-photo.jpg", _VALID_JPEG
        )
        msg: dict[str, Any] = {"role": "user", "content": ""}  # empty leading text
        _apply_attachments(
            msg,
            [
                {
                    "filename": "photo.jpg",
                    "content_type": "image/jpeg",
                    "size": len(_VALID_JPEG),
                    "in_sandbox_path": sandbox_path,
                }
            ],
            model="model/vision",
            session_id="sess-1",
        )
        content = msg["content"]
        assert isinstance(content, list), (
            "image-only message must still emit content parts when the image inlines"
        )
        # The guard: an empty leading text must yield NO text part at all, never
        # an empty one. If the omission broke, content would carry
        # {"type":"text","text":""} and this assertion would fire.
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                assert part.get("text"), f"empty text part rejected by Anthropic — got {part!r}"
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
        jpeg_bytes = _VALID_JPEG
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
        payload = _VALID_JPEG
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
        a = _stage_image(temp_workspace_root, "sess-1", "echo", "evt-1-a.jpg", _VALID_JPEG)
        b = _stage_image(temp_workspace_root, "sess-1", "echo", "evt-1-b.pdf", b"PDF")
        event = _user_event(
            content="two",
            attachments=[
                {
                    "filename": "a.jpg",
                    "content_type": "image/jpeg",
                    "size": len(_VALID_JPEG),
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

    def test_vanished_staged_file_renders_normally_not_quarantined(
        self, temp_workspace_root: Path
    ) -> None:
        """The inner ``context.attachment_read_failed`` OSError guard
        pre-empts the outer #686 quarantine: when the staged file has
        vanished (manual cleanup, FS corruption, GC race), the renderer
        falls back to the ``[image: …]`` text marker rather than raising —
        so ``build_messages`` produces a normal user message with NO
        ``[unrenderable`` quarantine marker."""
        sandbox_path = _stage_image(
            temp_workspace_root, "sess-1", "echo", "evt-1-photo.jpg", b"jpegbytes"
        )
        # The host path mirrors how _stage_image laid the file down.
        host_path = session_attachments_dir("sess-1") / "echo" / "evt-1-photo.jpg"
        event_data = _user_event(
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
        # Delete the staged file so read_bytes raises OSError mid-render.
        host_path.unlink()
        event = Event(
            id="evt_1",
            session_id="sess-1",
            seq=1,
            kind="message",
            data=event_data,
            created_at=_CREATED_AT,
            orig_channel="echo/acct/chat-1",
            focal_channel_at_arrival="echo/acct/chat-1",
        )
        ctx = build_messages([event], system_prompt=None, model="model/vision", session_id="sess-1")
        content = ctx.messages[0]["content"]
        assert isinstance(content, str)
        assert "[unrenderable" not in content
        assert "[image: photo.jpg" in content


class TestNonFocalAttachmentRendering:
    """Issue #718: attachments on a NON-focal channel must surface a
    ``read``-able ``text_marker`` — never inlined off-channel, never
    silently dropped (the pre-#718 behavior).

    Pixel recovery across a ``switch_channel`` is handled separately by
    the reorient recap (#226); these pin the arrival-time breadcrumb.
    The only contrast with :class:`TestVisionAwareRendering` is that
    ``focal_channel_at_arrival`` differs from ``orig_channel``.
    """

    def test_nonfocal_attachment_emits_marker_not_dropped(self, temp_workspace_root: Path) -> None:
        """A vision model + a present, inlinable image still renders as a
        text marker off-channel: the non-focal branch never inlines, and
        pre-#718 dropped the attachment entirely (no marker, no pixels)."""
        sandbox_path = _stage_image(
            temp_workspace_root, "sess-1", "echo", "evt-1-photo.jpg", b"jpegbytes"
        )
        event = _user_event(
            content="check this",
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
            "echo/acct/chat-1",  # orig_channel
            "other/acct/chat-9",  # focal_channel_at_arrival differs → non-focal
            model="model/vision",
            session_id="sess-1",
        )
        content = msg["content"]
        assert isinstance(content, str)
        assert content.startswith("🔔 channel_id=echo/acct/chat-1")
        assert "[image: photo.jpg" in content
        assert "/mnt/attachments/echo/evt-1-photo.jpg" in content
        # Off-channel is markers-only — no inlined pixels.
        assert "image_url" not in content
        assert "data:image" not in content

    def test_nonfocal_non_image_attachment_emits_attachment_marker(self) -> None:
        """Non-image attachments get the ``[attachment: …]`` marker
        off-channel too.  No staging needed — ``text_marker`` reads only
        the record dict, never the bytes."""
        event = _user_event(
            content="doc",
            attachments=[
                {
                    "filename": "report.pdf",
                    "content_type": "application/pdf",
                    "size": 4,
                    "in_sandbox_path": "/mnt/attachments/echo/evt-1-report.pdf",
                }
            ],
        )
        msg = render_user_event(
            event,
            "echo/acct/chat-1",
            "other/acct/chat-9",
            model="model/vision",
            session_id="sess-1",
        )
        content = msg["content"]
        assert isinstance(content, str)
        assert content.startswith("🔔 channel_id=echo/acct/chat-1")
        assert "[attachment: report.pdf" in content

    def test_nonfocal_non_dict_attachment_skipped(self) -> None:
        """A malformed (non-dict) record on the non-focal path is skipped,
        not crashed on — the same brick-the-session guard as the focal
        path's ``test_non_dict_attachment_record_is_skipped``."""
        malformed: list[Any] = ["oops", None, 42]
        event = _user_event(content="hi", attachments=malformed)
        msg = render_user_event(
            event,
            "echo/acct/chat-1",
            "other/acct/chat-9",
            model="model/vision",
            session_id="sess-1",
        )
        content = msg["content"]
        assert isinstance(content, str)
        assert content.startswith("🔔 channel_id=echo/acct/chat-1")
        # No attachment markers emitted for malformed records.
        assert "[image:" not in content
        assert "[attachment:" not in content
