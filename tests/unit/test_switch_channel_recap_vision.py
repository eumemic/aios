"""Vision-aware behaviour for :func:`render_reorient_block` (issue #226).

When the model bound to a session is vision-capable and a peer event
on the target channel carries inlinable image attachments, the recap
must surface those images as ``image_url`` content parts — not just
text path markers.  Without this, multi-channel sessions silently lose
vision on every non-focal image: arrival-time inlining only fires
while the channel is focal, so the only chance to surface pixels
across a ``switch_channel`` call is here.

The arrival-time path lives in
:func:`aios.harness.context.render_user_event` and is exercised by
``test_context_attachments.py``; this file pins the recap-time path,
which reuses the same vision policy via ``model``/``session_id``.
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from aios.config import get_settings
from aios.harness import vision
from aios.models.events import Event
from aios.sandbox.volumes import session_attachments_dir
from aios.tools.switch_channel import render_reorient_block

CHAN_A = "signal/acct/chat-a"
CHAN_B = "signal/acct/chat-b"
SESSION_ID = "sess-test"


@pytest.fixture
def temp_workspace_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    settings = get_settings()
    monkeypatch.setattr(settings, "workspace_root", tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def _stub_supports_vision(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> Any:
    saved = dict(vision._VISION_OVERRIDES)
    vision._VISION_OVERRIDES.clear()
    vision._VISION_OVERRIDES["model/vision"] = True
    vision._VISION_OVERRIDES["model/text"] = False
    yield
    vision._VISION_OVERRIDES.clear()
    vision._VISION_OVERRIDES.update(saved)


def _stage_image(workspace_root: Path, connector: str, name: str, payload: bytes) -> str:
    session_dir = session_attachments_dir(SESSION_ID) / connector
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / name).write_bytes(payload)
    return f"/mnt/attachments/{connector}/{name}"


def _user_with_image(
    seq: int,
    *,
    channel: str,
    content: str,
    sandbox_path: str,
    filename: str = "photo.jpg",
    content_type: str = "image/jpeg",
    size: int = 9,
) -> Event:
    return Event(
        id=f"evt_{seq:04d}",
        session_id=SESSION_ID,
        seq=seq,
        kind="message",
        data={
            "role": "user",
            "content": content,
            "metadata": {
                "channel": channel,
                "sender_name": "Peer",
                "timestamp_ms": 1000 + seq,
                "attachments": [
                    {
                        "filename": filename,
                        "content_type": content_type,
                        "size": size,
                        "in_sandbox_path": sandbox_path,
                    }
                ],
            },
        },
        cumulative_tokens=None,
        created_at=datetime(2026, 4, 20, tzinfo=UTC),
        orig_channel=channel,
        focal_channel_at_arrival=channel,
        channel=channel,
    )


def _user_text_only(seq: int, *, channel: str, content: str) -> Event:
    return Event(
        id=f"evt_{seq:04d}",
        session_id=SESSION_ID,
        seq=seq,
        kind="message",
        data={
            "role": "user",
            "content": content,
            "metadata": {
                "channel": channel,
                "sender_name": "Peer",
                "timestamp_ms": 1000 + seq,
            },
        },
        cumulative_tokens=None,
        created_at=datetime(2026, 4, 20, tzinfo=UTC),
        orig_channel=channel,
        focal_channel_at_arrival=channel,
        channel=channel,
    )


class TestRecapVision:
    def test_inlinable_image_emits_image_url_content_part(self, temp_workspace_root: Path) -> None:
        """A peer image on the target channel must appear as an
        ``image_url`` content part in the recap when the bound mind
        supports vision and the image fits the inline cap.  The recap
        return type widens from ``str`` to a content-parts list to
        carry the multimodal payload.
        """
        payload = b"jpegbytes"
        sandbox_path = _stage_image(temp_workspace_root, "signal", "evt-1-photo.jpg", payload)
        events = [
            _user_with_image(
                1,
                channel=CHAN_A,
                content="check this out",
                sandbox_path=sandbox_path,
                size=len(payload),
            )
        ]
        out = render_reorient_block(events, CHAN_A, model="model/vision", session_id=SESSION_ID)
        assert isinstance(out, list), (
            f"recap with inlinable image must return content-parts list, got {type(out).__name__}"
        )
        image_parts = [p for p in out if p.get("type") == "image_url"]
        assert len(image_parts) == 1, f"expected one image_url part, got {image_parts!r}"
        assert image_parts[0]["image_url"]["url"] == (
            f"data:image/jpeg;base64,{base64.b64encode(payload).decode()}"
        )

    def test_image_recap_preserves_recap_framing_and_caption(
        self, temp_workspace_root: Path
    ) -> None:
        """The text parts surrounding the image must still carry the
        ``Recap: recent messages on <target>`` header, the peer
        message's text caption, and the ``End recap`` footer.  The
        image content block sits between the framing text parts so
        the model reads "this is historical content from the channel,
        and here's the picture they sent".
        """
        sandbox_path = _stage_image(temp_workspace_root, "signal", "evt-1-photo.jpg", b"PNGdata")
        events = [
            _user_with_image(
                1,
                channel=CHAN_A,
                content="look at this banana",
                sandbox_path=sandbox_path,
                size=7,
            )
        ]
        out = render_reorient_block(events, CHAN_A, model="model/vision", session_id=SESSION_ID)
        assert isinstance(out, list)
        text_blob = "\n".join(p["text"] for p in out if p.get("type") == "text")
        assert f"Recap: recent messages on {CHAN_A}" in text_blob
        assert "End recap" in text_blob
        assert "look at this banana" in text_blob

    def test_non_vision_model_returns_str_with_text_marker(self, temp_workspace_root: Path) -> None:
        """Vision-incapable mind: recap stays a single ``str`` and the
        attachment is rendered as a text path marker, same policy as
        arrival-time rendering in ``render_user_event``.
        """
        sandbox_path = _stage_image(temp_workspace_root, "signal", "evt-1-photo.jpg", b"jpg")
        events = [
            _user_with_image(
                1,
                channel=CHAN_A,
                content="picture",
                sandbox_path=sandbox_path,
                size=3,
            )
        ]
        out = render_reorient_block(events, CHAN_A, model="model/text", session_id=SESSION_ID)
        assert isinstance(out, str)
        assert "[image: photo.jpg" in out
        assert sandbox_path in out

    def test_text_only_recap_unchanged(self) -> None:
        """No attachments on any window event: recap returns a ``str``
        with the same framing as before.  This is the no-images
        backwards-compat path — vision plumbing must not change the
        output for sessions that never produce inlinable content.
        """
        events = [_user_text_only(1, channel=CHAN_A, content="hello")]
        out = render_reorient_block(events, CHAN_A, model="model/vision", session_id=SESSION_ID)
        assert isinstance(out, str)
        assert "Recap: recent messages on" in out
        assert "hello" in out

    def test_no_model_kwargs_preserves_legacy_str_behavior(self, temp_workspace_root: Path) -> None:
        """The pre-#226 callers pass only ``(events, target)``.  Without
        ``model`` / ``session_id``, the recap stays string-only and any
        image attachments degrade to text markers — the behavior the
        code shipped with before this issue.
        """
        sandbox_path = _stage_image(temp_workspace_root, "signal", "evt-1-photo.jpg", b"jpg")
        events = [
            _user_with_image(
                1,
                channel=CHAN_A,
                content="picture",
                sandbox_path=sandbox_path,
                size=3,
            )
        ]
        out = render_reorient_block(events, CHAN_A)
        assert isinstance(out, str)
        assert "[image: photo.jpg" in out
