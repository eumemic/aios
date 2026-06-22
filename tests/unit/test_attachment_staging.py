"""Unit coverage for :mod:`aios.services.attachment_staging`.

Exercises the multipart-stream → disk state machine with a temp
``workspace_root`` so we never touch the production attachments
directory. Streams are simulated with a small in-memory shim that
matches the :class:`aios.services.files.UploadStream` Protocol.
"""

from __future__ import annotations

import functools
import io
import random
from pathlib import Path

import pytest
from PIL import Image

from aios.config import get_settings
from aios.harness.vision import INLINE_SIZE_CAP_BYTES
from aios.sandbox.volumes import safe_filename
from aios.services.attachment_staging import (
    AttachmentStagingError,
    InboundAttachment,
    stage_inbound_attachments,
)


@functools.cache
def _real_jpeg_bytes(width: int, height: int, *, quality: int = 95) -> bytes:
    """Return real JPEG bytes Pillow can decode (the staging downsample
    path actually opens the file, so opaque ``b"\\0"`` payloads aren't
    enough to exercise it).

    The pixel data is genuine seeded noise so the JPEG re-encode in the
    staging path really shrinks the multi-megapixel cases below
    ``INLINE_SIZE_CAP_BYTES`` (solid fills would compress to nothing and
    defeat the test). ``Random(seed).randbytes`` + :meth:`Image.frombytes`
    replaces the old per-pixel Python loop (12.25M writes for the
    3500x3500 case, regenerated identically per test); ``functools.cache``
    memoizes the immutable ``bytes`` result so repeated identical
    requests are free.
    """
    data = random.Random(f"{width}x{height}q{quality}").randbytes(width * height * 3)
    img = Image.frombytes("RGB", (width, height), data)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


@pytest.fixture
def temp_workspace_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the cached ``Settings`` at a tmpdir for the test."""
    settings = get_settings()
    monkeypatch.setattr(settings, "workspace_root", tmp_path)
    return tmp_path


class _FakeUploadStream:
    """In-memory ``UploadStream`` shim for tests.

    Mirrors the chunked-read contract of ``fastapi.UploadFile`` without
    pulling starlette into the unit-test path.
    """

    def __init__(self, payload: bytes, filename: str, content_type: str) -> None:
        self._payload = payload
        self._pos = 0
        self.filename: str | None = filename
        self.content_type: str | None = content_type

    async def read(self, size: int = -1) -> bytes:
        if self._pos >= len(self._payload):
            return b""
        if size < 0:
            chunk = self._payload[self._pos :]
            self._pos = len(self._payload)
            return chunk
        chunk = self._payload[self._pos : self._pos + size]
        self._pos += len(chunk)
        return chunk


def _attachment(
    payload: bytes, filename: str, content_type: str = "image/jpeg"
) -> InboundAttachment:
    return InboundAttachment(
        stream=_FakeUploadStream(payload, filename, content_type),
        filename=filename,
        content_type=content_type,
    )


class TestSafeFilename:
    def test_strips_path_separators(self) -> None:
        assert safe_filename("../../etc/passwd") == "passwd"

    def test_replaces_unsafe_chars(self) -> None:
        assert safe_filename("hello world!.jpg") == "hello_world_.jpg"

    def test_preserves_dots_and_dashes(self) -> None:
        assert safe_filename("photo-2026.05.04.jpg") == "photo-2026.05.04.jpg"

    def test_empty_falls_back_to_unnamed(self) -> None:
        assert safe_filename("") == "unnamed"

    def test_all_dots_falls_back_to_unnamed(self) -> None:
        assert safe_filename("...") == "unnamed"

    def test_caps_length(self) -> None:
        result = safe_filename("a" * 500)
        assert len(result) <= 200


class TestStaging:
    async def test_empty_returns_empty(self, temp_workspace_root: Path) -> None:
        assert await stage_inbound_attachments(
            session_id="sess-1",
            connector_name="echo",
            event_id="evt-1",
            attachments=[],
        ) == ([], [])

    async def test_happy_path_streams_to_disk_and_returns_record(
        self, temp_workspace_root: Path
    ) -> None:
        att = _attachment(b"jpegbytes", "photo.jpg")

        records, staged_paths = await stage_inbound_attachments(
            session_id="sess-1",
            connector_name="echo",
            event_id="evt-1",
            attachments=[att],
        )

        assert records == [
            {
                "filename": "photo.jpg",
                "content_type": "image/jpeg",
                "size": len(b"jpegbytes"),
                "in_sandbox_path": "/mnt/attachments/echo/evt-1-photo.jpg",
            }
        ]
        staged = temp_workspace_root / "_attachments" / "sess-1" / "echo" / "evt-1-photo.jpg"
        assert staged.exists()
        assert staged.read_bytes() == b"jpegbytes"
        assert staged_paths == [staged]

    async def test_multiple_attachments_all_staged(self, temp_workspace_root: Path) -> None:
        a = _attachment(b"AAA", "a.jpg")
        b = _attachment(b"BBBB", "b.png", content_type="image/png")

        records, staged_paths = await stage_inbound_attachments(
            session_id="sess-1",
            connector_name="echo",
            event_id="evt-1",
            attachments=[a, b],
        )

        assert len(records) == 2
        assert records[0]["filename"] == "a.jpg"
        assert records[1]["filename"] == "b.png"
        assert len(staged_paths) == 2

    async def test_unsafe_filename_sanitized_in_staged_path(
        self, temp_workspace_root: Path
    ) -> None:
        att = _attachment(b"x", "../escape/bad name.jpg")
        records, _ = await stage_inbound_attachments(
            session_id="sess-1",
            connector_name="echo",
            event_id="evt-1",
            attachments=[att],
        )
        # Filename in the record is the original (model-facing display);
        # the sanitized form only shows up in the staged path.
        assert records[0]["filename"] == "../escape/bad name.jpg"
        assert records[0]["in_sandbox_path"] == "/mnt/attachments/echo/evt-1-bad_name.jpg"

    async def test_replay_with_existing_target_skips_stream(
        self, temp_workspace_root: Path
    ) -> None:
        """Idempotent replay: same event_id delivered twice doesn't double-stage."""
        first_records, first_paths = await stage_inbound_attachments(
            session_id="sess-1",
            connector_name="echo",
            event_id="evt-1",
            attachments=[_attachment(b"first", "photo.jpg")],
        )
        # Replay: connector re-sends the SAME event_id with a fresh stream
        # (the SDK would re-emit from spool). The target is already on disk
        # so the stream is left untouched and size is read from disk.
        second_records, second_paths = await stage_inbound_attachments(
            session_id="sess-1",
            connector_name="echo",
            event_id="evt-1",
            attachments=[_attachment(b"first", "photo.jpg")],
        )
        assert first_records == second_records
        # First call materialized one path; replay materialized none —
        # critical so the compensating unlink on downstream dedup failure
        # doesn't blow away bytes already referenced by the prior event.
        assert len(first_paths) == 1
        assert second_paths == []

    async def test_same_inbound_filename_collision_fails_hard(
        self, temp_workspace_root: Path
    ) -> None:
        """Two attachments in the same inbound that sanitize to the
        same target name must fail loudly — silently appending a
        second record pointing at the first attachment's bytes would
        corrupt ``metadata.attachments``.
        """
        a = _attachment(b"AAA", "image.jpg")
        b = _attachment(b"BBB", "image.jpg")

        with pytest.raises(AttachmentStagingError, match="sanitize to the same"):
            await stage_inbound_attachments(
                session_id="sess-1",
                connector_name="echo",
                event_id="evt-1",
                attachments=[a, b],
            )

        # Compensating action: the first iteration's staged file is
        # rolled back so the orphan GC has nothing to clean up.
        sess_dir = temp_workspace_root / "_attachments" / "sess-1" / "echo"
        if sess_dir.exists():
            assert list(sess_dir.iterdir()) == []

    async def test_collision_via_sanitization_fails_hard(self, temp_workspace_root: Path) -> None:
        """Two distinct filenames that sanitize to identical names
        (``image_.jpg`` and ``image .jpg`` → both end up
        ``evt-1-image_.jpg`` after the space → ``_`` mapping) also collide.
        """
        a = _attachment(b"AAA", "image_.jpg")
        b = _attachment(b"BBB", "image .jpg")

        with pytest.raises(AttachmentStagingError, match="sanitize to the same"):
            await stage_inbound_attachments(
                session_id="sess-1",
                connector_name="echo",
                event_id="evt-1",
                attachments=[a, b],
            )

        sess_dir = temp_workspace_root / "_attachments" / "sess-1" / "echo"
        if sess_dir.exists():
            assert list(sess_dir.iterdir()) == []

    async def test_concurrent_same_event_id_keeps_winners_file_on_disk(
        self, temp_workspace_root: Path
    ) -> None:
        """Webhook retries (Telegram resend on 5xx, Signal at-least-once)
        can deliver the same inbound twice concurrently. Both calls pass
        ``target.exists()`` before either ``os.rename`` lands, both write
        to the SAME ``.part`` sibling, one wins the rename, the other's
        rename raises ``FileNotFoundError``. The loser's
        ``except BaseException`` then unconditionally ran
        ``target.unlink(missing_ok=True)`` — DELETING the winner's
        freshly-renamed file. The winner's call returned a
        ``staged_records`` entry pointing at the now-missing path; the
        renderer would later fail to open it (or worse, the GC sweep
        would race with the actual write).

        To reproduce the race in a single asyncio event loop the stream
        ``read`` is wrapped with an ``asyncio.sleep(0)`` yield so the
        scheduler can interleave the two calls between
        ``target.exists()`` and ``os.rename`` — the same shape real
        ``UploadFile`` streams produce when reading from a real socket.

        Fix: don't unlink ``target`` in the loser's cleanup path —
        nothing this call owns lives at ``target``.
        """
        import asyncio

        class _SlowStream:
            """Yield to the event loop after every chunk so two
            concurrent ``stage_inbound_attachments`` calls actually
            interleave between ``target.exists()`` and ``os.rename``."""

            filename: str | None = None
            content_type: str | None = None

            def __init__(self, payload: bytes) -> None:
                self._payload = payload
                self._pos = 0

            async def read(self, size: int = -1) -> bytes:
                await asyncio.sleep(0)
                if self._pos >= len(self._payload):
                    return b""
                chunk = (
                    self._payload[self._pos :]
                    if size < 0
                    else self._payload[self._pos : self._pos + size]
                )
                self._pos += len(chunk)
                return chunk

        a = InboundAttachment(
            stream=_SlowStream(b"A" * 100), filename="img.jpg", content_type="image/jpeg"
        )
        b = InboundAttachment(
            stream=_SlowStream(b"B" * 100), filename="img.jpg", content_type="image/jpeg"
        )

        results = await asyncio.gather(
            stage_inbound_attachments(
                session_id="sess-1",
                connector_name="echo",
                event_id="evt-1",
                attachments=[a],
            ),
            stage_inbound_attachments(
                session_id="sess-1",
                connector_name="echo",
                event_id="evt-1",
                attachments=[b],
            ),
            return_exceptions=True,
        )

        # At least one call should have succeeded (the rename winner).
        successes = [r for r in results if not isinstance(r, BaseException)]
        assert len(successes) >= 1, f"at least one stage must succeed; got {results!r}"

        target = temp_workspace_root / "_attachments" / "sess-1" / "echo" / "evt-1-img.jpg"
        assert target.exists(), (
            f"the rename winner's file must survive — pre-fix the loser's "
            f"``except BaseException`` cleanup unlinked target, deleting the "
            f"file the winner returned a ``staged_records`` reference to. "
            f"Subsequent renderer reads (or attachment_gc orphan-sweep races) "
            f"would then fail. Directory contents: "
            f"{list((temp_workspace_root / '_attachments' / 'sess-1' / 'echo').iterdir()) if (temp_workspace_root / '_attachments' / 'sess-1' / 'echo').exists() else 'dir missing'}"
        )


class TestInlineDownsample:
    """Coverage for the staging-time inline-downsample path.

    The bytes are real JPEGs so Pillow's decode succeeds and the size
    relationship to ``INLINE_SIZE_CAP_BYTES`` is genuine — synthetic
    ``b"\\0" * N`` payloads decode as garbage and the staging code
    would log warn + skip inline instead of exercising the resize.
    """

    async def test_under_cap_image_has_no_inline_subrecord(self, temp_workspace_root: Path) -> None:
        small = _real_jpeg_bytes(200, 200)
        assert len(small) < INLINE_SIZE_CAP_BYTES

        records, _ = await stage_inbound_attachments(
            session_id="sess-1",
            connector_name="echo",
            event_id="evt-1",
            attachments=[
                InboundAttachment(
                    stream=_FakeUploadStream(small, "small.jpg", "image/jpeg"),
                    filename="small.jpg",
                    content_type="image/jpeg",
                )
            ],
        )
        assert "inline" not in records[0]
        sess_dir = temp_workspace_root / "_attachments" / "sess-1" / "echo"
        assert sorted(p.name for p in sess_dir.iterdir()) == ["evt-1-small.jpg"]

    async def test_oversize_image_produces_inline_subrecord_and_sibling(
        self, temp_workspace_root: Path
    ) -> None:
        # Big and noisy enough that the JPEG re-encode genuinely shrinks
        # bytes (palette-PNG and dimension downscale both kick in).
        big = _real_jpeg_bytes(3500, 3500)
        assert len(big) > INLINE_SIZE_CAP_BYTES

        records, staged_paths = await stage_inbound_attachments(
            session_id="sess-1",
            connector_name="echo",
            event_id="evt-1",
            attachments=[
                InboundAttachment(
                    stream=_FakeUploadStream(big, "big.jpg", "image/jpeg"),
                    filename="big.jpg",
                    content_type="image/jpeg",
                )
            ],
        )

        record = records[0]
        assert record["filename"] == "big.jpg"
        assert record["size"] == len(big)
        assert record["in_sandbox_path"] == "/mnt/attachments/echo/evt-1-big.jpg"

        inline = record["inline"]
        assert inline["content_type"] == "image/jpeg"
        assert inline["size"] <= INLINE_SIZE_CAP_BYTES
        assert inline["width"] <= 2000
        assert inline["height"] <= 2000
        assert inline["in_sandbox_path"] == ("/mnt/attachments/echo/evt-1-big.jpg.inline.jpg")

        # Both original and inline live on disk and both are registered
        # for compensating cleanup on downstream rollback.
        sess_dir = temp_workspace_root / "_attachments" / "sess-1" / "echo"
        on_disk = sorted(p.name for p in sess_dir.iterdir())
        assert on_disk == ["evt-1-big.jpg", "evt-1-big.jpg.inline.jpg"]
        assert len(staged_paths) == 2

    async def test_non_image_content_type_skips_inline(self, temp_workspace_root: Path) -> None:
        big_pdf = b"%PDF-" + b"\0" * (INLINE_SIZE_CAP_BYTES + 1)
        records, _ = await stage_inbound_attachments(
            session_id="sess-1",
            connector_name="echo",
            event_id="evt-1",
            attachments=[
                InboundAttachment(
                    stream=_FakeUploadStream(big_pdf, "doc.pdf", "application/pdf"),
                    filename="doc.pdf",
                    content_type="application/pdf",
                )
            ],
        )
        assert "inline" not in records[0]

    async def test_undecodable_oversize_image_falls_through_without_inline(
        self, temp_workspace_root: Path
    ) -> None:
        """An image-typed payload Pillow can't decode (corrupt header,
        truncated bytes) skips the inline path silently — the renderer
        still gets a marker, but the original is staged normally so the
        sandbox tools can read it.
        """
        garbage = b"\xff\xd8\xff" + b"\x00" * (INLINE_SIZE_CAP_BYTES + 1)
        records, staged_paths = await stage_inbound_attachments(
            session_id="sess-1",
            connector_name="echo",
            event_id="evt-1",
            attachments=[
                InboundAttachment(
                    stream=_FakeUploadStream(garbage, "bad.jpg", "image/jpeg"),
                    filename="bad.jpg",
                    content_type="image/jpeg",
                )
            ],
        )
        assert "inline" not in records[0]
        # Original still on disk (staging-time downsample failure must
        # not block the upload — model can still ``read`` the path).
        sess_dir = temp_workspace_root / "_attachments" / "sess-1" / "echo"
        assert (sess_dir / "evt-1-bad.jpg").exists()
        assert len(staged_paths) == 1

    async def test_replay_reconstructs_inline_record_from_disk(
        self, temp_workspace_root: Path
    ) -> None:
        """Webhook replays of an event already on disk skip the upload
        stream and read size from ``stat``.  When the inline sibling
        also exists, we reconstruct the sub-record from disk rather
        than re-encoding.
        """
        big = _real_jpeg_bytes(3500, 3500)
        first_records, first_paths = await stage_inbound_attachments(
            session_id="sess-1",
            connector_name="echo",
            event_id="evt-1",
            attachments=[
                InboundAttachment(
                    stream=_FakeUploadStream(big, "big.jpg", "image/jpeg"),
                    filename="big.jpg",
                    content_type="image/jpeg",
                )
            ],
        )
        second_records, second_paths = await stage_inbound_attachments(
            session_id="sess-1",
            connector_name="echo",
            event_id="evt-1",
            attachments=[
                InboundAttachment(
                    stream=_FakeUploadStream(big, "big.jpg", "image/jpeg"),
                    filename="big.jpg",
                    content_type="image/jpeg",
                )
            ],
        )
        # Replay returns the same record shape, and materializes zero
        # new files (both original and inline are on disk from the
        # first call).
        assert first_records == second_records
        assert second_paths == []
        assert len(first_paths) == 2  # original + inline sibling

    async def test_compensating_cleanup_unlinks_inline_sibling(
        self, temp_workspace_root: Path
    ) -> None:
        """Same-inbound collision (two attachments sanitizing to the
        same name) fails hard; both the original and the inline sibling
        from the first iteration must be unlinked so the orphan GC
        sees nothing.
        """
        big = _real_jpeg_bytes(3500, 3500)
        small = _real_jpeg_bytes(100, 100)

        with pytest.raises(AttachmentStagingError, match="sanitize to the same"):
            await stage_inbound_attachments(
                session_id="sess-1",
                connector_name="echo",
                event_id="evt-1",
                attachments=[
                    InboundAttachment(
                        stream=_FakeUploadStream(big, "image.jpg", "image/jpeg"),
                        filename="image.jpg",
                        content_type="image/jpeg",
                    ),
                    InboundAttachment(
                        stream=_FakeUploadStream(small, "image.jpg", "image/jpeg"),
                        filename="image.jpg",
                        content_type="image/jpeg",
                    ),
                ],
            )

        sess_dir = temp_workspace_root / "_attachments" / "sess-1" / "echo"
        if sess_dir.exists():
            assert list(sess_dir.iterdir()) == []


def _solid_jpeg_bytes(width: int, height: int) -> bytes:
    """Return a small JPEG with given dimensions.  Solid-color content
    compresses to a few KB regardless of dimensions, exercising the
    small-bytes-large-dim path that the byte-size guard would otherwise
    let through to providers uncapped."""
    img = Image.new("RGB", (width, height), (200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _gif_bytes(width: int, height: int) -> bytes:
    """Return a real GIF (used to verify the replay-path format
    mapping rejects non-JPEG/PNG siblings rather than mislabeling)."""
    img = Image.new("P", (width, height), 0)
    buf = io.BytesIO()
    img.save(buf, format="GIF")
    return buf.getvalue()


class TestInlineDownsampleHardening:
    """Coverage for the post-code-review hardening of the inline path:
    dimension cap, collision pre-reservation, replay-path resilience,
    ceiling observability, atomic writes."""

    async def test_small_bytes_large_dim_still_downsampled(self, temp_workspace_root: Path) -> None:
        """A small but high-resolution JPEG (e.g. a 3000x3000 solid-color
        gradient compressing to a few KB) must STILL be downsampled —
        the pre-fix byte-size guard let these through, then Anthropic
        bricked the session on its 8000px hard limit.
        """
        wide_solid = _solid_jpeg_bytes(3000, 3000)
        assert len(wide_solid) < INLINE_SIZE_CAP_BYTES  # bypasses the byte guard

        records, _ = await stage_inbound_attachments(
            session_id="sess-1",
            connector_name="echo",
            event_id="evt-1",
            attachments=[
                InboundAttachment(
                    stream=_FakeUploadStream(wide_solid, "wide.jpg", "image/jpeg"),
                    filename="wide.jpg",
                    content_type="image/jpeg",
                )
            ],
        )
        record = records[0]
        assert "inline" in record, (
            "small-bytes/large-dim image must still produce an inline sub-record "
            "so the renderer doesn't ship dims-over-cap bytes to providers"
        )
        assert record["inline"]["width"] <= 2000
        assert record["inline"]["height"] <= 2000

    async def test_inline_name_collision_pre_reserved(self, temp_workspace_root: Path) -> None:
        """A user uploading both ``photo.jpg`` (triggers inline) and
        ``photo.jpg.inline.jpg`` must fail hard via the same-inbound
        collision guard — the second can't be allowed to silently
        inherit the first's downsampled bytes by hitting the replay
        branch on the auto-generated inline sibling."""
        big = _real_jpeg_bytes(3500, 3500)
        small = _real_jpeg_bytes(50, 50)

        with pytest.raises(AttachmentStagingError, match="sanitize to the same"):
            await stage_inbound_attachments(
                session_id="sess-1",
                connector_name="echo",
                event_id="evt-1",
                attachments=[
                    InboundAttachment(
                        stream=_FakeUploadStream(big, "photo.jpg", "image/jpeg"),
                        filename="photo.jpg",
                        content_type="image/jpeg",
                    ),
                    InboundAttachment(
                        stream=_FakeUploadStream(small, "photo.jpg.inline.jpg", "image/jpeg"),
                        filename="photo.jpg.inline.jpg",
                        content_type="image/jpeg",
                    ),
                ],
            )

    async def test_corrupt_replay_inline_sibling_falls_back_to_marker(
        self, temp_workspace_root: Path
    ) -> None:
        """A partial/corrupt ``.inline.jpg`` sibling (e.g. left by a
        prior staging crash mid-write) must NOT propagate Pillow's
        ``OSError`` past staging — pre-fix this 500'd the inbound and
        the connector would retry forever against the same bytes."""
        big = _real_jpeg_bytes(3500, 3500)

        # Manually seed a corrupt inline sibling so the replay path
        # finds it on the call (simulating a prior crash that left it
        # behind).
        sess_dir = temp_workspace_root / "_attachments" / "sess-1" / "echo"
        sess_dir.mkdir(parents=True, exist_ok=True)
        original_target = sess_dir / "evt-1-photo.jpg"
        original_target.write_bytes(big)  # original is on disk (replay)
        corrupt_inline = sess_dir / "evt-1-photo.jpg.inline.jpg"
        corrupt_inline.write_bytes(b"\xff\xd8\xff\x00" + b"\x00" * 50)

        records, _ = await stage_inbound_attachments(
            session_id="sess-1",
            connector_name="echo",
            event_id="evt-1",
            attachments=[
                InboundAttachment(
                    stream=_FakeUploadStream(big, "photo.jpg", "image/jpeg"),
                    filename="photo.jpg",
                    content_type="image/jpeg",
                )
            ],
        )
        # Inbound completes successfully — no exception propagated.
        assert "inline" not in records[0]
        # Original is preserved.
        assert original_target.exists()

    async def test_unrecognized_format_in_inline_sibling_falls_back(
        self, temp_workspace_root: Path
    ) -> None:
        """An on-disk ``.inline.jpg`` whose bytes are actually GIF (e.g.
        operator restore or future-format regression) must NOT be
        mislabeled as image/png and shipped to providers — providers
        reject mime/magic mismatches and brick the session."""
        big = _real_jpeg_bytes(3500, 3500)

        sess_dir = temp_workspace_root / "_attachments" / "sess-1" / "echo"
        sess_dir.mkdir(parents=True, exist_ok=True)
        original_target = sess_dir / "evt-1-photo.jpg"
        original_target.write_bytes(big)
        # GIF bytes at the .inline.jpg sibling path — Pillow detects
        # format='GIF' which is not in the JPEG-or-PNG map.
        gif_at_jpg_name = sess_dir / "evt-1-photo.jpg.inline.jpg"
        gif_at_jpg_name.write_bytes(_gif_bytes(100, 100))

        records, _ = await stage_inbound_attachments(
            session_id="sess-1",
            connector_name="echo",
            event_id="evt-1",
            attachments=[
                InboundAttachment(
                    stream=_FakeUploadStream(big, "photo.jpg", "image/jpeg"),
                    filename="photo.jpg",
                    content_type="image/jpeg",
                )
            ],
        )
        # Replay-path treats unrecognized format as decode failure →
        # warn + return None → no inline sub-record on the persisted
        # record.  Renderer falls through to text marker; model can
        # still ``read`` the original from sandbox.
        assert "inline" not in records[0]

    async def test_ceiling_skip_emits_warn_log(
        self,
        temp_workspace_root: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Above-ceiling images must warn-log so operators can spot a
        systematic class of inbound silently degrading.  Pre-fix the
        ceiling branch returned None with zero log signal."""
        from aios.harness.vision import PRE_RESIZE_CEILING_BYTES

        oversize = b"\xff\xd8\xff" + b"\x00" * (PRE_RESIZE_CEILING_BYTES + 10)

        records, _ = await stage_inbound_attachments(
            session_id="sess-1",
            connector_name="echo",
            event_id="evt-1",
            attachments=[
                InboundAttachment(
                    stream=_FakeUploadStream(oversize, "huge.jpg", "image/jpeg"),
                    filename="huge.jpg",
                    content_type="image/jpeg",
                )
            ],
        )
        assert "inline" not in records[0]
        # Structlog routes via stdlib logging in this project, so caplog
        # captures the event.  Match against either the message or the
        # structlog event key.
        ceiling_logs = [
            r
            for r in caplog.records
            if "inline_skipped_ceiling" in r.getMessage()
            or "inline_skipped_ceiling" in str(getattr(r, "event", ""))
        ]
        assert ceiling_logs, (
            f"expected a staging.inline_skipped_ceiling log; got: "
            f"{[r.getMessage() for r in caplog.records]}"
        )

    async def test_inline_write_atomic_no_part_left_after_success(
        self, temp_workspace_root: Path
    ) -> None:
        """Successful inline staging must leave no .part orphans.
        Confirms the atomic write convention so any leftover .part files
        in production genuinely indicate a crashed write (catchable by
        GC) rather than the success path."""
        big = _real_jpeg_bytes(3500, 3500)

        records, _ = await stage_inbound_attachments(
            session_id="sess-1",
            connector_name="echo",
            event_id="evt-1",
            attachments=[
                InboundAttachment(
                    stream=_FakeUploadStream(big, "big.jpg", "image/jpeg"),
                    filename="big.jpg",
                    content_type="image/jpeg",
                )
            ],
        )
        assert "inline" in records[0]
        sess_dir = temp_workspace_root / "_attachments" / "sess-1" / "echo"
        part_files = [p for p in sess_dir.iterdir() if p.name.endswith(".part")]
        assert part_files == [], (
            f"successful staging must leave no .part orphans; got {part_files!r}"
        )
