"""The screencast frame writer: basename manifest, per-boot seq, atomic
writes, and ENOENT tolerance when the reaper deletes the frames dir."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, cast

from aios_browser_driver.takeover.screencast import Screencast, _chrome_of

_JPEG = base64.b64encode(b"\xff\xd8\xff\xe0jpegbytes\xff\xd9").decode("ascii")


def _screencast(frames_dir: Path) -> Screencast:
    seq = iter(range(1, 1000))
    return Screencast(
        cast(Any, None),  # the context is unused by _persist
        frames_dir,
        boot="01BOOT",
        epoch=7,
        next_seq=lambda: next(seq),
    )


def test_persist_writes_a_basename_manifest_and_a_complete_frame(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    sc = _screencast(frames)
    sc._persist(_JPEG, {"deviceWidth": 1280, "deviceHeight": 800})

    manifest = json.loads((frames / "manifest.json").read_text())
    assert manifest["seq"] == 1
    assert manifest["file"] == "frame-1.jpg"  # frames-dir-relative BASENAME
    assert "/" not in manifest["file"]
    assert manifest["epoch"] == 7 and manifest["boot"] == "01BOOT"
    assert manifest["w"] == 1280 and manifest["h"] == 800
    # The frame the manifest points at exists and holds the whole JPEG.
    assert (frames / manifest["file"]).read_bytes() == base64.b64decode(_JPEG)


def test_no_temp_files_survive_a_write(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    sc = _screencast(frames)
    sc._persist(_JPEG, {})
    # tmp+rename leaves no partial artifacts a reader could trip over.
    leftovers = [p.name for p in frames.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


def test_seq_advances_and_the_previous_frame_is_unlinked(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    sc = _screencast(frames)
    sc._persist(_JPEG, {})
    sc._persist(_JPEG, {})
    manifest = json.loads((frames / "manifest.json").read_text())
    assert manifest["seq"] == 2
    # The dir stays O(1): only the current frame survives, the previous is gone.
    assert (frames / "frame-2.jpg").exists()
    assert not (frames / "frame-1.jpg").exists()


def test_persist_recreates_a_deleted_frames_dir(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    sc = _screencast(frames)
    sc._persist(_JPEG, {})
    # The reaper deletes the dir between frames; the next persist recreates it.
    for child in frames.iterdir():
        child.unlink()
    frames.rmdir()
    sc._persist(_JPEG, {})
    assert (frames / "manifest.json").exists()


def test_missing_dimensions_fall_back_to_the_viewport(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    sc = _screencast(frames)
    sc._persist(_JPEG, {})  # no metadata
    manifest = json.loads((frames / "manifest.json").read_text())
    assert manifest["w"] == 1280 and manifest["h"] == 800


def test_on_frame_enqueues_int_session_ids(tmp_path: Path) -> None:
    # CDP delivers screencastFrame.sessionId as an INT — a str-only guard would
    # silently drop every frame (regression guard).
    sc = _screencast(tmp_path / "frames")
    sc._on_frame({"data": _JPEG, "sessionId": 7, "metadata": {}})
    assert sc._queue.qsize() == 1
    _data, _meta, session_id = sc._queue.get_nowait()
    assert session_id == 7


def test_chrome_derives_security_from_the_url_scheme() -> None:
    # Chromium stopped emitting Security.securityStateChanged, so security is
    # derived from the committed URL's scheme, not a dead CDP event.
    assert _chrome_of("https://bank.example/login") == ("https://bank.example", "secure")
    assert _chrome_of("http://shop.example/") == ("http://shop.example", "insecure")
    assert _chrome_of("about:blank") == (None, None)
    assert _chrome_of("data:text/html,x") == (None, None)
