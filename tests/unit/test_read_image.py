"""Unit coverage for the read tool's image branch.

Stubs the sandbox handle and ``get_session_model`` so the tests never
touch Docker or the DB.  Same pattern as :mod:`test_read_handler`.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aios.config import get_settings
from aios.harness import runtime, vision
from aios.sandbox.backends.base import CommandResult, SandboxHandle
from aios.sandbox.volumes import session_attachments_dir, workspace_dir_for
from aios.tools.read import read_handler
from aios.tools.registry import ToolResult


class _StubRegistry:
    """Minimal stand-in for SandboxRegistry used by handler tests."""

    def __init__(self, handle: SandboxHandle, result: CommandResult) -> None:
        self._handle = handle
        self.exec = AsyncMock(return_value=result)

    async def get_or_provision(self, session_id: str, **_kwargs: Any) -> SandboxHandle:
        return self._handle


@pytest.fixture
def temp_workspace_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    settings = get_settings()
    monkeypatch.setattr(settings, "workspace_root", tmp_path)
    return tmp_path


@pytest.fixture
def stub_handle(temp_workspace_root: Path, **kwargs: Any) -> SandboxHandle:
    # Default to the legacy workspace_dir_for("sess_01TEST") shape so that
    # the existing _stage_workspace_image helper (which writes under
    # workspace_dir_for(session_id)) lines up with the handle's
    # workspace_path. Tests covering the post-#409 layout build their own
    # handle. Depends on temp_workspace_root so that the settings.workspace_root
    # monkeypatch is already in place when workspace_dir_for() is called.
    handle = SandboxHandle(
        session_id="sess_01TEST",
        sandbox_id="container_abc",
        workspace_path=workspace_dir_for("sess_01TEST"),
    )
    return handle


@pytest.fixture
def canned_result() -> CommandResult:
    return CommandResult(
        exit_code=0,
        stdout="",
        stderr="",
        timed_out=False,
        truncated=False,
    )


@pytest.fixture
def stub_runtime(stub_handle: SandboxHandle, canned_result: CommandResult, **kwargs: Any) -> Any:
    prev_registry = runtime.sandbox_registry
    prev_pool = runtime.pool
    stub = _StubRegistry(stub_handle, canned_result)
    runtime.sandbox_registry = stub  # type: ignore[assignment]
    runtime.pool = MagicMock()
    try:
        yield stub
    finally:
        runtime.sandbox_registry = prev_registry
        runtime.pool = prev_pool


@pytest.fixture(autouse=True)
def _vision_overrides(monkeypatch: pytest.MonkeyPatch) -> Any:
    saved = dict(vision._VISION_OVERRIDES)
    vision._VISION_OVERRIDES.clear()
    vision._VISION_OVERRIDES["model/vision"] = True
    vision._VISION_OVERRIDES["model/text"] = False
    yield
    vision._VISION_OVERRIDES.clear()
    vision._VISION_OVERRIDES.update(saved)


@pytest.fixture
def stub_get_session_model(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> Any:
    """Replace the DB lookup with a callable returning whatever the test sets."""

    class _Model:
        value: str = "model/vision"

    state = _Model()

    async def fake(_pool: Any, _session_id: str, **kwargs: Any) -> str:
        return state.value

    monkeypatch.setattr("aios.tools.read.sessions_service.get_session_model", fake)
    return state


def _stage_workspace_image(session_id: str, name: str, payload: bytes) -> Path:
    ws = workspace_dir_for(session_id)
    ws.mkdir(parents=True, exist_ok=True)
    file_path = ws / name
    file_path.write_bytes(payload)
    return file_path


def _stage_attachment(session_id: str, connector: str, name: str, payload: bytes) -> Path:
    target = session_attachments_dir(session_id) / connector
    target.mkdir(parents=True, exist_ok=True)
    file_path = target / name
    file_path.write_bytes(payload)
    return file_path


class TestImageBranch:
    async def test_workspace_image_inlines_for_vision_mind(
        self,
        temp_workspace_root: Path,
        stub_runtime: Any,
        stub_get_session_model: Any,
    ) -> None:
        stub_get_session_model.value = "model/vision"
        _stage_workspace_image("sess_01TEST", "screenshot.png", b"PNGbytes")

        result = await read_handler("sess_01TEST", {"path": "/workspace/screenshot.png"})

        assert isinstance(result, ToolResult)
        assert isinstance(result.content, list)
        assert result.content[0]["type"] == "text"
        assert "Image: screenshot.png" in result.content[0]["text"]
        assert result.content[1] == {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{base64.b64encode(b'PNGbytes').decode()}"},
        }
        assert result.is_error is False

    async def test_attachment_image_inlines_for_vision_mind(
        self,
        temp_workspace_root: Path,
        stub_runtime: Any,
        stub_get_session_model: Any,
    ) -> None:
        stub_get_session_model.value = "model/vision"
        _stage_attachment("sess_01TEST", "echo", "evt-1-photo.jpg", b"JPGbytes")

        result = await read_handler(
            "sess_01TEST", {"path": "/mnt/attachments/echo/evt-1-photo.jpg"}
        )

        assert isinstance(result, ToolResult)
        assert isinstance(result.content, list)
        assert result.content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    async def test_blocked_for_non_vision_mind_is_not_an_error(
        self,
        temp_workspace_root: Path,
        stub_runtime: Any,
        stub_get_session_model: Any,
    ) -> None:
        stub_get_session_model.value = "model/text"
        _stage_workspace_image("sess_01TEST", "screenshot.png", b"PNG")

        result = await read_handler("sess_01TEST", {"path": "/workspace/screenshot.png"})

        assert isinstance(result, ToolResult)
        assert isinstance(result.content, str)
        assert "Mind vision support: no" in result.content
        assert result.is_error is False

    async def test_oversize_image_blocked_with_explanatory_text(
        self,
        temp_workspace_root: Path,
        stub_runtime: Any,
        stub_get_session_model: Any,
    ) -> None:
        from aios.harness.vision import INLINE_SIZE_CAP_BYTES

        stub_get_session_model.value = "model/vision"
        _stage_workspace_image("sess_01TEST", "huge.png", b"\0" * (INLINE_SIZE_CAP_BYTES + 1))

        result = await read_handler("sess_01TEST", {"path": "/workspace/huge.png"})

        assert isinstance(result, ToolResult)
        assert isinstance(result.content, str)
        # Cap rendered with 2-decimal precision + explicit byte count so
        # the model can't confuse a 3.8MB-rounded image with the 3.75 MiB
        # cap (which ``human_size`` would also render as "3.8MB").
        assert f"({INLINE_SIZE_CAP_BYTES} bytes)" in result.content
        assert "Inline cap: 3.75 MiB" in result.content
        assert result.is_error is False

    async def test_image_read_uses_handle_workspace_path(
        self,
        temp_workspace_root: Path,
        stub_get_session_model: Any,
        canned_result: CommandResult,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Post-#409 regression (issue #630): ``_read_image`` must resolve
        ``/workspace/...`` against ``handle.workspace_path`` (the actual
        bind-mount source), NOT the pre-#409 ``workspace_root/session_id``
        layout that ``workspace_dir_for`` returns."""
        stub_get_session_model.value = "model/vision"

        # Construct a handle whose workspace_path is the post-#409 nested
        # ``<workspace_root>/<account>/<session>`` shape — distinct from
        # the pre-#409 path ``workspace_dir_for`` would compute.
        nested = (temp_workspace_root / "acct-X" / "sess_01TEST").resolve()
        nested.mkdir(parents=True)
        payload = b"PNG-NESTED"
        (nested / "img.png").write_bytes(payload)

        handle = SandboxHandle(
            session_id="sess_01TEST",
            sandbox_id="container_abc",
            workspace_path=nested,
        )

        prev_registry = runtime.sandbox_registry
        prev_pool = runtime.pool
        stub = _StubRegistry(handle, canned_result)
        runtime.sandbox_registry = stub  # type: ignore[assignment]
        runtime.pool = MagicMock()
        try:
            result = await read_handler("sess_01TEST", {"path": "/workspace/img.png"})
        finally:
            runtime.sandbox_registry = prev_registry
            runtime.pool = prev_pool

        assert isinstance(result, ToolResult)
        assert isinstance(result.content, list), (
            f"expected inlined image parts; got: {result.content!r}"
        )
        url = result.content[1]["image_url"]["url"]
        assert base64.b64encode(payload).decode() in url

    async def test_missing_image_returns_error(
        self,
        temp_workspace_root: Path,
        stub_runtime: Any,
        stub_get_session_model: Any,
    ) -> None:
        stub_get_session_model.value = "model/vision"
        workspace_dir_for("sess_01TEST").mkdir(parents=True, exist_ok=True)

        result = await read_handler("sess_01TEST", {"path": "/workspace/nope.png"})

        assert isinstance(result, ToolResult)
        assert isinstance(result.content, str)
        assert "file not found" in result.content
        assert result.is_error is True

    async def test_non_bind_mount_falls_back_to_docker_exec(
        self,
        temp_workspace_root: Path,
        stub_handle: SandboxHandle,
        stub_runtime: Any,
        stub_get_session_model: Any,
    ) -> None:
        """Reading ``/etc/foo.png`` (not a bind mount) hits docker-exec
        with one combined stat+base64 invocation."""
        stub_get_session_model.value = "model/vision"
        b64_payload = base64.b64encode(b"otherbytes").decode()
        stub_runtime.exec = AsyncMock(  # type: ignore[method-assign]
            return_value=CommandResult(
                exit_code=0,
                stdout=f"10\n{b64_payload}",
                stderr="",
                timed_out=False,
                truncated=False,
            )
        )

        result = await read_handler("sess_01TEST", {"path": "/etc/foo.png"})

        assert isinstance(result, ToolResult)
        run_command = stub_runtime.exec
        assert isinstance(run_command, AsyncMock)
        assert run_command.call_count == 1
        cmd_arg = run_command.call_args.args[1]
        assert "stat -c %s" in cmd_arg
        assert "base64 -w0" in cmd_arg


class TestImagePathTraversalAttack:
    """End-to-end regression for the path-traversal vulnerability.

    Each case plants a sentinel "host secret" file outside the
    workspace bind-mount root, then asks ``read`` to fetch it via a
    traversal-style sandbox path. The fixed behavior: the host-side
    fast path declines (containment check), the read falls through to
    docker-exec which is correctly contained by the container's
    namespace, and the host-secret bytes never reach the tool result.
    """

    async def test_dotdot_traversal_does_not_return_host_bytes(
        self,
        temp_workspace_root: Path,
        stub_handle: SandboxHandle,
        stub_runtime: Any,
        stub_get_session_model: Any,
    ) -> None:
        stub_get_session_model.value = "model/vision"
        host_secret = b"HOST-SECRET-BYTES-DO-NOT-LEAK"
        (temp_workspace_root / "host_secret.png").write_bytes(host_secret)
        workspace_dir_for("sess_01TEST").mkdir(parents=True, exist_ok=True)

        sandbox_payload = b"sandbox-side-content"
        b64 = base64.b64encode(sandbox_payload).decode()
        stub_runtime.exec = AsyncMock(  # type: ignore[method-assign]
            return_value=CommandResult(
                exit_code=0,
                stdout=f"{len(sandbox_payload)}\n{b64}",
                stderr="",
                timed_out=False,
                truncated=False,
            )
        )

        result = await read_handler("sess_01TEST", {"path": "/workspace/../host_secret.png"})

        b64_secret = base64.b64encode(host_secret).decode()
        if isinstance(result, ToolResult) and isinstance(result.content, list):
            url = result.content[1]["image_url"]["url"]
            assert b64_secret not in url
        elif isinstance(result, dict):
            assert b64_secret not in str(result)
        assert stub_runtime.exec.call_count == 1  # type: ignore[attr-defined]

    async def test_attachments_traversal_does_not_return_host_bytes(
        self,
        temp_workspace_root: Path,
        stub_handle: SandboxHandle,
        stub_runtime: Any,
        stub_get_session_model: Any,
    ) -> None:
        stub_get_session_model.value = "model/vision"
        host_secret = b"ATTACHMENTS-LEAK-PROBE"
        (temp_workspace_root / "leak.png").write_bytes(host_secret)
        session_attachments_dir("sess_01TEST").mkdir(parents=True, exist_ok=True)

        stub_runtime.exec = AsyncMock(  # type: ignore[method-assign]
            return_value=CommandResult(
                exit_code=0,
                stdout=f"{4}\n{base64.b64encode(b'safe').decode()}",
                stderr="",
                timed_out=False,
                truncated=False,
            )
        )

        result = await read_handler("sess_01TEST", {"path": "/mnt/attachments/../../leak.png"})

        b64_secret = base64.b64encode(host_secret).decode()
        if isinstance(result, ToolResult) and isinstance(result.content, list):
            url = result.content[1]["image_url"]["url"]
            assert b64_secret not in url
        elif isinstance(result, dict):
            assert b64_secret not in str(result)

    async def test_symlink_escape_does_not_return_host_bytes(
        self,
        temp_workspace_root: Path,
        stub_handle: SandboxHandle,
        stub_runtime: Any,
        stub_get_session_model: Any,
    ) -> None:
        """The model creates a symlink inside ``/workspace`` whose target
        lives outside the bind-mount root (e.g. via ``ln -s`` from inside
        the sandbox), then ``read``s through it. Containment must follow
        the symlink and reject."""
        stub_get_session_model.value = "model/vision"
        host_secret = b"SYMLINK-ESCAPE-PROBE-BYTES"
        outside = temp_workspace_root / "outside.png"
        outside.write_bytes(host_secret)
        ws = workspace_dir_for("sess_01TEST")
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "sneaky.jpg").symlink_to(outside)

        stub_runtime.exec = AsyncMock(  # type: ignore[method-assign]
            return_value=CommandResult(
                exit_code=0,
                stdout=f"{3}\n{base64.b64encode(b'sbx').decode()}",
                stderr="",
                timed_out=False,
                truncated=False,
            )
        )

        result = await read_handler("sess_01TEST", {"path": "/workspace/sneaky.jpg"})

        b64_secret = base64.b64encode(host_secret).decode()
        if isinstance(result, ToolResult) and isinstance(result.content, list):
            url = result.content[1]["image_url"]["url"]
            assert b64_secret not in url
        elif isinstance(result, dict):
            assert b64_secret not in str(result)
        assert stub_runtime.exec.call_count == 1  # type: ignore[attr-defined]


class TestExtensionlessImageDetection:
    """Issue #715 + the follow-up redesign: a file is routed to the image
    path when its extension names a known image type OR — for any other
    extension (none, or a non-image one like ``.dat``) — when a magic-byte
    sniff of its leading bytes matches.  Connector-staged chat attachments
    arrive at ``/mnt/attachments/<connector>/<filename>`` with arbitrary or
    no extension, so an image among them still inlines as an ``image_url``
    instead of being dumped down the text path as raw mojibake.

    For bind-mounted paths (``/workspace``, ``/mnt/attachments``) the 16-byte
    probe is read locally from the host bind-mount source — the same fast path
    ``_read_image`` uses for the full read — so detection is free (``exec``
    count 0).  Only paths outside any mount fall back to a docker-exec probe.
    """

    async def test_extensionless_png_inlines_via_magic_byte_sniff(
        self,
        temp_workspace_root: Path,
        stub_runtime: Any,
        stub_get_session_model: Any,
    ) -> None:
        stub_get_session_model.value = "model/vision"
        payload = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRrest-of-png-bytes"
        _stage_workspace_image("sess_01TEST", "unnamed", payload)
        result = await read_handler("sess_01TEST", {"path": "/workspace/unnamed"})
        assert isinstance(result, ToolResult)
        assert isinstance(result.content, list)
        assert result.content[1] == {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{base64.b64encode(payload).decode()}"},
        }
        assert result.is_error is False
        assert stub_runtime.exec.call_count == 0  # bind-mounted: probe + full read are both local

    async def test_extensionless_jpeg_inlines_via_magic_byte_sniff(
        self,
        temp_workspace_root: Path,
        stub_runtime: Any,
        stub_get_session_model: Any,
    ) -> None:
        stub_get_session_model.value = "model/vision"
        payload = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00rest-of-jpeg"
        _stage_workspace_image("sess_01TEST", "signal-abc-unnamed", payload)
        result = await read_handler("sess_01TEST", {"path": "/workspace/signal-abc-unnamed"})
        assert isinstance(result, ToolResult)
        assert isinstance(result.content, list)
        assert result.content[1]["image_url"]["url"] == (
            f"data:image/jpeg;base64,{base64.b64encode(payload).decode()}"
        )
        assert stub_runtime.exec.call_count == 0

    async def test_extensionless_gif_inlines_via_magic_byte_sniff(
        self,
        temp_workspace_root: Path,
        stub_runtime: Any,
        stub_get_session_model: Any,
    ) -> None:
        stub_get_session_model.value = "model/vision"
        payload = b"GIF89a\x01\x00\x01\x00\x80\x00\x00rest-of-gif-bytes"
        _stage_workspace_image("sess_01TEST", "img", payload)
        result = await read_handler("sess_01TEST", {"path": "/workspace/img"})
        assert isinstance(result, ToolResult)
        assert isinstance(result.content, list)
        assert result.content[1]["image_url"]["url"].startswith("data:image/gif;base64,")
        assert stub_runtime.exec.call_count == 0

    async def test_wrong_extension_image_inlines_via_magic_byte_sniff(
        self,
        temp_workspace_root: Path,
        stub_runtime: Any,
        stub_get_session_model: Any,
    ) -> None:
        """Finding 1: a NON-image extension whose bytes are a real image is detected by the sniff and inlined."""
        stub_get_session_model.value = "model/vision"
        payload = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00jpeg-mislabeled-as-dat"
        _stage_workspace_image("sess_01TEST", "scan.dat", payload)
        result = await read_handler("sess_01TEST", {"path": "/workspace/scan.dat"})
        assert isinstance(result, ToolResult)
        assert isinstance(result.content, list)
        assert result.content[1]["image_url"]["url"] == (
            f"data:image/jpeg;base64,{base64.b64encode(payload).decode()}"
        )
        assert stub_runtime.exec.call_count == 0

    async def test_text_with_extension_does_not_sniff_as_image(
        self,
        temp_workspace_root: Path,
        stub_runtime: Any,
        stub_get_session_model: Any,
    ) -> None:
        """A non-image-extension TEXT file reads as text: the local probe sniffs to None and the read falls through to the one-exec text path."""
        stub_get_session_model.value = "model/vision"
        _stage_workspace_image("sess_01TEST", "notes.dat", b"plain text, no magic header")
        stub_runtime.exec = AsyncMock(  # type: ignore[method-assign]
            return_value=CommandResult(
                exit_code=0, stdout="     1\thello\n", stderr="", timed_out=False, truncated=False
            )
        )
        result = await read_handler("sess_01TEST", {"path": "/workspace/notes.dat"})
        assert result == {"path": "/workspace/notes.dat", "content": "     1\thello\n"}
        assert stub_runtime.exec.call_count == 1  # only the cat-n; the probe was a local read

    async def test_extensionless_text_falls_through_to_text_path(
        self,
        temp_workspace_root: Path,
        stub_runtime: Any,
        stub_get_session_model: Any,
    ) -> None:
        """An extension-less non-image file must not false-positive into the image path."""
        stub_get_session_model.value = "model/vision"
        _stage_workspace_image(
            "sess_01TEST", "random_name", b"just some text, definitely not an image"
        )
        stub_runtime.exec = AsyncMock(  # type: ignore[method-assign]
            return_value=CommandResult(
                exit_code=0, stdout="     1\thello\n", stderr="", timed_out=False, truncated=False
            )
        )
        result = await read_handler("sess_01TEST", {"path": "/workspace/random_name"})
        assert result == {"path": "/workspace/random_name", "content": "     1\thello\n"}
        assert stub_runtime.exec.call_count == 1

    async def test_extension_path_skips_probe(
        self,
        temp_workspace_root: Path,
        stub_runtime: Any,
        stub_get_session_model: Any,
    ) -> None:
        """An image extension resolves the mime from the extension alone — no probe read."""
        stub_get_session_model.value = "model/vision"
        _stage_workspace_image("sess_01TEST", "screenshot.png", b"PNGbytes")
        result = await read_handler("sess_01TEST", {"path": "/workspace/screenshot.png"})
        assert isinstance(result, ToolResult)
        assert isinstance(result.content, list)
        assert result.content[1]["image_url"]["url"].startswith("data:image/png;base64,")
        assert stub_runtime.exec.call_count == 0

    async def test_non_bind_mount_extensionless_image_sniffs_via_exec(
        self,
        temp_workspace_root: Path,
        stub_runtime: Any,
        stub_get_session_model: Any,
    ) -> None:
        """Outside any bind mount the probe falls back to one docker-exec (with set -o pipefail), then the full read is a second exec."""
        stub_get_session_model.value = "model/vision"
        png_head = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        full = png_head + b"-more-png-bytes"
        stub_runtime.exec = AsyncMock(  # type: ignore[method-assign]
            side_effect=[
                CommandResult(
                    exit_code=0,
                    stdout=base64.b64encode(png_head).decode(),
                    stderr="",
                    timed_out=False,
                    truncated=False,
                ),
                CommandResult(
                    exit_code=0,
                    stdout=f"{len(full)}\n{base64.b64encode(full).decode()}",
                    stderr="",
                    timed_out=False,
                    truncated=False,
                ),
            ]
        )
        result = await read_handler("sess_01TEST", {"path": "/etc/unnamed"})
        assert isinstance(result, ToolResult)
        assert isinstance(result.content, list)
        assert result.content[1]["image_url"]["url"] == (
            f"data:image/png;base64,{base64.b64encode(full).decode()}"
        )
        assert stub_runtime.exec.call_count == 2
        probe_cmd = stub_runtime.exec.call_args_list[0].args[1]
        assert "head -c 16" in probe_cmd
        assert "set -o pipefail" in probe_cmd
