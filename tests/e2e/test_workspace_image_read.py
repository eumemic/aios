"""E2E regression for issue #660 — the read tool's image branch must
inline files the bash tool just wrote to ``/workspace/...``.

Pre-#642 (issue #630), ``resolve_to_host_path`` recomputed the host-side
bind-mount source as ``<workspace_root>/<session_id>``, omitting the
``account_id`` segment that post-#409 sessions carry on
``workspace_volume_path``.  The image branch then opened a non-existent
path and surfaced ``"file not found"`` to the model even though bash had
just written the file.  #642 threaded the actual ``workspace_path`` from
the live :class:`SandboxHandle` through the resolver.

This test pins the round-trip end-to-end with a real container + bind mount:
bash writes a sentinel into the post-#409 nested host path and the read
tool inlines those exact bytes as an ``image_url`` part.
"""

from __future__ import annotations

import base64

import pytest

from aios.harness import runtime, vision
from aios.tools.bash import bash_handler
from aios.tools.read import read_handler
from aios.tools.registry import ToolResult
from tests.conftest import needs_docker
from tests.e2e.harness import Harness


@needs_docker
class TestWorkspaceImageRead:
    async def test_bash_written_workspace_image_round_trips_through_read(
        self,
        docker_harness: Harness,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The fake test model has no LiteLLM metadata, so pin vision
        # support on so ``_read_image`` does not bail with the
        # "Mind vision support: no" branch.
        monkeypatch.setitem(vision._VISION_OVERRIDES, "fake/test", True)

        # ``harness.start`` creates a post-#409 session: default
        # ``workspace_volume_path`` =
        # ``<workspace_root>/acc_test_stub/<session_id>`` (see
        # ``db/queries.py:insert_session``).
        session = await docker_harness.start("write then read an image", tools=["bash", "read"])

        # Bash writes a known-bytes ``.png`` into ``/workspace`` from
        # inside the container.  The container sees ``/workspace`` as the
        # bind mount; the host sees it at
        # ``<workspace_root>/<account_id>/<session_id>``.
        sentinel = b"PNG-SENTINEL-660"
        b64 = base64.b64encode(sentinel).decode("ascii")
        write_cmd = f"printf '%s' {b64} | base64 -d > /workspace/img.png"
        write_result = await bash_handler(session.id, {"command": write_cmd})
        assert write_result["exit_code"] == 0, write_result

        # Read tool's image branch resolves ``/workspace/img.png`` to the
        # host path via ``resolve_to_host_path(..., workspace_path=handle.workspace_path)``.
        # Pre-#642 the resolver returned a wrong path and ``read_bytes``
        # raised ``FileNotFoundError``.
        read_result = await read_handler(session.id, {"path": "/workspace/img.png"})

        assert isinstance(read_result, ToolResult), read_result
        assert read_result.is_error is False
        assert isinstance(read_result.content, list), (
            f"expected inlined image parts; got: {read_result.content!r}"
        )
        # ``content[0]`` is a text label part, ``content[1]`` is the
        # ``image_url`` part — see ``_read_image`` in ``tools/read.py``.
        image_part = read_result.content[1]
        assert image_part["type"] == "image_url"
        assert b64 in image_part["image_url"]["url"]

        # Cross-check: the host-side bind-mount source actually contains
        # the bytes bash wrote, pinning that we exercised the real mount.
        sandbox = runtime.require_sandbox_registry()
        handle = await sandbox.get_or_provision(session.id, pool=docker_harness._pool)
        assert (handle.workspace_path / "img.png").read_bytes() == sentinel
