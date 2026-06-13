"""E2E regression for issue #715 — the read tool must inline images whose
filenames carry no extension.

Connector-staged chat attachments are saved at
``/mnt/attachments/<connector>/<filename>`` where ``<filename>`` may have no
extension (the reported Signal file was literally ``…-unnamed``).
Extension-only detection routed these to the text path, so the tool result
was raw image bytes decoded as UTF-8 mojibake and the model never saw an
``image_url``.  The fix sniffs the leading bytes when the extension is not a
known image type.

This pins both probe paths in a live container.  Each case is written and read
under ``/workspace`` AND ``/tmp``:

* ``/workspace`` is a bind mount, so the 16-byte probe is read locally from the
  host bind-mount source (the same fast path ``_read_image`` uses for the full
  read) — no docker-exec.
* ``/tmp`` is NOT a bind mount, so the probe exercises the real
  ``set -o pipefail; head -c 16 -- <path> | base64 -w0`` docker-exec and the
  full read goes through ``_stat_and_read_via_exec``.

bash writes extension-less PNG/JPEG/GIF files and the read tool inlines each as
an ``image_url`` with the correct mime.
"""

from __future__ import annotations

import base64

import pytest

from aios.harness import vision
from aios.tools.bash import bash_handler
from aios.tools.read import read_handler
from aios.tools.registry import ToolResult
from tests.conftest import needs_docker
from tests.e2e.harness import Harness
from tests.helpers.images import valid_gif_bytes, valid_jpeg_bytes, valid_png_bytes

pytestmark = pytest.mark.docker

# (extension-less filename, full decodable image bytes, expected mime). Real
# images, not magic-prefix fragments: the read tool now full-decodes before
# inlining (the provider 400s on undecodable bytes), so this sniff-routing test
# must stage bytes that both sniff AND decode.
_CASES = [
    ("unnamed_png", valid_png_bytes(), "image/png"),
    ("unnamed_jpg", valid_jpeg_bytes(), "image/jpeg"),
    ("unnamed_gif", valid_gif_bytes(), "image/gif"),
]


@needs_docker
class TestExtensionlessImageRead:
    async def test_extensionless_images_round_trip_through_read(
        self,
        docker_harness: Harness,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The fake test model carries no LiteLLM metadata; pin vision on so
        # ``_read_image`` doesn't bail on the "vision support: no" branch.
        monkeypatch.setitem(vision._VISION_OVERRIDES, "fake/test", True)

        session = await docker_harness.start("read extension-less images", tools=["bash", "read"])

        # ``/workspace`` exercises the local bind-mount probe; ``/tmp`` is not a
        # bind mount, so it exercises the real ``set -o pipefail; head -c 16 |
        # base64`` exec probe + ``_stat_and_read_via_exec`` full read.
        for parent in ("/workspace", "/tmp"):
            for name, payload, expected_mime in _CASES:
                b64 = base64.b64encode(payload).decode("ascii")
                write_cmd = f"printf '%s' {b64} | base64 -d > {parent}/{name}"
                write_result = await bash_handler(session.id, {"command": write_cmd})
                assert write_result["exit_code"] == 0, (parent, name, write_result)

                read_result = await read_handler(session.id, {"path": f"{parent}/{name}"})

                assert isinstance(read_result, ToolResult), (parent, name, read_result)
                assert read_result.is_error is False, (parent, name, read_result)
                assert isinstance(read_result.content, list), (
                    f"{parent}/{name}: expected inlined image parts, got {read_result.content!r}"
                )
                image_part = read_result.content[1]
                assert image_part["type"] == "image_url", (parent, name, image_part)
                assert image_part["image_url"]["url"] == f"data:{expected_mime};base64,{b64}", (
                    parent,
                    name,
                )
