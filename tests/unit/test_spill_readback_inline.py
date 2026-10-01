"""Following a spill stub must not re-spill (F1, #2292 follow-up).

Real ``cap_tool_result``, real ``read_handler`` (sandbox exec swapped for local
bash against the real spill dir), real ``_shape_tool_result``: reading a spill
file exactly as the stub instructs must come back inline, never spilled again.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.config import get_settings
from aios.harness import runtime
from aios.harness.tool_dispatch import _shape_tool_result, _ToolCall
from aios.sandbox.backends.base import CommandResult, SandboxHandle
from aios.sandbox.tool_result_spill import cap_tool_result
from aios.sandbox.volumes import ensure_session_attachments_dir
from aios.tools.read import read_handler

_SESSION_ID = "sess_spill_readback"


class _LocalBashRegistry:
    """Runs the read tool's command with local bash, mapping the in-sandbox
    ``/mnt/attachments`` path to the session's host attachments dir."""

    def __init__(self, handle: SandboxHandle, host_root: Path) -> None:
        self._handle = handle
        self._host_root = host_root

    async def get_or_provision(self, session_id: str, **_kw: Any) -> SandboxHandle:
        return self._handle

    async def exec(
        self, handle: SandboxHandle, cmd: str, *, timeout_seconds: int, max_output_bytes: int
    ) -> CommandResult:
        cmd = cmd.replace("/mnt/attachments", str(self._host_root))
        proc = await asyncio.create_subprocess_exec(
            "bash", "-c", cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        out, err = await proc.communicate()
        stdout = out.decode("utf-8", errors="replace")
        truncated = len(out) > max_output_bytes
        if truncated:
            stdout = stdout[:max_output_bytes] + "\n\n[output truncated]"
        return CommandResult(
            exit_code=proc.returncode or 0,
            stdout=stdout,
            stderr=err.decode(),
            timed_out=False,
            truncated=truncated,
        )


def _content(n_lines: int, width: int) -> str:
    return "".join(
        f'row {i:05d} | {{"k": "v\\t{i}", "pad": "{"x" * width}"}}\n' for i in range(n_lines)
    )


@pytest.fixture
def local_sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(get_settings(), "workspace_root", tmp_path)
    host_root = ensure_session_attachments_dir(_SESSION_ID)
    handle = SandboxHandle(owner_id=_SESSION_ID, sandbox_id="local", workspace_path=tmp_path)
    prev_reg, prev_pool = runtime.sandbox_registry, runtime.pool
    runtime.sandbox_registry = _LocalBashRegistry(handle, host_root)  # type: ignore[assignment]
    runtime.pool = MagicMock()
    try:
        with patch(
            "aios.tools.read.resolve_bash_timeout_ceiling",
            new_callable=AsyncMock,
            return_value=60,
        ):
            yield
    finally:
        runtime.sandbox_registry, runtime.pool = prev_reg, prev_pool


async def _dispatch_read(call_id: str, args: dict[str, Any]) -> tuple[str, Any]:
    result = await read_handler(_SESSION_ID, args)
    tc = _ToolCall(call_id=call_id, name="read", raw_args=args, bound_log=MagicMock())
    shaped = _shape_tool_result(tc, result)
    capped = await cap_tool_result(
        _SESSION_ID,
        call_id,
        shaped["content"],
        max_chars=get_settings().tool_result_max_chars,
    )
    assert isinstance(capped.content, str)
    return capped.content, capped.attachment


@pytest.mark.parametrize(("n_lines", "width"), [(500, 10), (8_000, 20), (3, 30_000)])
async def test_read_of_spill_file_per_stub_stays_inline(
    local_sandbox: Any, n_lines: int, width: int
) -> None:
    assert get_settings().tool_result_max_chars == 16_000
    original = _content(n_lines, width)
    stub, attachment = await _dispatch_read_original(original)
    assert attachment is not None, "fixture result should have spilled"
    m = re.search(r"saved to (/mnt/attachments/\S+?\.txt)", stub)
    assert m is not None
    spill_path = m.group(1)

    # Exactly what the stub says: read the named path, no extra arguments.
    content, att = await _dispatch_read("tc_read_1", {"path": spill_path})
    assert att is None, "reading the spill file as the stub instructs re-spilled it"
    assert isinstance(content, str)
    payload = json.loads(content)
    assert payload["path"] == spill_path
    assert payload["content"].startswith("     1\trow 00000")

    # Paging via next_offset recovers the whole file inline, never re-spilling.
    pieces = [payload["content"]]
    hops = 0
    while payload.get("truncated"):
        hops += 1
        assert hops < 500
        content, att = await _dispatch_read(
            f"tc_read_{hops + 1}", {"path": spill_path, "offset": payload["next_offset"]}
        )
        assert att is None, f"paged read hop {hops} re-spilled"
        payload = json.loads(content)
        pieces.append(payload["content"])
    if n_lines >= 500:  # whole lines fit → full verbatim recovery via paging
        recovered = "".join(ln.split("\t", 1)[1] for ln in "".join(pieces).splitlines(True))
        assert recovered == original


async def _dispatch_read_original(original: str) -> tuple[str, Any]:
    capped = await cap_tool_result(
        _SESSION_ID, "tc_orig", original, max_chars=get_settings().tool_result_max_chars
    )
    assert isinstance(capped.content, str)
    return capped.content, capped.attachment
