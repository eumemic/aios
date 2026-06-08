"""Spill oversized tool-result content to the session attachments mount.

A tool result larger than ``tool_result_max_chars`` would, stored inline,
exceed the model context window on its own (windowing can only drop whole
events, never shrink one). Capping it here at the append boundary keeps
``cumulative_tokens`` honest. The full body is written to the session's
attachments dir (bind-mounted read-only at ``/mnt/attachments`` every
provision), recoverable by the model with the ``read`` tool (#735).

Only ``str`` content is capped (the two append sinks guard on
``isinstance(content, str)``); structured/list tool-result content — rare and
typically bounded — passes through uncapped, as do oversized non-tool events
(e.g. an unbounded ``switch_channel`` recap).  Tool results are the dominant
overflow source; the residual shapes are left for a follow-up.
"""

from __future__ import annotations

import asyncio

from aios.sandbox.volumes import ensure_session_attachments_dir, safe_filename

_SPILL_SUBDIR = "tool_results"


async def cap_tool_result_content(
    session_id: str, tool_call_id: str, content: str, *, max_chars: int
) -> str:
    """Return ``content`` unchanged if within ``max_chars``; otherwise spill the
    full body to ``<attachments>/tool_results/<tool_call_id>.txt`` and return a
    stub string referencing the in-sandbox path + the ``read`` tool."""
    if len(content) <= max_chars:
        return content
    fname = safe_filename(f"{tool_call_id}.txt")
    sandbox_path = f"/mnt/attachments/{_SPILL_SUBDIR}/{fname}"
    await asyncio.to_thread(_write_spill, session_id, fname, content)
    return (
        f"[Tool result truncated: the full output ({len(content):,} characters) "
        f"exceeded the inline result limit and was saved to {sandbox_path}. "
        f"Use the read tool to view it.]"
    )


def _write_spill(session_id: str, fname: str, content: str) -> None:
    spill_dir = ensure_session_attachments_dir(session_id) / _SPILL_SUBDIR
    spill_dir.mkdir(parents=True, exist_ok=True)
    target = spill_dir / fname
    tmp = target.with_name(target.name + ".part")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(target)  # atomic rename
