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

Spill files are recorded in the tool-result event's
``metadata.attachments`` (#1093): the attachment GC's referenced-set query
(``list_attachment_paths_for_sessions``) is built *exclusively* from
``data->'metadata'->'attachments'``, so a spill file whose only reference
lived in the result-content stub was invisible to the GC and reaped on the
next worker boot — pointing the model at a now-missing file it was told to
``read``.  Recording the spill under the same ``metadata.attachments``
convention every staged inbound uses collapses the two reference conventions
into one: there is exactly one way a file under ``_attachments/`` becomes
live, and the existing referenced-set query covers it automatically.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from aios.sandbox.volumes import (
    ensure_owned_dir,
    ensure_session_attachments_dir,
    safe_filename,
)

_SPILL_SUBDIR = "tool_results"


@dataclass(frozen=True)
class CappedToolResult:
    """Outcome of :func:`cap_tool_result_content`.

    ``content`` is the value to store on the tool-result event — the
    original when within the cap, or the truncation stub when spilled.
    ``attachment`` is ``None`` when no spill occurred, otherwise the
    ``metadata.attachments`` record the caller MUST persist on the event
    so the attachment GC's referenced-set sees the spill file as live
    (#1093).  The record mirrors a staged-inbound record's shape — it
    carries ``in_sandbox_path`` (the key the GC query keys on) plus
    descriptive fields for the renderer/console.
    """

    content: str
    attachment: dict[str, Any] | None


async def cap_tool_result_content(
    session_id: str, tool_call_id: str, content: str, *, max_chars: int
) -> CappedToolResult:
    """Return ``content`` unchanged if within ``max_chars``; otherwise spill the
    full body to ``<attachments>/tool_results/<tool_call_id>.txt`` and return a
    stub string referencing the in-sandbox path + the ``read`` tool, paired
    with the ``metadata.attachments`` record the caller must persist so the
    spill file is protected from the orphan GC (#1093)."""
    if len(content) <= max_chars:
        return CappedToolResult(content=content, attachment=None)
    fname = safe_filename(f"{tool_call_id}.txt")
    sandbox_path = f"/mnt/attachments/{_SPILL_SUBDIR}/{fname}"
    size = len(content)
    await asyncio.to_thread(_write_spill, session_id, fname, content)
    stub = (
        f"[Tool result truncated: the full output ({size:,} characters) "
        f"exceeded the inline result limit and was saved to {sandbox_path}. "
        f"Use the read tool to view it.]"
    )
    attachment = {
        "filename": fname,
        "content_type": "text/plain",
        "size": size,
        "in_sandbox_path": sandbox_path,
        # Marks the reference convention this record came in through so a
        # future reader can tell a spill record from a staged inbound.
        "source": "tool_result_spill",
    }
    return CappedToolResult(content=stub, attachment=attachment)


def record_spill_attachment(data: dict[str, Any], attachment: dict[str, Any] | None) -> None:
    """Append ``attachment`` to ``data['metadata']['attachments']`` in place.

    No-op when ``attachment`` is ``None`` (the within-cap path).  This is the
    single seam both tool-result append sinks use to register a spill file
    under the same ``metadata.attachments`` convention staged inbounds use, so
    the attachment GC's referenced-set query protects it (#1093).  Creates the
    ``metadata`` dict and ``attachments`` list if absent; appends if a list is
    already present.
    """
    if attachment is None:
        return
    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        data["metadata"] = metadata
    existing = metadata.get("attachments")
    if isinstance(existing, list):
        existing.append(attachment)
    else:
        metadata["attachments"] = [attachment]


def _write_spill(session_id: str, fname: str, content: str) -> None:
    spill_dir = ensure_session_attachments_dir(session_id) / _SPILL_SUBDIR
    ensure_owned_dir(spill_dir)
    target = spill_dir / fname
    tmp = target.with_name(target.name + ".part")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(target)  # atomic rename
