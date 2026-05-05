"""Sandbox path resolution for connector send tools.

Connector tool methods annotate path parameters with :data:`SandboxPath`;
the SDK's dispatch wrapper resolves each value to its host-side
:class:`pathlib.Path` before the tool body runs.  Connector authors never
call the resolver directly.

Duplicated from :func:`aios.sandbox.volumes.resolve_to_host_path` rather
than imported so the reference SDK has zero aios-server dependency
(mirrors the ``_AIOS_NOTIFICATION_PREFIX`` precedent in base.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

# Trailing slash on _WORKSPACE_PREFIX is load-bearing — prevents
# ``/workspaces/foo`` matching ``/workspace``.
_WORKSPACE_PREFIX = "/workspace/"
_ATTACHMENTS_PREFIX = "/mnt/attachments/"
_ATTACHMENTS_HOST_SUBDIR = "_attachments"

_SANDBOX_PATH_DESCRIPTION = (
    "In-sandbox file path under /workspace/ or /mnt/attachments/ "
    "(/mnt/attachments/ is read-only — cp into /workspace/ first to "
    "forward an inbound photo)."
)


@dataclass(frozen=True, slots=True)
class _SandboxPathMarker:
    description: str = _SANDBOX_PATH_DESCRIPTION


SANDBOX_PATH_MARKER = _SandboxPathMarker()
SandboxPath = Annotated[Path, SANDBOX_PATH_MARKER]


def _resolve_sandbox_path(
    *,
    session_id: str,
    sandbox_path: str,
    workspace_root: Path,
) -> Path | None:
    """Map an in-sandbox path to its host equivalent, or ``None`` when it
    falls outside ``/workspace/`` and ``/mnt/attachments/`` or escapes the
    bind-mount root after ``..`` / symlink resolution."""
    if sandbox_path.startswith(_WORKSPACE_PREFIX):
        base = workspace_root / session_id
        suffix = sandbox_path[len(_WORKSPACE_PREFIX) :]
    elif sandbox_path == "/workspace":
        base = workspace_root / session_id
        suffix = ""
    elif sandbox_path.startswith(_ATTACHMENTS_PREFIX):
        base = workspace_root / _ATTACHMENTS_HOST_SUBDIR / session_id
        suffix = sandbox_path[len(_ATTACHMENTS_PREFIX) :]
    elif sandbox_path == "/mnt/attachments":
        base = workspace_root / _ATTACHMENTS_HOST_SUBDIR / session_id
        suffix = ""
    else:
        return None

    candidate = base if not suffix else base / suffix
    try:
        resolved = candidate.resolve(strict=False)
        resolved_base = base.resolve(strict=False)
    except OSError:
        return None
    if resolved != resolved_base and not resolved.is_relative_to(resolved_base):
        return None
    return resolved
