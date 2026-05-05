"""Sandbox path resolution for connector send tools.

Connectors take a structured ``attachments: list[str]`` parameter on
their send tools (the model passes in-sandbox paths like
``/workspace/cat.jpg``).  This module's :func:`resolve_sandbox_path`
maps each one to its host-side equivalent so the connector can
``open()`` the file before uploading to the platform.
"""

from __future__ import annotations

from pathlib import Path

# Trailing slash is load-bearing — prevents ``/workspaces/foo`` matching
# ``/workspace``.  Duplicated from :mod:`aios.sandbox.volumes` rather
# than imported so the reference SDK has zero dependency on aios server
# code (mirrors the ``_AIOS_NOTIFICATION_PREFIX`` precedent in base.py).
_WORKSPACE_PREFIX = "/workspace/"
_ATTACHMENTS_PREFIX = "/mnt/attachments/"
_ATTACHMENTS_HOST_SUBDIR = "_attachments"


def resolve_sandbox_path(
    *,
    session_id: str,
    sandbox_path: str,
    workspace_root: Path,
) -> Path | None:
    """Map an in-sandbox path to its host-side equivalent, or ``None`` when
    it doesn't fall under ``/workspace/`` or ``/mnt/attachments/`` or escapes
    the bind-mount root after ``..`` / symlink resolution.

    Duplicated from :func:`aios.sandbox.volumes.resolve_to_host_path`
    rather than imported so the reference SDK has zero aios server
    dependency (mirrors the ``_AIOS_NOTIFICATION_PREFIX`` precedent).
    """
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
