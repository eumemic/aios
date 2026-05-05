"""Sandbox path resolution for connector send tools.

Connector tool methods that accept in-sandbox file paths annotate the
parameter with :data:`SandboxPath` (or ``list[SandboxPath]``).  The SDK's
dispatch wrapper (in :mod:`aios_connector.base`) resolves each value to
its host-side equivalent BEFORE the tool body runs, so the body receives
:class:`pathlib.Path` objects already containment-checked.  Connector
authors never call the resolver directly — that's the entire point of
the marker.

The schema published to the model surfaces ``{"type": "string"}`` (or
``{"type": "array", "items": {"type": "string"}}`` for the list form):
the model passes path strings, not pydantic-serialized ``Path`` objects.

Internal — :func:`_resolve_sandbox_path` is called by the dispatch wrapper
and tested directly; it is NOT re-exported from :mod:`aios_connector` and
is not part of the public API.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

# Trailing slash is load-bearing — prevents ``/workspaces/foo`` matching
# ``/workspace``.  Duplicated from :mod:`aios.sandbox.volumes` rather
# than imported so the reference SDK has zero dependency on aios server
# code (mirrors the ``_AIOS_NOTIFICATION_PREFIX`` precedent in base.py).
_WORKSPACE_PREFIX = "/workspace/"
_ATTACHMENTS_PREFIX = "/mnt/attachments/"
_ATTACHMENTS_HOST_SUBDIR = "_attachments"


# Description surfaced in the JSON Schema for every ``SandboxPath``
# parameter — single source of truth, so wording stays consistent across
# every connector's send tools.
_SANDBOX_PATH_DESCRIPTION = (
    "In-sandbox file path under /workspace/ or /mnt/attachments/ "
    "(/mnt/attachments/ is read-only — cp into /workspace/ first to "
    "forward an inbound photo)."
)


@dataclass(frozen=True, slots=True)
class _SandboxPathMarker:
    """Sentinel embedded in :data:`SandboxPath`'s ``Annotated`` metadata.

    Identity-comparable; the SDK detects it via ``isinstance`` rather than
    equality so the description string can change without breaking
    detection.
    """

    description: str = _SANDBOX_PATH_DESCRIPTION


SANDBOX_PATH_MARKER = _SandboxPathMarker()

# The public marker connector authors annotate with.  ``Annotated[Path,
# marker]`` keeps the parameter typed as ``Path`` for static checkers
# while letting the SDK's dispatch wrapper recognize the marker via
# ``typing.get_type_hints(..., include_extras=True)`` and auto-resolve
# the model-supplied string before the tool body runs.
SandboxPath = Annotated[Path, SANDBOX_PATH_MARKER]


def _resolve_sandbox_path(
    *,
    session_id: str,
    sandbox_path: str,
    workspace_root: Path,
) -> Path | None:
    """Map an in-sandbox path to its host-side equivalent, or ``None`` when
    it doesn't fall under ``/workspace/`` or ``/mnt/attachments/`` or escapes
    the bind-mount root after ``..`` / symlink resolution.

    Internal — the dispatch wrapper in :mod:`aios_connector.base` is the
    only caller.  Connector authors annotate parameters with
    :data:`SandboxPath` instead.

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
