"""Sandbox path resolution for connector tools (#301).

Connector tool methods annotate path parameters with :data:`SandboxPath`;
the SDK's dispatcher resolves each value to its host-side
:class:`pathlib.Path` before the tool body runs.  Connector authors
never call the resolver directly.

The connector container must have ``<workspace_root>`` bind-mounted at
the same path as the worker process — set ``AIOS_WORKSPACE_ROOT`` to
match.  Without that mount, resolved paths exist in the resolver's
namespace but the bytes aren't reachable to the connector runtime.

Mirrors :func:`aios.sandbox.volumes.resolve_to_host_path` rather than
importing it: the SDK has zero aios-server dependency, and the path
mapping convention is small and stable.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

from aios_connector_http.mime import sniff_image_mime

# Trailing slash on _WORKSPACE_PREFIX is load-bearing — prevents
# ``/workspaces/foo`` matching ``/workspace``.
_WORKSPACE_PREFIX = "/workspace/"
_ATTACHMENTS_PREFIX = "/mnt/attachments/"
_ATTACHMENTS_HOST_SUBDIR = "_attachments"

_DEFAULT_WORKSPACE_ROOT = Path("/var/lib/aios/workspaces")

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


def workspace_root() -> Path:
    """Read ``AIOS_WORKSPACE_ROOT`` from env, falling back to the default.

    Same default as ``aios.config.Settings.workspace_root`` so a
    connector container with no override and a default-config worker
    line up without extra deployment work.
    """
    raw = os.environ.get("AIOS_WORKSPACE_ROOT")
    return Path(raw) if raw else _DEFAULT_WORKSPACE_ROOT


def resolve_sandbox_path(
    *,
    session_id: str,
    sandbox_path: str,
    root: Path | None = None,
) -> Path | None:
    """Map an in-sandbox path to its host equivalent, or ``None``.

    Returns ``None`` when ``sandbox_path`` falls outside ``/workspace/``
    and ``/mnt/attachments/``, or when the resolved path escapes the
    bind-mount root after ``..`` / symlink resolution.
    """
    base_root = root or workspace_root()
    if sandbox_path.startswith(_WORKSPACE_PREFIX):
        base = base_root / session_id
        suffix = sandbox_path[len(_WORKSPACE_PREFIX) :]
    elif sandbox_path == "/workspace":
        base = base_root / session_id
        suffix = ""
    elif sandbox_path.startswith(_ATTACHMENTS_PREFIX):
        base = base_root / _ATTACHMENTS_HOST_SUBDIR / session_id
        suffix = sandbox_path[len(_ATTACHMENTS_PREFIX) :]
    elif sandbox_path == "/mnt/attachments":
        base = base_root / _ATTACHMENTS_HOST_SUBDIR / session_id
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


_MAX_ATTACHMENT_BYTES = 5 * 1024 * 1024


class AttachmentError(ValueError):
    """Raised by :meth:`Attachment.as_params` when the host path is unreadable
    or exceeds the 5 MiB SDK boundary cap.

    Connector code that catches this can decide whether to skip the
    attachment, send a placeholder text message, or fail loudly.
    """


@dataclass(frozen=True, slots=True)
class Attachment:
    """Inbound attachment record handed to :meth:`HttpConnector.emit_inbound`.

    The connector hands aios a *host path* it owns; bytes never traverse
    HTTP.  The API process stages the file into the session's read-only
    ``/mnt/attachments`` bind mount before appending the inbound event.
    The connector container and the API process must share a filesystem
    view (typically a bind-mounted ``workspace_root``) so the staging
    rename can see the source.
    """

    host_path: str
    filename: str
    content_type: str

    def as_params(self) -> dict[str, Any]:
        """Validate the host path and return the JSON wire dict.

        Raises :class:`AttachmentError` when the file is missing or
        exceeds the 5 MiB cap.  Stat'ing here rather than at construction
        lets callers build :class:`Attachment` instances before their
        backing files are fully written.

        For image attachments, the declared ``content_type`` is
        reconciled against the actual magic bytes — Anthropic rejects
        mime-vs-magic mismatches, so the truth must reach the event log.
        """
        try:
            st = os.stat(self.host_path)
        except FileNotFoundError as err:
            raise AttachmentError(
                f"attachment host_path does not exist: {self.host_path!r}"
            ) from err
        if not stat.S_ISREG(st.st_mode):
            raise AttachmentError(f"attachment host_path is not a regular file: {self.host_path!r}")
        if st.st_size > _MAX_ATTACHMENT_BYTES:
            raise AttachmentError(
                f"attachment {self.filename!r} is {st.st_size} bytes; "
                f"SDK cap is {_MAX_ATTACHMENT_BYTES} bytes (5 MiB)."
            )
        content_type = self.content_type
        if content_type.startswith("image/"):
            with open(self.host_path, "rb") as f:
                sniffed = sniff_image_mime(f.read(16))
            if sniffed is not None:
                content_type = sniffed
        return {
            "host_path": self.host_path,
            "filename": self.filename,
            "content_type": content_type,
            "size": st.st_size,
        }
