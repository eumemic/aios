"""Sandbox path resolution for connector tools (#301).

Connector tool methods annotate path parameters with :data:`SandboxPath`;
the SDK's dispatcher resolves each value to its host-side
:class:`pathlib.Path` before the tool body runs.  Connector authors
never call the resolver directly.

The connector container must share a filesystem view with the worker
process so the resolved host paths are reachable from inside the
connector runtime.  For ``/workspace*`` the bind-mount source is
session-specific and carried on each tool-call event as
``workspace_path``; for ``/mnt/attachments*`` it's derived from
``AIOS_WORKSPACE_ROOT`` and ``session_id``.

Mirrors :func:`aios.sandbox.volumes.resolve_to_host_path` rather than
importing it: the SDK has zero aios-server dependency.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import stat
import unicodedata
from collections.abc import Callable
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
    workspace_path: Path | None = None,
    root: Path | None = None,
) -> Path | None:
    """Map an in-sandbox path to its host equivalent, or ``None``.

    Returns ``None`` when ``sandbox_path`` falls outside ``/workspace/``
    and ``/mnt/attachments/``, or when the resolved path escapes the
    bind-mount root after ``..`` / symlink resolution.

    ``workspace_path`` is the host-side bind-mount source for
    ``/workspace`` (the session row's ``workspace_volume_path``).  The
    worker is the authority — when ``workspace_path`` is ``None``,
    ``/workspace*`` paths fail closed (return ``None``) rather than
    silently falling back to a synthetic path that might not be the
    bind-mount source.
    """
    base_root = root or workspace_root()
    if sandbox_path.startswith(_WORKSPACE_PREFIX):
        if workspace_path is None:
            return None
        base = workspace_path
        suffix = sandbox_path[len(_WORKSPACE_PREFIX) :]
    elif sandbox_path == "/workspace":
        if workspace_path is None:
            return None
        base = workspace_path
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
    except (OSError, ValueError):
        # ValueError covers embedded NUL bytes (``Path('foo\x00bar')``);
        # OSError covers symlink-loop / path-too-long. Both are model-
        # induced garbage — fail closed.
        return None
    if resolved != resolved_base and not resolved.is_relative_to(resolved_base):
        return None
    return resolved


MAX_ATTACHMENT_BYTES = 5 * 1024 * 1024
# Alias kept for existing internal / test references.  ``as_params`` and
# the marker helpers read the public name at call time so a future cap
# change (or a monkeypatched cap in a test) flows through everywhere.
_MAX_ATTACHMENT_BYTES = MAX_ATTACHMENT_BYTES


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
        if st.st_size > MAX_ATTACHMENT_BYTES:
            raise AttachmentError(
                f"attachment {self.filename!r} is {st.st_size} bytes; "
                f"SDK cap is {MAX_ATTACHMENT_BYTES} bytes (5 MiB)."
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


# ── undelivered-attachment markers (#2551) ─────────────────────────────
#
# When a connector cannot deliver an inbound attachment (over the cap,
# signal-cli download failed, unreadable, ...), a one-line marker takes
# its place in the inbound ``content`` so the agent learns a file was
# dropped instead of reasoning from an empty event.  The marker is
# advisory, untrusted text: a sender can type a lookalike into the body.
# Sanitising only guarantees that sender-controlled metadata (filename,
# MIME type) cannot break a *real* marker's single-line structure.

FALLBACK_MARKER = "[attachment not delivered]"

_SIZE_RE = re.compile(r"[0-9]+")
_STRIPPED_CATEGORIES = frozenset({"Cc", "Cf", "Zl", "Zp"})
_QUOTE_BRACKET_CHARS = frozenset(
    "'\"[]\u2018\u2019\u201a\u201b\u201c\u201d\u2039\u203a\u3010\u3011\u300c\u300d"
)
_TYPE_RE = re.compile(r"[\w.+-]+/[\w.+-]+", re.ASCII)
_NAME_MAX = 120
_TYPE_MAX = 100


def coerce_size(value: object) -> int | None:
    """Coerce a platform-declared size to a non-negative ``int`` or ``None``.

    ``bool`` and every ``float`` map to ``None``; ``str`` is accepted only
    when it is ASCII digits.  Never raises.
    """
    try:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value if value >= 0 else None
        if isinstance(value, str):
            return int(value) if _SIZE_RE.fullmatch(value) else None
        return None
    except Exception:
        return None


def format_size(n: int) -> str:
    """Human-readable binary size: ``1023 B``, ``1 KiB``, ``7.1 MiB``."""
    if n < 1024:
        return f"{n} B"
    v = n / 1024
    unit = "KiB"
    if round(v, 1) >= 1024:
        v = n / 2**20
        unit = "MiB"
        if round(v, 1) >= 1024:
            v = n / 2**30
            unit = "GiB"
    text = f"{v:.1f}"
    if text.endswith(".0"):
        text = text[:-2]
    return f"{text} {unit}"


def is_failed_download(st_size: int | None, declared_size: int | None) -> bool:
    """A 0-byte host file whose platform declared a size > 0 is a drop."""
    try:
        st = coerce_size(st_size)
        declared = coerce_size(declared_size)
        return st == 0 and declared is not None and declared > 0
    except Exception:
        return False


def is_over_cap(size: int | None) -> bool:
    """Single source of truth for "over the cap" (cap read at call time)."""
    try:
        coerced = coerce_size(size)
        return coerced is not None and coerced > MAX_ATTACHMENT_BYTES
    except Exception:
        return False


def _safe_name(name: object) -> str:
    try:
        s = str(name) if name is not None else ""
    except Exception:
        s = "unnamed"
    s = unicodedata.normalize("NFKC", s)
    s = "".join("_" if unicodedata.category(ch) in _STRIPPED_CATEGORIES else ch for ch in s)
    s = "".join("_" if ch in _QUOTE_BRACKET_CHARS else ch for ch in s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > _NAME_MAX:
        s = s[: _NAME_MAX - 1] + "\u2026"
    return s or "unnamed"


def _safe_type(content_type: object) -> str:
    if not isinstance(content_type, str):
        return "unknown type"
    t = unicodedata.normalize("NFKC", content_type)
    # Drop MIME parameters (``; charset=...``); only plain spaces are
    # trimmed so a smuggled newline still fails the full match below.
    t = t.split(";", 1)[0].strip(" ").lower()
    if len(t) > _TYPE_MAX or not _TYPE_RE.fullmatch(t):
        return "unknown type"
    return t


def safe_marker(name: object, content_type: object) -> tuple[str, str]:
    """Sanitise sender-controlled ``(name, content_type)`` for a marker.

    The results contain no line break, no Cc/Cf/Zl/Zp code point, and no
    quote or bracket that could close the marker.  Never raises.
    """
    try:
        safe_name = _safe_name(name)
    except Exception:
        safe_name = "unnamed"
    try:
        safe_type = _safe_type(content_type)
    except Exception:
        safe_type = "unknown type"
    return safe_name, safe_type


def _size_vs_limit(size: int, limit: int) -> tuple[str, str]:
    rendered_size, rendered_limit = format_size(size), format_size(limit)
    if rendered_size == rendered_limit:
        # No-contradiction rule: "5 MiB > 5 MiB limit" reads as a bug.
        return f"{size:,} bytes", f"{limit:,} bytes"
    return rendered_size, rendered_limit


def rejection_marker(
    name: object,
    content_type: object,
    size: int | None,
    *,
    failed_download: bool = False,
    over_cap: bool | None = None,
) -> str:
    """Build the one-line marker for an attachment that was not delivered."""
    try:
        safe_name, safe_type = safe_marker(name, content_type)
        coerced = coerce_size(size)
        over = is_over_cap(coerced) if over_cap is None else bool(over_cap)
        if coerced is None:
            over = False
        head = f"[attachment not delivered: '{safe_name}', {safe_type}"
        if over:
            assert coerced is not None
            rendered_size, rendered_limit = _size_vs_limit(coerced, MAX_ATTACHMENT_BYTES)
            body = f"{head}, {rendered_size} > {rendered_limit} limit"
            return f"{body}: download failed]" if failed_download else f"{body}]"
        sized = head if coerced is None else f"{head}, {format_size(coerced)}"
        reason = "download failed" if failed_download else "rejected"
        return f"{sized}: {reason}]"
    except Exception:
        return FALLBACK_MARKER


def append_markers(text: str | None, markers: list[str]) -> str:
    """Append ``markers`` (one per line) after the original ``text``."""
    if not markers:
        return text or ""
    return "\n".join([t for t in [text] if t] + list(markers))


def _emit(logger: Any, event: str, **fields: Any) -> None:
    """Log ``event`` with ``fields`` at warning level; never raises.

    stdlib loggers get ``extra=`` (field names avoid reserved
    ``LogRecord`` attributes such as ``name``); structlog-style loggers
    get keyword arguments, falling back to a formatted message.
    """
    try:
        if isinstance(logger, (logging.Logger, logging.LoggerAdapter)):
            logger.warning(event, extra=fields)
            return
        try:
            logger.warning(event, **fields)
        except TypeError:
            logger.warning("%s %s", event, fields)
    except Exception:
        with contextlib.suppress(Exception):
            logging.getLogger("aios_connector_http").warning("%s %r", event, fields)


def drop_attachment(
    markers: list[str],
    logger: Any,
    event_prefix: str,
    *,
    name: object,
    content_type: object,
    size: int | None,
    reason: str,
    failed_download: bool = False,
    over_cap: bool | None = None,
    builder: Callable[..., str] | None = None,
    **fields: Any,
) -> None:
    """Record one undelivered attachment: append a marker and log it.

    The only place connectors append markers.  Never raises: a builder
    failure appends :data:`FALLBACK_MARKER` and logs
    ``<prefix>.inbound.attachment_marker_failed``, so a drop can never
    become silent.
    """
    build = builder or rejection_marker
    try:
        marker = build(name, content_type, size, failed_download=failed_download, over_cap=over_cap)
        if not isinstance(marker, str) or not marker:
            raise ValueError("empty marker")
    except Exception as err:
        marker = FALLBACK_MARKER
        _emit(
            logger,
            f"{event_prefix}.inbound.attachment_marker_failed",
            reason=reason,
            error_type=type(err).__name__,
        )
    with contextlib.suppress(Exception):
        markers.append(marker)
    try:
        attachment_name = safe_marker(name, content_type)[0]
    except Exception:
        attachment_name = "unnamed"
    _emit(
        logger,
        f"{event_prefix}.inbound.attachment_not_delivered",
        reason=reason,
        size=size,
        attachment_name=attachment_name,
        marker_emitted=True,
        **fields,
    )
