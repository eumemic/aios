"""Per-session workspace directory conventions.

Each session gets a stable host-side directory at
``settings.workspace_root / session_id``, which is bind-mounted into the
session's container at ``/workspace``. The directory is created on the
first tool call that provisions a container for the session — chat-only
sessions never create one — and persists across container lifetimes
(container death, session re-wake, worker restart).

The directory is NOT deleted when the container goes away. A session can
be resumed tomorrow and still find its files. Cleanup of stale workspace
dirs is a Phase 6 polish item.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from aios.config import get_settings
from aios.errors import ForbiddenError
from aios.logging import get_logger

log = get_logger("aios.sandbox.volumes")

_UNSAFE_FILENAME_CHARS = re.compile(r"[^\w.\-]")
_MAX_FILENAME_LEN = 200
_FILENAME_FALLBACK = "unnamed"


def safe_filename(name: str | None) -> str:
    """Sanitize ``name`` for use as a path leaf.

    Strips directory separators (defeats ``../`` traversal), maps
    unsupported characters to ``_``, falls back to ``"unnamed"`` for
    None / empty / all-dot inputs, and caps length so a pathological
    filename combined with a per-file id prefix can't exhaust the
    host FS's per-component limit.

    Unicode-aware: Python's ``\\w`` matches the full Unicode word class
    by default, so non-ASCII letters are preserved (e.g.
    ``café.jpg``, ``图片.png``). Only structurally unsafe punctuation
    and whitespace get rewritten.
    """
    if not name:
        return _FILENAME_FALLBACK
    base = os.path.basename(name)
    cleaned = _UNSAFE_FILENAME_CHARS.sub("_", base)
    if not cleaned or cleaned.replace(".", "") == "":
        return _FILENAME_FALLBACK
    return cleaned[:_MAX_FILENAME_LEN]


def ensure_owned_dir(path: Path) -> Path:
    """mkdir(parents=True, exist_ok=True) ``path``, then — iff running as root —
    chown every component newly created by THIS call (intermediates included) to
    the configured workspaces owner uid:gid. Non-root callers (api at uid 1000,
    dev single-uid) get exactly today's plain-mkdir behavior.

    Fixes #959: the worker (root) and api (uid 1000) both write under
    ``workspace_root``. A shared dir the worker creates first is ``root:root``
    and the api (no CAP_CHOWN) can't write into it. Chowning the components
    this call creates keeps the shared tree writable by the api owner.
    """
    settings = get_settings()
    root = settings.workspace_root.resolve()
    target = path.resolve()
    newly: list[Path] = []
    cur = target
    while True:
        if cur.exists():
            break
        newly.append(cur)
        if cur == root or cur.parent == cur:
            break
        cur = cur.parent
    path.mkdir(parents=True, exist_ok=True)
    if os.geteuid() == 0:
        uid, gid = settings.workspaces_owner_uid, settings.workspaces_owner_gid
        for component in newly:
            # Only chown within the workspace tree (workspace_root itself and its
            # descendants); never touch out-of-tree ancestors. If a caller passed
            # a path outside the tree (the mkdir honors any path — caller's
            # contract), the component-walk collected out-of-tree ancestors; do
            # NOT chown those.
            if not component.is_relative_to(root):
                continue
            # A racing provision may have created+chowned this component first
            # (benign), but the failure must stay observable per CLAUDE.md's
            # no-silent-error stance — log and continue rather than crash.
            try:
                # lchown (not chown): closes the mkdir→chown symlink-swap race —
                # a container with workspace write access could replace a
                # freshly-created component with a symlink before we chown,
                # redirecting os.chown to an out-of-tree target. Matches the
                # repair path's lchown.
                os.lchown(component, uid, gid)
            except OSError as e:
                log.warning("workspace.chown_failed", path=str(component), error=str(e))
    return path


# DEPRECATED post-#409 — do not use in new code; see issue #630.
def workspace_dir_for(session_id: str) -> Path:
    """Return the absolute host directory for ``session_id``'s workspace.

    The returned path is always absolute — Docker bind mounts reject
    relative paths. If ``workspace_root`` was configured as a relative
    path (e.g. ``./workspaces`` in a dev ``.env``), it is resolved
    against the current working directory at call time.

    Pure — does not touch the filesystem. Use :func:`ensure_workspace_dir`
    to both compute and create.
    """
    return (get_settings().workspace_root / session_id).resolve()


def ensure_workspace_dir(session_id: str) -> Path:
    """Return the absolute host directory for ``session_id``, creating it if needed.

    Also ensures the parent ``workspace_root`` exists. ``parents=True,
    exist_ok=True`` semantics — safe to call repeatedly.
    """
    path = workspace_dir_for(session_id)
    return ensure_owned_dir(path)


def ensure_workspace_path(raw_path: str) -> Path:
    """Resolve ``raw_path`` to an absolute ``Path``, creating it if needed."""
    path = Path(raw_path).resolve()
    return ensure_owned_dir(path)


def validate_workspace_path(
    raw_path: str, account_id: str, *, session_id: str | None = None
) -> None:
    """Refuse ``raw_path`` if it resolves outside the account's
    workspace subdirectory.

    ``raw_path`` MUST be absolute.  Relative inputs are rejected
    before any ``Path.resolve()`` runs: ``resolve()`` would
    interpret them against the current process's working directory,
    which differs between the API and worker (and between worker
    restarts), producing diverging targets across boundaries.  See
    #626 — legacy session rows persisted relative
    ``workspaces/<account>/<session>`` strings (back when
    ``AIOS_WORKSPACE_ROOT`` itself was permitted to be relative);
    every cold-start re-validation surfaced ``ForbiddenError``
    blamed on whatever path the model had just tried to use.
    Failing fast on the relative-input branch produces an
    unambiguous error identifying the stored
    ``workspace_volume_path`` as the culprit.

    Without this check an authenticated client could POST
    ``/v1/sessions`` with e.g. ``workspace_path="/etc"`` and the
    sandbox would bind-mount the host's ``/etc`` read-write at
    ``/workspace`` — arbitrary host filesystem read/write via any
    bash / write / edit tool call.  A path under another account's
    subdir (``workspace_root/{other_account_id}/...``) would defeat
    the per-account-subdir isolation ``insert_session`` enforces by
    default.

    ``Path.resolve()`` collapses ``..`` traversal and dereferences
    symlinks on the supplied path before the ``is_relative_to``
    check, so create-time inputs that already point outside the jail
    are rejected — including ``/etc``, ``..``-traversal back up to
    ``workspace_root/{other_account_id}``, and symlinks under the
    account's subdir that point outward at validate time.

    ``session_id`` opens a tiny backward-compat carve-out for the
    pre-#409 default ``<workspace_root>/<session_id>`` (no per-tenant
    subdir).  Sessions created before PR #409 have ``workspace_volume_path``
    rows in exactly that shape; without the carve-out the bind-mount
    boundary re-check (added by PR #590) rejects them on every
    cold-start, leaving the model staring at a ``ForbiddenError`` on
    every tool call.  The carve-out is keyed on the session_id the
    caller is currently provisioning — a path matching the legacy shape
    but naming a *different* session_id is still rejected, so the
    cross-tenant defense holds.  The carve-out also requires the
    resolved path to remain under ``workspace_root``: a symlink at
    ``<workspace_root>/<session_id>`` pointing at ``/etc`` (or any
    other path outside the jail) is rejected, preserving the
    host-FS-escape defense the strict branch provides.  The create-time
    call sites leave ``session_id`` unset so user-supplied paths remain
    strictly jailed to the account subdir.

    Limitations: this is the create-time + bind-mount-time check on
    the workspace_path argument.  Symlinks WRITTEN inside the
    mounted ``/workspace`` after the bind-mount is live still resolve
    on the host filesystem at access time (Docker bind-mount semantic),
    so a tool ``ln -s /etc /workspace/sneaky`` followed by
    ``cat /workspace/sneaky/passwd`` still reads host ``/etc/passwd``.
    Fencing that surface requires kernel-level mount options
    (``nosymfollow``) or container-level MAC; out of scope for this
    fix.

    Raises ``ForbiddenError`` (403, not 422): semantically this is an
    attempted privilege escalation per the project's "fail hard" stance,
    and surfacing it under the auth-tier error family makes it visible
    in audit logs as such.
    """
    if not Path(raw_path).is_absolute():
        raise ForbiddenError(
            "workspace_volume_path must be absolute (starts with '/'); got "
            f"non-absolute value {raw_path!r}. This usually indicates a "
            "stale pre-#409 session row that needs the absolute-legacy "
            "backfill migration (see aios#626).",
            detail={"workspace_path": raw_path, "session_id": session_id},
        )
    path = Path(raw_path).resolve()
    workspace_root = get_settings().workspace_root.resolve()
    account_root = (workspace_root / account_id).resolve()
    if path.is_relative_to(account_root):
        return
    if session_id is not None:
        legacy_path = (workspace_root / session_id).resolve()
        if path == legacy_path and legacy_path.is_relative_to(workspace_root):
            return
    raise ForbiddenError(
        "workspace_path must resolve to within the account's workspace subdirectory",
        detail={"workspace_path": raw_path},
    )


_RUNS_ROOT = "_runs"


def run_workspace_dir(run_id: str) -> Path:
    """Per-run host workspace directory backing ``/workspace`` in a run sandbox (#988).

    Rooted at ``<workspace_root>/_runs/<run_id>`` — under a reserved prefix so it
    never collides with a session workspace (``<workspace_root>/<account_id>/…``)
    and is independent of any user-supplied path. The run sandbox is **ephemeral
    scratch**: the directory is created at provision and is not part of any durable
    snapshot machinery (that is M2). Pure — does not touch the filesystem; use
    :func:`ensure_run_workspace_dir` to create.
    """
    return (get_settings().workspace_root / _RUNS_ROOT / run_id).resolve()


def ensure_run_workspace_dir(run_id: str) -> Path:
    """Return the per-run workspace directory, creating it (and its parents) if needed."""
    return ensure_owned_dir(run_workspace_dir(run_id))


_MEMORY_STORES_ROOT = "_memory_stores"


def memory_stores_root() -> Path:
    """Return ``<workspace_root>/_memory_stores`` — the parent of all
    shared memory-store host directories.

    Per-store host dirs (one per ``memory_store.id``) live as siblings in
    here and are bind-mounted into every attached session's container at
    ``/mnt/memory/<store_name>/``. Sharing the source dir across attached
    sessions is what makes cross-session reads live: a tool write from
    session A appears in session B's mount immediately.
    """
    return (get_settings().workspace_root / _MEMORY_STORES_ROOT).resolve()


def memory_store_host_dir(store_id: str) -> Path:
    """Return the shared host-side directory backing memory store ``store_id``.

    Pure — does not create the directory. Materialization is handled by
    :mod:`aios.sandbox.memory_mounts`, which acquires the matching lock
    file (see :func:`memory_store_lock_path`) before populating from DB.
    """
    return memory_stores_root() / store_id


def memory_store_lock_path(store_id: str) -> Path:
    """Return the file-lock path used to serialize first-attach materialization
    of ``store_id``.

    Two sessions provisioning concurrently for the same store both call
    :func:`materialize_store_to_host`; the lock ensures only one of them
    writes the initial DB snapshot to the host dir. The loser observes
    the ``.materialized`` marker and skips."""
    return memory_stores_root() / f"{store_id}.lock"


_GITHUB_REPOS_ROOT = "_github_repos"


def github_repos_cache_root() -> Path:
    """Return ``<workspace_root>/_github_repos`` — the parent of all
    cache-bare-clone host directories.

    Each cache dir is keyed by ``sha256(repo_url)`` so two sessions that
    reference the same upstream repo share the object database, regardless
    of ``mount_path``. Per-session working trees ``--reference`` this cache
    via :func:`session_repo_working_tree_dir`.
    """
    return (get_settings().workspace_root / _GITHUB_REPOS_ROOT).resolve()


def github_repo_cache_dir(url_hash: str) -> Path:
    """Bare-clone host dir for a given repo-url hash. Pure — see
    :mod:`aios.sandbox.github_clone` for materialization."""
    return github_repos_cache_root() / url_hash


def github_repo_cache_lock_path(url_hash: str) -> Path:
    """Per-cache file lock path. Two sessions racing on first-clone of the
    same url need to serialize so we don't run two ``git clone --bare``
    side by side and corrupt the cache dir."""
    return github_repos_cache_root() / f"{url_hash}.lock"


_SESSION_REPOS_ROOT = "_session_repos"


def session_repos_root(session_id: str) -> Path:
    """Per-session host dir for github_repository working trees.

    Rooted at ``<workspace_root>/_session_repos/<session_id>`` so the
    location is independent of any user-supplied ``workspace_path``
    override on the session — those overrides are user-managed and we
    don't want plaintext-token ``.git/config`` files leaking into them.
    """
    return (get_settings().workspace_root / _SESSION_REPOS_ROOT / session_id).resolve()


def session_repo_working_tree_dir(session_id: str, repo_id: str) -> Path:
    """Per-session working tree for a single ``github_repository``
    attachment. Bind-mounted into the container at the user-supplied
    ``mount_path``.
    """
    return session_repos_root(session_id) / repo_id


_ATTACHMENTS_ROOT = "_attachments"


def attachments_root() -> Path:
    """Return ``<workspace_root>/_attachments`` — the parent of all
    per-session inbound attachment directories.

    Each session subdir is bind-mounted read-only into its container at
    ``/mnt/attachments`` (see :mod:`aios.sandbox.provisioner`). Inbound
    binary blobs (Signal photos, Telegram voice notes, etc.) are staged
    here by :mod:`aios.services.attachment_staging` from
    :func:`aios.services.inbound.handle_inbound` before the inbound
    event is appended; the model sees them at stable in-sandbox paths
    of the form ``/mnt/attachments/<connector>/<event-ulid>-<filename>``.
    """
    return (get_settings().workspace_root / _ATTACHMENTS_ROOT).resolve()


def session_attachments_dir(session_id: str) -> Path:
    """Per-session host directory backing ``/mnt/attachments``.

    Pure — does not create the directory. Use
    :func:`ensure_session_attachments_dir` to create.
    """
    return attachments_root() / session_id


def ensure_session_attachments_dir(session_id: str) -> Path:
    """Return the per-session attachments directory, creating it if needed.

    Called eagerly from the provisioner at every container start so the
    bind-mount source always exists before Docker tries to mount it,
    even for sessions that have never received an attachment.
    """
    path = session_attachments_dir(session_id)
    return ensure_owned_dir(path)


_UPLOADS_ROOT = "_uploads"


def uploads_root() -> Path:
    """Return ``<workspace_root>/_uploads`` — the parent of all per-session
    upload directories.

    Each session subdir is bind-mounted read-only into its container at
    ``/mnt/uploads`` (see :mod:`aios.sandbox.spec`). Bytes uploaded via
    ``POST /v1/sessions/<id>/files`` land here under
    ``<workspace_root>/_uploads/<session_id>/<file_id>/<filename>``; the
    model sees them at ``/mnt/uploads/<file_id>/<filename>``.
    """
    return (get_settings().workspace_root / _UPLOADS_ROOT).resolve()


def session_uploads_dir(session_id: str) -> Path:
    """Per-session host directory backing ``/mnt/uploads``.

    Pure — does not create the directory. Use
    :func:`ensure_session_uploads_dir` to create.
    """
    return uploads_root() / session_id


def ensure_session_uploads_dir(session_id: str) -> Path:
    """Return the per-session uploads directory, creating it if needed.

    Called eagerly from the provisioner at every container start so the
    bind-mount source always exists before Docker tries to mount it,
    even for sessions that have never received an upload.
    """
    path = session_uploads_dir(session_id)
    return ensure_owned_dir(path)


def resolve_to_host_path(
    session_id: str,
    sandbox_path: str,
    *,
    workspace_path: Path | None = None,
) -> Path | None:
    """Map an in-sandbox path to its host-side equivalent for known bind mounts.

    Returns ``None`` when:

    * ``sandbox_path`` doesn't resolve into ``/workspace`` or
      ``/mnt/attachments`` (e.g. ``/etc/hostname``, ``/mnt/memory/...``,
      ``/tmp/...``), or
    * the resolved candidate escapes the bind-mount root after ``..``
      normalization or symlink dereferencing.

    ``workspace_path`` is the actual host-side bind-mount source for
    ``/workspace`` (recorded on the session row as
    ``workspace_volume_path``).  Post-PR-#409 it lives at
    ``<workspace_root>/<account_id>/<session_id>``; pre-#409 sessions
    had it at ``<workspace_root>/<session_id>``.  Either way, the
    caller is the authority — when ``workspace_path`` is ``None``,
    ``/workspace*`` paths fail closed (return ``None``) rather than
    silently falling back to a synthetic path that might no longer be
    the bind-mount source.  The ``/mnt/attachments`` and
    ``/mnt/uploads`` branches ignore ``workspace_path`` — their
    locations are derived from ``session_id`` alone (see
    :func:`session_attachments_dir` / :func:`session_uploads_dir`).

    The containment check defends model-controlled callers (notably the
    image branch of :mod:`aios.tools.read`): without it, a path like
    ``/workspace/../../etc/hostname`` or a symlink at
    ``/workspace/sneaky.jpg`` pointing outside the bind mount would
    let the model read arbitrary host files the worker can access.
    """
    base, suffix = _bind_mount_base(session_id, sandbox_path, workspace_path=workspace_path)
    if base is None:
        return None
    candidate = base if suffix is None else base / suffix
    try:
        resolved = candidate.resolve(strict=False)
        resolved_base = base.resolve(strict=False)
    except OSError:
        return None
    if resolved != resolved_base and not resolved.is_relative_to(resolved_base):
        return None
    return resolved


def _bind_mount_base(
    session_id: str,
    sandbox_path: str,
    *,
    workspace_path: Path | None = None,
) -> tuple[Path | None, str | None]:
    """Return the host base dir + remainder for ``sandbox_path``.

    ``(None, None)`` when the path doesn't fall under a known bind
    mount, or when ``sandbox_path`` is under ``/workspace`` and
    ``workspace_path`` is ``None`` (fail-closed).  ``(base, None)``
    when the path is exactly the root.
    """
    if sandbox_path == "/workspace":
        if workspace_path is None:
            return None, None
        return workspace_path.resolve(), None
    if sandbox_path.startswith("/workspace/"):
        if workspace_path is None:
            return None, None
        return workspace_path.resolve(), sandbox_path[len("/workspace/") :]
    if sandbox_path == "/mnt/attachments":
        return session_attachments_dir(session_id), None
    if sandbox_path.startswith("/mnt/attachments/"):
        return session_attachments_dir(session_id), sandbox_path[len("/mnt/attachments/") :]
    if sandbox_path == "/mnt/uploads":
        return session_uploads_dir(session_id), None
    if sandbox_path.startswith("/mnt/uploads/"):
        return session_uploads_dir(session_id), sandbox_path[len("/mnt/uploads/") :]
    return None, None
