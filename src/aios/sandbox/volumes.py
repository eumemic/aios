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

from pathlib import Path

from aios.config import get_settings


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
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_workspace_path(raw_path: str) -> Path:
    """Resolve ``raw_path`` to an absolute ``Path``, creating it if needed."""
    path = Path(raw_path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


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
    here by :mod:`aios.harness.connector_supervisor` before the inbound
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
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_to_host_path(session_id: str, sandbox_path: str) -> Path | None:
    """Map an in-sandbox path to its host-side equivalent for known bind mounts.

    Returns ``None`` when:

    * ``sandbox_path`` doesn't resolve into ``/workspace`` or
      ``/mnt/attachments`` (e.g. ``/etc/hostname``, ``/mnt/memory/...``,
      ``/tmp/...``), or
    * the resolved candidate escapes the bind-mount root after ``..``
      normalization or symlink dereferencing.

    The containment check defends model-controlled callers (notably the
    image branch of :mod:`aios.tools.read`): without it, a path like
    ``/workspace/../../etc/hostname`` or a symlink at
    ``/workspace/sneaky.jpg`` pointing outside the bind mount would
    let the model read arbitrary host files the worker can access.
    """
    base, suffix = _bind_mount_base(session_id, sandbox_path)
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


def _bind_mount_base(session_id: str, sandbox_path: str) -> tuple[Path | None, str | None]:
    """Return the host base dir + remainder for ``sandbox_path``.

    ``(None, None)`` when the path doesn't fall under a known bind
    mount.  ``(base, None)`` when the path is exactly the root.
    """
    if sandbox_path == "/workspace":
        return workspace_dir_for(session_id), None
    if sandbox_path.startswith("/workspace/"):
        return workspace_dir_for(session_id), sandbox_path[len("/workspace/") :]
    if sandbox_path == "/mnt/attachments":
        return session_attachments_dir(session_id), None
    if sandbox_path.startswith("/mnt/attachments/"):
        return session_attachments_dir(session_id), sandbox_path[len("/mnt/attachments/") :]
    return None, None
